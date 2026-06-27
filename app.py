import streamlit as st
from google import genai
from google.genai import types
import pandas as pd
import json
import time
import plotly.express as px
import requests
from pydantic import BaseModel

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pick For Me", page_icon="⚖️", layout="centered")

# --- API CONFIG ---
api_key = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)

def clean_llm_json(raw_text):
    """Strips markdown formatting so json.loads() doesn't crash."""
    cleaned = raw_text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)

def call_gemini_with_retry(prompt, config, max_retries=5): 
    """Tries to call Gemini, waiting and retrying if Google's servers are overloaded."""
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=config
            )
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⚠️ Caught 503 error. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            raise e

# --- UI SKELETON ---
st.title("⚖️ Pick For Me")
st.success("Environment and API configured successfully!")

st.divider()
st.header("🔍 Search Live Market")

user_search = st.text_input(
    "What are you looking for?", 
    value=""
)

if not user_search.strip():
    st.info("👋 Welcome to Pick For Me! Enter a product or venue category above to get started.")
    st.stop()

# --- BUG FIX 2 & 4: INTENT ROUTER ---
@st.cache_data(ttl=3600)
def classify_search_intent(search_term):
    prompt = f"""
    Analyze this search term: '{search_term}'.
    Return a strict JSON object with two keys:
    1. 'domain': Either 'retail' (for purchasable goods) or 'local_business' (for venues, restaurants, services).
    2. 'is_apparel': Boolean true ONLY if it is clothing or footwear that strictly requires gender and size variations.
    Format: {{"domain": "retail", "is_apparel": true}}
    """
    try:
        response = call_gemini_with_retry(
            prompt, 
            types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
        )
        return clean_llm_json(response.text)
    except Exception:
        return {"domain": "retail", "is_apparel": False}

with st.spinner("Analyzing intent..."):
    intent = classify_search_intent(user_search)

# --- BUG FIX 4: DEMOGRAPHIC GATEKEEPER ---
demographic_query = ""
if intent.get("is_apparel"):
    st.warning("👕 **Apparel Detected:** Please specify sizing to filter out unpurchasable inventory.")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Target Demographic:", ["", "Men", "Women", "Unisex", "Kids"])
    with col2:
        sizes = st.multiselect("Acceptable Sizes:", ["6", "7", "8", "9", "10", "11", "12", "S", "M", "L", "XL"])
    
    if not gender or not sizes:
        st.stop() # Halt execution until filled
    demographic_query = f"{gender} size {','.join(sizes)}"

st.divider()
st.header("1. Hard Constraints & Dealbreakers")
max_budget = st.number_input("Maximum Budget ($)", min_value=0.0, value=0.0, step=50.0, help="Set to 0 for no budget limit.")
unstructured_constraints = st.text_area("Specific Requirements (Optional)")

combined_search_query = f"{user_search} {demographic_query} {unstructured_constraints}".strip()

# --- BUG FIX 2: DECOUPLED FETCHER ---
@st.cache_data(ttl=3600)
def fetch_dynamic_data(search_term, domain):
    if "SERPAPI_KEY" not in st.secrets:
        st.error("Missing SERPAPI_KEY in secrets.toml!")
        st.stop()
        
    engine = "google_local" if domain == "local_business" else "google_shopping"
    params = {
        "engine": engine,
        "q": search_term,
        "api_key": st.secrets["SERPAPI_KEY"],
        "hl": "en",
        "gl": "us",
    }

    try:
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        data = response.json()
        
        dynamic_items = []
        seen_names = {} 
        
        # Route parser based on domain
        if domain == "local_business":
            results = data.get("local_results", [])[:40]
            for item in results:
                base_name = item.get("title", "Unknown Venue")[:75]
                final_name = f"{base_name} ({seen_names[base_name]})" if base_name in seen_names else base_name
                seen_names[base_name] = seen_names.get(base_name, 0) + 1
                
                dynamic_items.append({
                    "name": final_name,
                    "price_usd": 0.0, 
                    "price_str": item.get("price", "N/A"), # <-- ADD THIS LINE to grab the $$ string
                    "link": item.get("website", item.get("links", {}).get("website", "")),
                    "seller": item.get("address", "Unknown Address") 
                })
        else:
            results = data.get("shopping_results", [])[:40]
            for item in results:
                base_name = item.get("title", "Unknown Product")[:75]
                final_name = f"{base_name} ({seen_names[base_name]})" if base_name in seen_names else base_name
                seen_names[base_name] = seen_names.get(base_name, 0) + 1
                
                dynamic_items.append({
                    "name": final_name,
                    "price_usd": float(item.get("extracted_price", 0.0)),
                    "link": item.get("product_link", item.get("link", "")),
                    "seller": item.get("source", "Unknown") 
                })
        return dynamic_items
    except Exception as e:
        st.error(f"Failed to fetch live data: {e}")
        return []

with st.spinner(f"Scouring the web for '{combined_search_query}'..."):
    raw_products = fetch_dynamic_data(combined_search_query, intent.get("domain"))

# --- PHASE 1: THE LLM BOUNCER ---
@st.cache_data(ttl=3600)
def filter_dealbreakers(product_list, budget, constraints_text):
    budget_filtered = [p for p in product_list if budget <= 0 or float(p.get("price_usd", 0) or 0) <= budget]
        
    if not constraints_text.strip() or not budget_filtered:
        return budget_filtered
    
    simple_products = [{"id": i, "name": p["name"]} for i, p in enumerate(budget_filtered)]
    
    filter_prompt = f"""
    You are a product filtering agent.
    Evaluate this list of items: {simple_products}. 
    The user's strict dealbreakers are: '{constraints_text}'. 
    Return ONLY a JSON array of the integer IDs for items that pass this initial screening. 
    
    FILTERING RULES:
    1. OBVIOUS CONTRADICTIONS: Delete any item that directly violates a clear constraint (e.g., straight cable when right-angle is required).
    2. THE CABLE & VIDEO PROTOCOL RULE: If the user explicitly requires video display, monitor, or 4K support, you MUST eliminate items whose titles indicate they are purely for mobile phones or basic charging (e.g., 'iPhone Charger', 'Fast Charging Cable', 'Power Cord', 'Charging Cable'). You may ONLY let a cable pass if its title explicitly includes video/high-bandwidth indicators like 'Display', 'Video', '4K', '8K', 'Monitor', 'DisplayPort', 'Alt Mode', '10Gbps', '20Gbps', or '40Gbps'. Pure charging cables cannot drive monitors.
    3. COMPLETE EVALUATION: Evaluate every single item in the provided list. Do not truncate.
    """
    try:
        response = call_gemini_with_retry(
            prompt=filter_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        valid_ids = json.loads(response.text)
        
        # --- THE FAIL-SAFE ---
        # If the bouncer kills everything because of bad retail titles, bypass it.
        if not valid_ids:
            st.warning("⚠️ Phase 1 Bouncer was too strict (titles lacked specs). Bypassing to Phase 3.")
            return budget_filtered
            
        return [budget_filtered[i] for i in valid_ids if i < len(budget_filtered)]
        
    except Exception as e:
        print(f"❌ Filter Error: {e}")
        return budget_filtered

with st.spinner("Screening options against your dealbreakers..."):
    surviving_products = filter_dealbreakers(raw_products, max_budget, unstructured_constraints)

if not surviving_products:
    st.warning("⚠️ No options matched your strict constraints.")
    st.stop()

# --- PHASE 2: STRICT QUANTITATIVE CRITERIA ---
@st.cache_data(ttl=3600)
def generate_dropdown_options(category: str, domain: str) -> dict:
    system_prompt = f"""
    You are an analytics engine. 
    Suggest 10 distinct, quantifiable decision criteria a user should consider when evaluating a '{category}'.
    Context: This is a '{domain}' (e.g. if local_business, criteria like 'Ambiance', 'Location', 'Capacity').
    
    CRITICAL: Output ONLY a valid JSON dictionary. Key = Criterion (1-2 words). Value = 1 sentence definition.
    """
    try:
        response = call_gemini_with_retry(
            prompt=system_prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
        )
        return clean_llm_json(response.text)
    except Exception:
        return {"Quality": "Overall excellence.", "Value": "Return on investment."}

if "dropdown_options" not in st.session_state or st.session_state.get("last_search") != user_search:
    st.session_state.dropdown_options = generate_dropdown_options(user_search, intent.get("domain"))
    st.session_state.last_search = user_search

st.divider()
st.header("2. Define Your Criteria")
criteria_names = list(st.session_state.dropdown_options.keys())

selected_criteria = st.multiselect(
    "Choose your criteria:",
    options=criteria_names,
    default=criteria_names[:4] if len(criteria_names) >= 4 else criteria_names
)

if not selected_criteria:
    st.stop()

# --- RIGID SCHEMA DEFINITIONS ---
class CriterionScore(BaseModel):
    criterion: str
    score: int

class ProductEvaluation(BaseModel):
    scores: list[CriterionScore]

# --- PHASE 3: THE LLM SCORER ---
@st.cache_data(ttl=3600)
def generate_ai_scores(product_list, criteria_list):
    simple_products = [{"name": p["name"]} for p in product_list]
    scoring_prompt = f"""
    Evaluate these options: {simple_products} against these criteria: {criteria_list}.
    Rate each on a scale of 1 to 10 based on market knowledge. 
    You must return a list of evaluations matching the EXACT order of the provided products.
    """
    try:
        response = call_gemini_with_retry(
            prompt=scoring_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", 
                response_schema=list[ProductEvaluation], # The API-safe schema
                temperature=0.1
            )
        )
        
        # Pydantic schema returns {"scores": [{"criterion": "Price", "score": 8}]}
        # We parse it into Python objects
        raw_evaluations = json.loads(response.text) 
        
        # Convert it back to the flat {"Price": 8} dictionary the rest of the app expects
        formatted_scores = []
        for eval_obj in raw_evaluations:
            flat_dict = {item["criterion"]: item["score"] for item in eval_obj.get("scores", [])}
            formatted_scores.append(flat_dict)
            
        return formatted_scores
        
    except Exception as e:
        print(f"❌ CRITICAL LLM SCORING ERROR: {e}")
        return [{c: 5 for c in criteria_list} for _ in product_list]

with st.spinner("Evaluating the market..."):
    ai_scores = generate_ai_scores(surviving_products, selected_criteria)
    for i, product in enumerate(surviving_products):
        if i < len(ai_scores):
            product["scores"] = ai_scores[i]

# --- UI: WEIGHTING & SCORING ---
st.divider()
st.header("3. Set Your Weights")

weights = {}
for crit in selected_criteria:
    definition = st.session_state.dropdown_options.get(crit, "Score from 1-10.")
    weights[crit] = st.slider(crit, min_value=1, max_value=10, value=5, help=definition)

# --- SCORING ENGINE ---
st.divider()
st.header("4. Your Recommendations")

total_weight = sum(weights.values())
results = []
breakdown_data = [] 

for product in surviving_products:
    final_score = 0
    for crit, weight in weights.items():
        normalized_weight = weight / total_weight if total_weight > 0 else 0
        product_crit_score = product.get("scores", {}).get(crit, 5) 
        
        weighted_contribution = (normalized_weight * product_crit_score) * 10
        final_score += weighted_contribution
        
        breakdown_data.append({
            "Option": product["name"],
            "Criterion": crit,
            "Points Contributed": weighted_contribution
        })
        
    results.append({
        "Option": product["name"],
        "Price": product.get("price_str", "N/A") if intent.get("domain") == "local_business" else (f"${product.get('price_usd', 0)}" if product.get('price_usd', 0) > 0 else "N/A"),
        "Match Score": round(final_score, 1),
        "Link": product.get("link", "")
    })

# --- BUG FIX 1: REACTIVE SORTING ---
# Forcing the dataframe to sort top-to-bottom every time a slider is moved
df = pd.DataFrame(results).sort_values(by="Match Score", ascending=False).reset_index(drop=True)
df.index = df.index + 1 

if not df.empty:
    st.success(f"🏆 **Top Pick:** {df.iloc[0]['Option']} (Score: {df.iloc[0]['Match Score']})")
    
    st.dataframe(
        df, 
        column_config={"Link": st.column_config.LinkColumn("Link", display_text="View")},
        use_container_width=True
    )

# --- BUG FIX 3: AXIS CALIBRATION ---
st.divider()
st.subheader("📊 Score Breakdown")

if not df.empty:
    df_breakdown = pd.DataFrame(breakdown_data)
    # Sort the chart's categorical Y-axis to match the newly sorted DataFrame
    df_breakdown['Option'] = pd.Categorical(df_breakdown['Option'], categories=df['Option'][::-1], ordered=True)

    fig = px.bar(
        df_breakdown, 
        x="Points Contributed", 
        y="Option", 
        color="Criterion", 
        orientation="h",
        height=500
    )

    # Calculate the dynamic floor (Lowest score in top 5, minus a visual buffer of 5)
    min_top_score = df.head(5)['Match Score'].min()
    axis_floor = max(0, min_top_score - 5)

    fig.update_layout(
        xaxis_title="Match Score",
        yaxis_title="",
        margin=dict(l=0, r=0, t=0, b=0),
        yaxis=dict(categoryorder='array', categoryarray=df['Option'].tolist()[::-1])
    )

    st.plotly_chart(fig, use_container_width=True)