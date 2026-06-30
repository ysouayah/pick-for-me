import streamlit as st
from google import genai
from google.genai import types
import pandas as pd
import json
import time
import plotly.express as px
import requests
from pydantic import BaseModel
from urllib.parse import quote_plus
import concurrent.futures

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pick For Me", page_icon="⚖️", layout="centered")

# --- API CONFIG ---
api_key = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)

def clean_llm_json(raw_text):
    cleaned = raw_text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)

def call_gemini_with_retry(prompt, config, max_retries=5): 
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

# --- INTENT ROUTER ---
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
        # Smarter fallback if the API glitches
        if any(word in search_term.lower() for word in ["restaurant", "cafe", "food", "dinner", "bar"]):
            return {"domain": "local_business", "is_apparel": False}
        return {"domain": "retail", "is_apparel": False}

with st.spinner("Analyzing intent..."):
    intent = classify_search_intent(user_search)

# --- DEMOGRAPHIC GATEKEEPER ---
demographic_query = ""
if intent.get("is_apparel"):
    st.warning("👕 **Apparel Detected:** Please specify sizing to filter out unpurchasable inventory.")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Target Demographic:", ["", "Men", "Women", "Unisex", "Kids"])
    with col2:
        sizes = st.multiselect("Acceptable Sizes:", ["6", "7", "8", "9", "10", "11", "12", "S", "M", "L", "XL"])
    
    if not gender or not sizes:
        st.stop() 
    demographic_query = f"{gender} size {','.join(sizes)}"

st.divider()
st.header("1. Hard Constraints & Dealbreakers")
max_budget = st.number_input("Maximum Budget ($)", min_value=0.0, value=0.0, step=50.0, help="Set to 0 for no budget limit.")
unstructured_constraints = st.text_area("Specific Requirements (Optional)")

# --- THE SEARCH QUERY DECOUPLER ---
if intent.get("domain") == "local_business":
    api_search_query = user_search.strip()
else:
    api_search_query = f"{user_search} {demographic_query}".strip()

# --- DECOUPLED FETCHER WITH PARALLEL PAGINATION & METADATA ---
# NOTE: Caching removed so dead API calls do not permanently stick to the session
@st.cache_data(ttl=3600)
def fetch_dynamic_data(search_term, domain):
    if "SERPAPI_KEY" not in st.secrets:
        st.error("Missing SERPAPI_KEY in secrets.toml!")
        st.stop()
        
    engine = "google_local" if domain == "local_business" else "google_shopping"
    
    def fetch_page(start_offset):
        params = {
            "engine": engine,
            "q": search_term,
            "api_key": st.secrets["SERPAPI_KEY"],
            "hl": "en",
            "gl": "us",
            "start": start_offset
        }
        try:
            response = requests.get("https://serpapi.com/search", params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"⚠️ Page {start_offset} fetch failed: {e}")
            return {}

    dynamic_items = []
    seen_names = {}
    raw_results = []
    
    offsets = [0, 40, 80] if domain == "retail" else [0, 20, 40]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_offset = {executor.submit(fetch_page, offset): offset for offset in offsets}
        for future in concurrent.futures.as_completed(future_to_offset):
            data = future.result()
            if domain == "local_business":
                raw_results.extend(data.get("local_results", []))
            else:
                raw_results.extend(data.get("shopping_results", []))
                
    for item in raw_results:
        base_name = item.get("title", "Unknown Product")[:75]
        if base_name in seen_names:
            continue
        seen_names[base_name] = True
        
        if domain == "local_business":
            dynamic_items.append({
                "name": base_name,
                "type": item.get("type", "Venue"), 
                "price_usd": 0.0, 
                "price_str": item.get("price", "N/A"), 
                "primary_link": item.get("website", item.get("links", {}).get("website", "")),
                "fallback_link": "",
                "seller": item.get("address", "Unknown Address") 
            })
        else:
            seller_name = item.get("source", "Unknown")
            original_link = item.get("product_link", item.get("link", ""))
            
            if "amazon" in seller_name.lower() or "amazon.com" in original_link.lower():
                primary_link = original_link
                fallback_link = ""
            else:
                encoded_title = quote_plus(base_name)
                primary_link = f"https://www.amazon.com/s?k={encoded_title}"
                fallback_link = original_link
            
            dynamic_items.append({
                "name": base_name,
                "type": "Product",
                "price_usd": float(item.get("extracted_price", 0.0)),
                "primary_link": primary_link,
                "fallback_link": fallback_link,
                "seller": seller_name
            })
            
    return dynamic_items

with st.spinner(f"Scouring the web for '{api_search_query}'..."):
    raw_products = fetch_dynamic_data(api_search_query, intent.get("domain"))

# New Check: Make sure Google actually returned data before moving to Phase 1
if not raw_products:
    st.error("⚠️ The web scraper found 0 results. Google might have blocked the request or timed out. Try clearing your cache.")
    st.stop()

# --- PHASE 1: THE LLM BOUNCER ---
@st.cache_data(ttl=3600)
def filter_dealbreakers(product_list, budget, constraints_text, core_query):
    budget_filtered = [p for p in product_list if budget <= 0 or float(p.get("price_usd", 0) or 0) <= budget]
        
    if not budget_filtered:
        return []
    
    # FIX: If the user didn't enter constraints, skip the AI entirely and push data to Phase 3
    if not constraints_text.strip():
        return budget_filtered[:40]
    
    simple_products = [{"id": i, "name": p["name"], "category_type": p.get("type", "N/A")} for i, p in enumerate(budget_filtered)]
    
    filter_prompt = f"""
    You are a strict data filtering agent.
    Evaluate this list of options: {simple_products}. 
    The user's original core search query is: '{core_query}'.
    The user's strict dealbreakers are: '{constraints_text}'. 
    Return ONLY a JSON array of the integer IDs for items that pass this initial screening. 
    
    FILTERING RULES:
    1. CATEGORY & CUISINE ALIGNMENT (LOCAL BUSINESS): If the core search or constraints specify a distinct type of cuisine, establishment, or service, you MUST immediately eliminate any option whose 'category_type' or name explicitly contradicts it. 
    2. OBVIOUS CONTRADICTIONS: Delete any item that directly violates a clear constraint.
    3. BENEFIT OF THE DOUBT (RETAIL ONLY): Do NOT delete a retail item just because a requested spec is missing from the title. If you aren't 100% sure it violates the rule, let it pass to Phase 3.
    4. THE CABLE & VIDEO RULE: If video/display is required, eliminate pure charging cables.
    5. COMPLETE EVALUATION: Evaluate every single item in the provided list. Do not truncate.
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
        
        if not valid_ids:
            st.warning("⚠️ Phase 1 Bouncer was too strict (titles lacked specs). Bypassing to Phase 3.")
            return budget_filtered[:40]
            
        return [budget_filtered[i] for i in valid_ids if i < len(budget_filtered)][:40]
        
    except Exception as e:
        print(f"❌ Filter Error: {e}")
        return budget_filtered[:40]

with st.spinner("Screening options against your dealbreakers..."):
    surviving_products = filter_dealbreakers(raw_products, max_budget, unstructured_constraints, user_search)

if not surviving_products:
    st.warning("⚠️ No options matched your strict constraints.")
    st.stop()

# --- PHASE 2: DYNAMIC AI CRITERIA ---
@st.cache_data(ttl=3600)
def generate_dropdown_options(category: str, domain: str) -> dict:
    system_prompt = f"""
    You are an analytics engine. 
    Suggest 10 distinct, quantifiable decision criteria a user should consider when evaluating a '{category}'.
    Context: This is a '{domain}'.
    
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
# --- PHASE 3: THE LLM SCORER ---
@st.cache_data(ttl=3600)
def generate_ai_scores(product_list, criteria_list):
    # FIX: We are finally passing the actual live scraped prices to the AI Scorer
    simple_products = [
        {
            "name": p["name"], 
            "price": p.get("price_str") if p.get("price_str") != "N/A" else f"${p.get('price_usd', 0)}"
        } 
        for p in product_list
    ]
    
    scoring_prompt = f"""
    Evaluate these options: {simple_products} against these criteria: {criteria_list}.
    Rate each on a scale of 1 to 10 based on market knowledge and the provided price data. 
    
    CRITICAL SCORING RULES:
    1. THE POLARITY RULE: A score of 10 must ALWAYS represent the most positive/desirable outcome for the consumer. 
       - For 'Price'/Cost: 10 = Extremely cheap/Amazing value. 1 = Outrageously expensive.
       - For 'Quality', 'Durability', etc.: 10 = World-class. 1 = Terrible.
       
    2. THE DROPSHIPPER PENALTY (BRAND REALITY CHECK): You are evaluating live web-scraped retail data. You MUST aggressively penalize unbranded, keyword-stuffed products. 
       - If a product title is a salad of generic buzzwords (e.g., "Best Ergonomic 3D Adjustable...") and lacks a reputable premium brand name, you must assume it is cheap, low-quality white-label garbage. Score it extremely low (1 to 3) for criteria like Durability, Quality, Comfort, or Adjustability.
       - Conversely, heavily reward verified premium brands (e.g., Steelcase, Herman Miller, Haworth, Humanscale) for those same criteria, even if their title is short.
    
    You must return a list of evaluations matching the EXACT order of the provided products.
    """
    try:
        response = call_gemini_with_retry(
            prompt=scoring_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", 
                response_schema=list[ProductEvaluation], 
                temperature=0.1
            )
        )
        
        raw_evaluations = json.loads(response.text) 
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

display_limit = st.slider("Number of Results to Show", min_value=1, max_value=40, value=10)

# AMPLIFY WEIGHTS: Convert linear 0-10 UI scale to an exponential internal scale for visual separation
internal_weights = {crit: (val ** 2) for crit, val in weights.items()}
total_weight = sum(internal_weights.values())

results = []
breakdown_data = [] 

for product in surviving_products:
    final_score = 0
    # FIX: Loop through the amplified internal_weights
    for crit, internal_weight in internal_weights.items():
        normalized_weight = internal_weight / total_weight if total_weight > 0 else 0
        product_crit_score = product.get("scores", {}).get(crit, 5) 
        
        weighted_contribution = (normalized_weight * product_crit_score) * 10
        final_score += weighted_contribution
        
        breakdown_data.append({
            "Option": product["name"],
            "Criterion": crit,
            "Points Contributed": weighted_contribution
        })
        
    is_local = intent.get("domain") == "local_business"
    primary_label = "Website" if is_local else "Find on Amazon"
    fallback_label = "Address" if is_local else "Alternative Link"
    
    if is_local:
        raw_url = product.get("primary_link", "")
        primary_val = raw_url if raw_url else f"https://www.google.com/search?q={quote_plus(product['name'])}"
        fallback_val = product.get("seller", "N/A")
    else:
        primary_val = product.get("primary_link", "")
        fallback_val = product.get("fallback_link") if product.get("fallback_link") else None
    
    results.append({
        "Option": product["name"],
        "Price": product.get("price_str", "N/A") if is_local else (f"${product.get('price_usd', 0)}" if product.get('price_usd', 0) > 0 else "N/A"),
        "Match Score": round(final_score, 1),
        primary_label: primary_val,
        fallback_label: fallback_val
    })

# --- REACTIVE SORTING & LIMITING ---
df = pd.DataFrame(results).sort_values(by="Match Score", ascending=False).reset_index(drop=True)
df = df.head(display_limit)
df.index = df.index + 1 

if not df.empty:
    st.success(f"🏆 **Top Pick:** {df.iloc[0]['Option']} (Score: {df.iloc[0]['Match Score']})")
    
    column_config = {}
    if intent.get("domain") == "local_business":
        column_config["Website"] = st.column_config.LinkColumn("Website", display_text="Visit")
    else:
        column_config["Find on Amazon"] = st.column_config.LinkColumn("Find on Amazon", display_text="Search Amazon")
        column_config["Alternative Link"] = st.column_config.LinkColumn("Alternative Link", display_text="Direct Vendor")

    st.dataframe(
        df, 
        column_config=column_config,
        use_container_width=True
    )

# --- AXIS CALIBRATION ---
st.divider()
st.subheader("📊 Score Breakdown")

if not df.empty:
    df_breakdown = pd.DataFrame(breakdown_data)
    df_breakdown = df_breakdown[df_breakdown['Option'].isin(df['Option'])]
    df_breakdown['Option'] = pd.Categorical(df_breakdown['Option'], categories=df['Option'][::-1], ordered=True)

    fig = px.bar(
        df_breakdown, 
        x="Points Contributed", 
        y="Option", 
        color="Criterion", 
        orientation="h",
        # FIX 1: Increased vertical breathing room to 60px per row
        height=max(500, len(df) * 60)
    )

    fig.update_layout(
        xaxis_title="Match Score",
        yaxis_title="",
        # FIX 2: Restored natural margins and cranked bottom margin for the legend
        margin=dict(l=10, r=20, t=30, b=120),
        # FIX 3: Moved legend to the bottom safely below the x-axis text
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=-0.2, 
            xanchor="center", 
            x=0.5,
            title=""
        ),
        yaxis=dict(
            categoryorder='array', 
            categoryarray=df['Option'].tolist()[::-1],
            # FIX 4: Truncate massive product names to 45 chars so bars can expand
            tickmode='array',
            ticktext=[name[:45] + '...' if len(name) > 45 else name for name in df['Option'].tolist()[::-1]],
            tickvals=df['Option'].tolist()[::-1]
        )
    )

    st.plotly_chart(fig, use_container_width=True)