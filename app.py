import streamlit as st
from google import genai
from google.genai import types
import pandas as pd
import json
import time
import plotly.express as px
import requests

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
                print(f"⚠️ Caught 503 error from Google. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            raise e

# --- UI SKELETON ---
st.title("⚖️ Pick For Me")
st.success("Environment and API configured successfully!")

# --- LIVE MARKET SEARCH & CONSTRAINTS ---
st.divider()
st.header("🔍 Search Live Market")

user_search = st.text_input(
    "What are you shopping for?", 
    value="", 
    placeholder="e.g., Laptops, Tennis Shoes, Sunscreen"
)

if not user_search.strip():
    st.info("👋 Welcome to Pick For Me! Enter a product category above to get started.")
    st.stop()

st.divider()
st.header("1. Hard Constraints & Dealbreakers")
st.write("Filter out options based on budget or specific requirements (e.g., 'Size 10', 'Must be white', 'On brand').")
max_budget = st.number_input("Maximum Budget ($)", min_value=0.0, value=0.0, step=50.0, help="Set to 0 for no budget limit.")
unstructured_constraints = st.text_area("Specific Requirements (Optional)", placeholder="e.g., Must be office compatible")

# Combine the search term and constraints so Amazon fetches highly relevant items from the start
combined_search_query = f"{user_search} {unstructured_constraints}".strip()

# --- DYNAMIC FETCHER (GOOGLE SHOPPING VIA SERPAPI) ---
@st.cache_data(ttl=3600)
def fetch_shopping_products(search_term):
    if "SERPAPI_KEY" not in st.secrets:
        st.error("Missing SERPAPI_KEY in secrets.toml! Please add it.")
        st.stop()
        
    params = {
        "engine": "google_shopping",
        "q": search_term,
        "api_key": st.secrets["SERPAPI_KEY"],
        "hl": "en", # English
        "gl": "us", # US Market
    }

    try:
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        data = response.json()
        
        top_results = data.get("shopping_results", [])[:40]
        dynamic_products = []
        seen_names = {} 
        
        for item in top_results:
            # Google Shopping titles can be messy; this deduplicates identical strings
            base_name = item.get("title", "Unknown Product")[:75] + "..."
            if base_name in seen_names:
                seen_names[base_name] += 1
                final_name = f"{base_name} ({seen_names[base_name]})"
            else:
                seen_names[base_name] = 1
                final_name = base_name
                
            # SerpApi provides a cleanly formatted 'extracted_price' float
            price = item.get("extracted_price", 0.0)
                
            dynamic_products.append({
                "name": final_name,
                "price_usd": float(price),
                # THE FIX: Tell it to look for Google's 'product_link' key first
                "link": item.get("product_link", item.get("link", "")),
                "seller": item.get("source", "Unknown") 
            })
        return dynamic_products
    except Exception as e:
        st.error(f"Failed to fetch live Google Shopping data: {e}")
        return []

with st.spinner(f"Scouring the web for '{combined_search_query}'..."):
    raw_products = fetch_shopping_products(combined_search_query)


# --- PHASE 1: THE LLM BOUNCER (PRE-FILTER) ---
@st.cache_data(ttl=3600)
def filter_dealbreakers(product_list, budget, constraints_text):
    budget_filtered = [p for p in product_list if budget <= 0 or float(p.get("price_usd", 0) or 0) <= budget]
        
    # Step 2: LLM Filter (Categorical)
    if not constraints_text.strip() or not budget_filtered:
        return budget_filtered
    
    simple_products = [{"id": i, "name": p["name"]} for i, p in enumerate(budget_filtered)]
    
    filter_prompt = f"""
    You are a product filtering agent.
    Evaluate this full list of products: {simple_products}. 
    The user's requirements are: '{constraints_text}'. 
    Return ONLY a JSON array of the integer IDs for products that match. 
    
    FILTERING RULES:
    1. EXPLICIT EXCLUSIONS: If the user explicitly asks for a specific brand (e.g., 'Only On brand') or color, you MUST rigidly delete any product that belongs to a competing brand or clearly violates the rule.
    2. SUBJECTIVE INCLUSIONS: If the requirement is subjective (e.g., 'Office compatible'), give the product the benefit of the doubt unless it is an obvious mismatch.
    3. COMPLETE EVALUATION: You must evaluate every single item in the provided list from start to finish. Do not truncate the list.
    """
    
    try:
        response = call_gemini_with_retry(
            prompt=filter_prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
        )
        valid_ids = clean_llm_json(response.text) 
        return [budget_filtered[i] for i in valid_ids if i < len(budget_filtered)]
    except Exception:
        return budget_filtered

with st.spinner("Screening products against your dealbreakers..."):
    surviving_products = filter_dealbreakers(raw_products, max_budget, unstructured_constraints)

if not surviving_products:
    st.warning("⚠️ No products matched your strict constraints. Try loosening your dealbreakers or raising your budget.")
    st.stop()

# --- PHASE 2: STRICT QUANTITATIVE CRITERIA ---
@st.cache_data(ttl=3600)
def generate_dropdown_options(category: str) -> dict:
    system_prompt = f"""
    You are a product analytics engine. 
    Suggest 10 distinct, quantifiable decision criteria a consumer should consider when shopping for: '{category}'.
    
    CRITICAL RULE: DO NOT suggest categorical or binary preferences like 'Color' or 'Brand'. 
    
    RULES:
    1. Output ONLY a valid JSON dictionary.
    2. The key must be the criterion name (1-2 words).
    3. The value must be a concise, 1-sentence definition of that criterion.
    4. Always include "Price" as a key.
    
    EXPECTED FORMAT:
    {{"Price": "The overall financial cost.", "Traction": "The level of grip provided on various surfaces."}}
    """
    try:
        response = call_gemini_with_retry(
            prompt=system_prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
        )
        return clean_llm_json(response.text)
    except Exception as e:
        st.error(f"Criteria generation failed: {e}")
        # Fallback dictionary
        return {
            "Price": "The overall financial cost.", 
            "Performance": "How well the product executes its primary function.", 
            "Durability": "The expected lifespan and resistance to wear.", 
            "Design": "The aesthetic and structural build quality."
        }

if "dropdown_options" not in st.session_state or st.session_state.get("last_search_dropdown") != user_search:
    st.session_state.dropdown_options = generate_dropdown_options(user_search)
    st.session_state.last_search_dropdown = user_search

st.divider()
st.header("2. Define Your Criteria")
st.write(f"Select the quantifiable factors that matter most for scoring. Hover over the question marks for definitions.")

# Extract just the keys (the criteria names) for the multiselect dropdown
criteria_names = list(st.session_state.dropdown_options.keys())

selected_criteria = st.multiselect(
    "Choose your criteria:",
    options=criteria_names,
    default=criteria_names[:4] if len(criteria_names) >= 4 else criteria_names
)

if not selected_criteria:
    st.warning("Please select at least one criterion to continue.")
    st.stop()

# --- PHASE 3: THE LLM SCORER ---
@st.cache_data(ttl=3600)
def generate_ai_scores(product_list, criteria_list):
    simple_products = [{"name": p["name"], "price": p["price_usd"]} for p in product_list]
    
    scoring_prompt = f"""
    You are a product evaluation engine.
    Evaluate these products: {simple_products}
    Against these specific criteria: {criteria_list}

    Rate each product on a scale of 1 to 10 for each criterion based on real-world market knowledge.
    Rule: For 'Price', cheaper items MUST receive HIGHER scores closer to 10.

    Return ONLY a flat JSON array of dictionaries. The array order MUST exactly match the product order.
    """
    try:
        response = call_gemini_with_retry(
            prompt=scoring_prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
        )
        return clean_llm_json(response.text)
    except Exception:
        st.warning("⚠️ Google's AI servers are currently experiencing high traffic. Defaulting to neutral scores so you can still browse your matches.") 
        return [{c: 5 for c in criteria_list} for _ in product_list]

with st.spinner("AI is evaluating the surviving market against your criteria..."):
    ai_scores = generate_ai_scores(surviving_products, selected_criteria)
    
    for i, product in enumerate(surviving_products):
        if i < len(ai_scores):
            product["scores"] = ai_scores[i]


# --- UI: WEIGHTING & SCORING ---
st.divider()
st.header("3. Set Your Weights")
st.write("Rate how important each factor is to you from 1 (Not Important) to 10 (Crucial).")

weights = {}
for crit in selected_criteria:
    definition = st.session_state.dropdown_options.get(crit, "Score from 1-10.")
    weights[crit] = st.slider(crit, min_value=1, max_value=10, value=5, help=definition)


# --- THE SCORING ENGINE ---
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
            "Product": product["name"],
            "Criterion": crit,
            "Points Contributed": weighted_contribution
        })
        
    results.append({
        "Product": product["name"],
        "Price": f"${product.get('price_usd', 0)}",
        "Match Score": round(final_score, 1),
        "Link": product.get("link", "")
    })

# --- DATA FORMATTING & DISPLAY ---
df = pd.DataFrame(results).sort_values(by="Match Score", ascending=False).reset_index(drop=True)
df.index = df.index + 1 

if not df.empty:
    st.success(f"🏆 **Top Pick:** {df.iloc[0]['Product']} (Score: {df.iloc[0]['Match Score']})")
    
    # Configure the dataframe to render the URL as a clean, clickable hyperlink
    st.dataframe(
        df, 
        column_config={
            "Link": st.column_config.LinkColumn(
                "Buy Link", 
                help="Click to view this product on Google Shopping",
                display_text="View Product"
            )
        },
        use_container_width=True
    )

# --- AI EXPLAINABILITY LAYER ---
st.divider()
st.subheader("🤖 AI Analysis")

if st.button("Explain My Top Match") and not df.empty:
    top_product_name = df.iloc[0]['Product']
    
    explanation_prompt = f"""
    You are a concise shopping assistant. A user just used a weighted decision engine.
    The top recommended product is the '{top_product_name}'.
    The user's weighting preferences (on a scale of 1 to 10) were: {weights}.
    Write a 2-sentence conversational justification explaining exactly why this product won, focusing specifically on their highest-weighted criteria.
    Do not use markdown, bullet points, or introductory filler. Just provide the 2 sentences.
    """

    with st.spinner("Analyzing your top match..."):
        try:
            explanation_response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=explanation_prompt,
            )
            st.info(f"💡 **Why it won:** {explanation_response.text.strip()}")
        except Exception:
            st.warning("💡 **Why it won:** This product scored the highest aggregate match across your weighted priorities.")

# --- VISUALIZATION: EXPLAINABILITY CHART ---
st.divider()
st.subheader("📊 Score Breakdown")

if not df.empty:
    df_breakdown = pd.DataFrame(breakdown_data)
    df_breakdown['Product'] = pd.Categorical(df_breakdown['Product'], categories=df['Product'][::-1], ordered=True)

    fig = px.bar(
        df_breakdown, 
        x="Points Contributed", 
        y="Product", 
        color="Criterion", 
        orientation="h",
        height=500
    )

    fig.update_layout(
        xaxis_title="Total Score (out of 100)",
        yaxis_title="",
        legend_title="Criteria",
        margin=dict(l=0, r=0, t=0, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)