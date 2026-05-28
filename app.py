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

# --- UI SKELETON ---
st.title("⚖️ Pick For Me")
st.subheader("The transparent decision engine.")
st.success("Environment and API configured successfully!")

# --- LIVE MARKET SEARCH ---
st.divider()
st.header("🔍 Search Live Market")
user_search = st.text_input("What are you shopping for?", value="Sneakers")

# --- DYNAMIC FETCHER (RAINFOREST API) ---
@st.cache_data(ttl=3600)
def fetch_amazon_products(search_term):
    if "RAINFOREST_API_KEY" not in st.secrets:
        st.error("Missing RAINFOREST_API_KEY in secrets.toml! Please add it.")
        st.stop()
        
    rainforest_key = st.secrets["RAINFOREST_API_KEY"]
    
    params = {
        "api_key": rainforest_key,
        "type": "search",
        "amazon_domain": "amazon.com",
        "search_term": search_term,
        "sort_by": "featured"
    }

    try:
        response = requests.get("https://api.rainforestapi.com/request", params=params)
        response.raise_for_status()
        data = response.json()
        
        # Bumped to 15 to give the pre-filter a larger pool of options
        top_results = data.get("search_results", [])[:15]
        
        dynamic_products = []
        for item in top_results:
            dynamic_products.append({
                "name": item.get("title", "Unknown Product")[:80] + "...",
                "price_usd": item.get("price", {}).get("value", 0.0),
                "link": item.get("link", ""),
                "raw_specs": item.get("rating", 0)
            })
            
        return dynamic_products
        
    except Exception as e:
        st.error(f"Failed to fetch live Amazon data: {e}")
        return []

with st.spinner(f"Pulling live Amazon data for '{user_search}'..."):
    raw_products = fetch_amazon_products(user_search)


# --- PHASE 1: THE LLM BOUNCER (PRE-FILTER) ---
st.divider()
st.header("1. Hard Constraints & Dealbreakers")
st.write("Filter out options based on budget or specific categorical requirements (e.g., 'Size 10', 'Nike brand only', 'Must be black').")

max_budget = st.number_input("Maximum Budget ($)", min_value=0.0, value=0.0, step=50.0, help="Set to 0 for no budget limit.")
unstructured_constraints = st.text_area("Specific Requirements (Optional)", placeholder="e.g., Must have a backlit keyboard and weigh under 3 lbs.")

@st.cache_data(ttl=3600)
def filter_dealbreakers(product_list, budget, constraints_text):
    # Step 1: Standard Math Filter (Budget)
    budget_filtered = []
    for p in product_list:
        price = float(p.get("price_usd", 0) or 0)
        if budget > 0 and price > budget:
            continue
        budget_filtered.append(p)
        
    # Step 2: LLM Filter (Categorical)
    if not constraints_text.strip() or not budget_filtered:
        return budget_filtered
        
    simple_products = [{"id": i, "name": p["name"]} for i, p in enumerate(budget_filtered)]
    
    filter_prompt = f"""
    You are a strict product screening agent.
    The user has these mandatory dealbreakers: "{constraints_text}"
    
    Evaluate this list of products: {simple_products}
    
    Return ONLY a flat JSON array of the integer IDs for the products that likely meet the user's requirements based on their name. If a product clearly violates a requirement, exclude its ID.
    Example output: [0, 2, 5, 6]
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=filter_prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
        )
        valid_ids = json.loads(response.text)
        return [budget_filtered[i] for i in valid_ids if i < len(budget_filtered)]
    except Exception as e:
        st.warning("AI Pre-filter failed, passing all budget-approved items through.")
        return budget_filtered

with st.spinner("Screening products against your dealbreakers..."):
    surviving_products = filter_dealbreakers(raw_products, max_budget, unstructured_constraints)

if not surviving_products:
    st.warning("⚠️ No products matched your strict constraints. Try loosening your dealbreakers or raising your budget.")
    st.stop()


# --- PHASE 2: STRICT QUANTITATIVE CRITERIA ---
@st.cache_data(ttl=3600)
def generate_dropdown_options(category: str) -> list:
    system_prompt = f"""
    You are a product analytics engine. 
    Suggest a list of 10 to 12 distinct decision criteria a consumer should consider when shopping for: '{category}'.
    
    CRITICAL RULE: DO NOT suggest categorical, binary, or subjective preferences like 'Color', 'Size', 'Brand', or 'Style'. 
    ONLY suggest quantifiable metrics that can logically be scored on a 1-10 scale (e.g., 'Durability', 'Battery Life', 'Weight', 'Refresh Rate').
    
    RULES:
    1. Output ONLY a valid, flat JSON array of strings.
    2. Limit each criterion to 1-2 words max.
    3. Always include "Price" or "Affordability" as an option.
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=system_prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
        )
        return json.loads(response.text)
    except Exception:
        return ["Price", "Performance", "Build Quality", "Reliability", "Design", "Value"]

if "dropdown_options" not in st.session_state or st.session_state.get("last_search_dropdown") != user_search:
    st.session_state.dropdown_options = generate_dropdown_options(user_search)
    st.session_state.last_search_dropdown = user_search

st.divider()
st.header("2. Define Your Criteria")
st.write(f"Select the quantifiable factors that matter most for scoring.")

selected_criteria = st.multiselect(
    "Choose your criteria:",
    options=st.session_state.dropdown_options,
    default=st.session_state.dropdown_options[:4] if len(st.session_state.dropdown_options) >= 4 else st.session_state.dropdown_options
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
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=scoring_prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
        )
        return json.loads(response.text)
    except Exception:
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
    weights[crit] = st.slider(crit, min_value=1, max_value=10, value=5)

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
        "Match Score": round(final_score, 1)
    })

# --- DATA FORMATTING & DISPLAY ---
df = pd.DataFrame(results).sort_values(by="Match Score", ascending=False).reset_index(drop=True)
df.index = df.index + 1 

if not df.empty:
    st.success(f"🏆 **Top Pick:** {df.iloc[0]['Product']} (Score: {df.iloc[0]['Match Score']})")
    st.dataframe(df, use_container_width=True)

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
        except Exception as e:
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