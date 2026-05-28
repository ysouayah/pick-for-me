import streamlit as st
from google import genai
from google.genai import types
import pandas as pd
import json
import time
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pick For Me", page_icon="⚖️", layout="centered")

# --- API CONFIG ---
# Pull the key securely from Streamlit's secrets manager
api_key = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)

# --- UI SKELETON ---
st.title("⚖️ Pick For Me")
st.subheader("The transparent decision engine.")

st.success("Environment and API configured successfully!")

# --- CRITERIA ENGINE (LLM) ---
def generate_criteria(category: str) -> list:
    """
    Calls Gemini to generate 5-6 decision criteria for a given category.
    Includes exponential backoff retry logic to safeguard against API drops.
    """
    
    system_prompt = f"""
    You are a precise decision-analysis data pipeline. 
    Your sole task is to suggest 5 to 6 critical decision criteria a consumer should use when buying a product in the category: '{category}'.
    
    RULES:
    1. Output ONLY a valid, flat JSON array of strings.
    2. Do not include markdown formatting (e.g., ```json).
    3. Do not include any conversational text.
    4. Limit criteria to 1-2 words maximum (e.g., "Battery Life", "Durability").
    5. Always include "Affordability" or "Price" as one of the criteria.
    
    EXPECTED OUTPUT FORMAT:
    ["Criterion 1", "Criterion 2", "Criterion 3", "Criterion 4", "Criterion 5"]
    """

    # Exponential backoff retry logic (1s, 2s, 4s, 8s, 16s)
    max_retries = 5
    base_delay = 1

    for attempt in range(max_retries):
        try:
            # New SDK syntax for generation
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=system_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )
            
            # Parse the guaranteed JSON string into a Python list
            criteria_list = json.loads(response.text)
            
            # Enforce the hard limit of 6 criteria just in case
            return criteria_list[:6]
            
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"API Failed after {max_retries} attempts: {e}")
                # Fallback safeguard
                return ["Price", "Quality", "Performance", "Design", "Reliability"]
            
            time.sleep(base_delay * (2 ** attempt))

# --- STATE MANAGEMENT ---
# Store criteria in session state so it doesn't regenerate on every slider click later
if "criteria" not in st.session_state:
    with st.spinner("AI is analyzing category: Laptops..."):
        st.session_state.criteria = generate_criteria("Laptops")

# --- UI: CRITERIA EDITOR ---
st.divider()
st.header("1. Define Your Criteria")
st.write("The AI suggested these factors. Feel free to edit them (Maximum of 6).")

# Create a dynamic list of text inputs for the user to edit the criteria
updated_criteria = []
for i, crit in enumerate(st.session_state.criteria):
    # The key ensures Streamlit tracks each input box independently
    user_edit = st.text_input(f"Criterion {i+1}", value=crit, key=f"crit_input_{i}")
    
    # Only keep non-empty strings to allow users to "delete" a criterion by clearing the box
    if user_edit.strip():
        updated_criteria.append(user_edit.strip())

# Enforce the hard limit of 6 in the final list
st.session_state.criteria = updated_criteria[:6]

# --- UI: WEIGHTING & SCORING ---
st.divider()
st.header("2. Set Your Weights")
st.write("Rate how important each factor is to you from 1 (Not Important) to 10 (Crucial).")

# 1. Create the interactive sliders dynamically based on the active criteria
weights = {}
for crit in st.session_state.criteria:
    weights[crit] = st.slider(crit, min_value=1, max_value=10, value=5)

st.divider()

# 2. Load the curated data
try:
    with open("data.json", "r") as f:
        database = json.load(f)
except FileNotFoundError:
    st.error("data.json file not found. Please ensure it is in the same directory.")
    st.stop()

target_category = "Laptops"
products = database.get(target_category, [])

# --- NEW: HARD CONSTRAINTS HANDLER ---
st.divider()
st.header("3. Set Hard Constraints")
st.write("Filter out options that are immediate dealbreakers.")

# Optional max budget input (0 means no limit)
max_budget = st.number_input(
    "Maximum Budget ($)", 
    min_value=0, 
    value=0, 
    step=100, 
    help="Set to 0 for no budget limit."
)

# Intercept and filter the dataset
filtered_products = []
for product in products:
    # If a budget is set and the laptop costs more than the budget, skip it entirely
    if max_budget > 0 and product["price_usd"] > max_budget:
        continue 
    filtered_products.append(product)

# Failsafe if the constraint filters out literally everything
if not filtered_products:
    st.warning("⚠️ No products match your current budget constraint. Try raising your maximum price.")
    st.stop() # Halts the app here so the math engine doesn't crash on an empty list

# 4. The Scoring Engine & Data Collection
st.divider()
st.header("4. Your Recommendations")

total_weight = sum(weights.values())
results = []
breakdown_data = [] 

# Notice we are now looping through 'filtered_products' instead of 'products'
for product in filtered_products:
    final_score = 0
    
    for crit, weight in weights.items():
        normalized_weight = weight / total_weight
        product_crit_score = product["scores"].get(crit, 5) 
        
        weighted_contribution = (normalized_weight * product_crit_score) * 10
        final_score += weighted_contribution
        
        breakdown_data.append({
            "Product": product["name"],
            "Criterion": crit,
            "Points Contributed": weighted_contribution
        })
        
    results.append({
        "Product": product["name"],
        "Price": f"${product['price_usd']}",
        "Match Score": round(final_score, 1)
    })

# 4. Data Formatting & Display
df = pd.DataFrame(results).sort_values(by="Match Score", ascending=False).reset_index(drop=True)
df.index = df.index + 1 

st.success(f"🏆 **Top Pick:** {df.iloc[0]['Product']} (Score: {df.iloc[0]['Match Score']})")
st.dataframe(df, use_container_width=True)

# --- AI EXPLAINABILITY LAYER ---
st.divider()
st.subheader("🤖 AI Analysis")
st.write("Want to know why this laptop took the #1 spot?")

# By wrapping this in a button, Streamlit will ONLY call the API when clicked,
# saving your quota from slider-spam!
if st.button("Explain My Top Match"):
    
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

# 5. Visualization: The Explainability Chart
st.divider()
st.subheader("📊 Score Breakdown")
st.write("See exactly why your top pick won based on your weighted preferences.")

# Create a DataFrame specifically for the chart and sort it to match our ranked table
df_breakdown = pd.DataFrame(breakdown_data)
# Sort the chart's y-axis so the #1 product is at the top
df_breakdown['Product'] = pd.Categorical(df_breakdown['Product'], categories=df['Product'][::-1], ordered=True)

# Build the stacked horizontal bar chart
fig = px.bar(
    df_breakdown, 
    x="Points Contributed", 
    y="Product", 
    color="Criterion", 
    orientation="h",
    height=400
)

# Clean up the UI layout of the chart
fig.update_layout(
    xaxis_title="Total Score (out of 100)",
    yaxis_title="",
    legend_title="Criteria",
    margin=dict(l=0, r=0, t=0, b=0)
)

st.plotly_chart(fig, use_container_width=True)