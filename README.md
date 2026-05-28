# ⚖️ Pick For Me

**Pick For Me** is an AI-assisted decision engine designed to cure e-commerce choice paralysis. Instead of relying on rigid, binary filters (which often hide the best options), this application uses a Multi-Criteria Decision Analysis (MCDA) architecture to help users balance competing tradeoffs using live market data.

## ✨ Core Architecture (v2)

### The Two-Stage AI Pipeline
MCDA math engines break when fed subjective or categorical data (e.g., "Must be Adidas" or "Size 10"). To solve this, the app uses a dual-LLM approach:
1. **Stage 1 (The Bouncer):** An AI pre-filter evaluates an unstructured text box of user dealbreakers, instantly deleting non-compliant products from the raw dataset before the math starts.
2. **Stage 2 (The Scorer):** The AI evaluates the surviving products, dynamically assigning 1-10 ratings for strictly *quantifiable* metrics (e.g., Weight, Battery Life) based on the user's chosen criteria.

### Features
* **Live Market Integration:** Uses the Rainforest API to bypass anti-bot CAPTCHAs and fetch real-time organic search results and pricing from Amazon.
* **Smart UI Generation:** Dynamically generates contextual dropdown criteria based on the user's search query (e.g., suggesting "Cushioning" for sneakers, but "Refresh Rate" for monitors).
* **Transparent Scoring Engine:** Normalizes user slider weights (1-10) and computes a deterministic final match score. The AI never touches the final math, preventing hallucinated rankings.
* **Visual Explainability:** Uses Plotly to generate a real-time stacked bar chart, visually proving exactly how many points each criterion contributed to a product's final score. Includes automatic deduplication to handle keyword-stuffed Amazon titles.
* **Direct Action:** Automatically routes users to the winning Amazon product pages via clickable UI links.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Backend:** Python, Pandas
* **Data Visualization:** Plotly
* **AI Integration:** Google GenAI SDK (`gemini-2.5-flash`)
* **Data Pipeline:** Rainforest API (Amazon Search)

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ysouayah/pick-for-me.git
   cd pick-for-me
   ```

2. **Set up your environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Configure the API Keys:**
   Create a `.streamlit/secrets.toml` file in the root directory and add your keys:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   RAINFOREST_API_KEY = "your_rainforest_api_key_here"
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```
