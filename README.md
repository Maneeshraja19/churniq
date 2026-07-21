# 📊 ChurnIQ — AI-Augmented Customer Churn Prediction

**[🔗 Live Demo](https://churniq-maneesh.streamlit.app)**

![ChurnIQ Screenshot](screenshots/app-screenshot.png)

An end-to-end customer churn prediction tool that goes beyond a raw probability score — it uses a free, local LLM to translate model output into a plain-English risk summary and a concrete retention recommendation, the kind of thing you could actually hand to a customer success manager.

## The Problem

Telecom companies lose significant revenue to customer churn every year. Knowing *that* a customer might leave is only half the problem — knowing *why*, and what to actually do about it, is what makes a churn model useful to a non-technical team. ChurnIQ predicts churn risk from a customer's profile, then uses an AI layer to explain that risk in business terms and suggest a retention action.

## How It Works

1. **Data & EDA** — Cleaned and explored the IBM Telco Customer Churn dataset (~7,000 customers), identifying contract type, tenure, and monthly charges as the strongest churn signals.
2. **Modeling** — Trained and honestly compared Logistic Regression and Random Forest. Logistic Regression won on every metric (ROC-AUC: 0.86 vs. 0.84), a good reminder that a simpler, more interpretable model can beat a more complex one when the underlying relationships are fairly linear.
3. **AI Layer** — For each prediction, the model's real per-customer coefficients are used to identify that specific customer's top churn drivers (not just the model's overall most important features). These are fed to a free, local LLM (Google's FLAN-T5-small, running entirely on CPU) to generate a plain-English summary and retention suggestion.
4. **App** — Wrapped in an interactive Streamlit app so anyone can enter a customer's details and get a live prediction.

**[🔗 Try it live](https://churniq-maneesh.streamlit.app)**

## Tech Stack

- **Python** — pandas, scikit-learn, matplotlib/seaborn
- **Modeling** — Logistic Regression (primary), Random Forest (compared)
- **AI Layer** — Hugging Face Transformers, Google FLAN-T5-small (free, local, no API key required)
- **App** — Streamlit, deployed on Streamlit Community Cloud
- **Versioning** — Git & GitHub

## Key Results

- **ROC-AUC: 0.86** (Logistic Regression)
- **Recall (churned customers): 0.60** — catches 60% of customers who actually churn
- **Top churn drivers identified:** contract length, tenure, monthly charges, internet service type

## Known Limitations

This project deliberately uses free, local, small-scale tools throughout (no paid APIs), which comes with real tradeoffs worth being upfront about:

- **The local LLM (FLAN-T5-small, ~80M parameters) occasionally generates a recommendation that doesn't perfectly align with the stated risk factor's direction** (e.g., suggesting a customer add something that's already increasing their risk). This is a known limitation of using a small model for text generation rather than a larger, paid, frontier model — the first thing to improve with more resources.
- **Random Forest underperformed Logistic Regression here** — included in the analysis anyway as an honest comparison, rather than only reporting the model that "won."

## What I'd Do Next

- Swap in a larger LLM (e.g., via a paid API) for more consistent, logically-grounded recommendations
- Add SHAP values for more rigorous per-customer explainability
- Retrain periodically on rolling data to catch changing churn patterns over time
- Add authentication and a database backend to support real customer data, not just a demo form

## Project Structure
churniq/
├── data/               # Raw dataset
├── notebooks/          # EDA, modeling, and AI-layer notebook
├── examples/           # Sample AI-generated churn summaries
├── models/             # Saved trained model
├── app.py              # Streamlit app
└── requirements.txt    # Dependencies

## Running Locally

```bash
git clone https://github.com/Maneeshraja19/churniq.git
cd churniq
pip install -r requirements.txt
streamlit run app.py
```