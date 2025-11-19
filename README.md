# Credit-Risk-ESG-Model
Test whether the integration of esg data improves the predictive ability of a credit risk model for banks, compared to a model that uses only traditional financial data.
# Credit Risk ESG Model

Research-style project combining **financial ratios**, **market risk metrics** and **ESG scores** to predict 12-month downside risk for industrial firms.

## 🔍 Goal

Explore whether ESG adds predictive power to traditional credit risk models.  
The idea is to simulate how a **risk desk in a bank or asset manager** could integrate:

- Financial statement data (leverage, coverage, margins)
- Market-based indicators (volatility, drawdown)
- ESG scores (E, S, G and composite)

to identify companies with **high downside risk**.

## 📦 Data (conceptual design)

- **Financial**: EBITDA margin, Debt/EBITDA, Interest coverage  
- **Market**: 90-day volatility, 12-month max drawdown  
- **ESG**: E, S, G scores and a composite ESG score  
- **Target**: 1 if the stock experiences a drawdown worse than –30% over 12 months (high risk), 0 otherwise.

(At the beginning, data can be simulated but realistic. Later, real KPIs from sustainability reports can be added.)

## 🧠 Methods

- Descriptive statistics & group comparison (high vs low risk)
- Econometric models: Logit / Probit / OLS
- Machine learning: Logistic Regression, Random Forest, Gradient Boosting
- Explainability: SHAP values to understand drivers of risk

## 🧰 Tech stack

Python (pandas, numpy, scikit-learn, xgboost, shap), Jupyter Notebooks, possibly Streamlit for a simple dashboard.

## 🚀 Next steps

- Build a clean dataset (finance + ESG + target)
- Run econometric and ML models
- Publish a short report and key charts
- Share the project on LinkedIn
