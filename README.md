Integrated Market & Credit Risk: ESG as a Leading Indicator for Tail Risk Events

📌 Project Overview
This project investigates whether the integration of Environmental, Social, and Governance (ESG) data enhances the predictive power of traditional credit risk models. By simulating a sophisticated dataset of 1,000 industrial firms, the model analyzes the relationship between corporate sustainability and financial resilience.

Core Thesis: ESG factors, particularly Governance (G), serve as leading indicators for 12-month downside risk, providing "alpha" for risk managers beyond what is captured by standard financial ratios like leverage and interest coverage.

🎯 Goals
To simulate the workflow of a Risk Desk at a major investment bank (e.g., J.P. Morgan, Goldman Sachs) or Asset Manager. The objective is to identify firms with high "Tail Risk" by integrating:

- Financial Fundamentals: Solvency and liquidity metrics.
- Market Indicators: Historical volatility as a proxy for market sentiment.
- ESG Integration: E, S, and G scores as non-traditional risk drivers.

📊 Data Strategy & Features
The dataset is generated using a Latent Risk Factor model, ensuring realistic correlations between governance quality, financial health, and tail risk events.

- Financial Features: ebitda_margin, debt_to_ebitda (Leverage), interest_coverage (Liquidity).
- Market Features: vol_90d (90-day realized volatility).
- ESG Scores: Individual E, S, and G scores + a weighted Composite_ESG_Score.
- Target Variable (target_risk_12m): A binary flag (1 = High Risk) representing a probability of severe downside events (drawdown > 30%) or financial distress.

📈 Model Performance (Phase 2 Results)
We compared a traditional Econometric approach with a Machine Learning Ensemble to evaluate performance and interpretability. The results confirm a high-conviction signal for risk prediction.

Model	ROC-AUC Score	Performance
Logistic Regression	0.9661	Outstanding
XGBoost	0.9574	Excellent

Key Insights:
- Interpretability over Complexity: The Logistic Regression slightly outperformed XGBoost, indicating a strong, near-linear relationship between the selected features and the 12-month risk target.
- Governance Alpha: The feature importance analysis confirms that Interest Coverage and G_score (Governance) are the most critical predictors of tail risk.
- Statistical Validation: Independent T-tests yielded a p-value of 3.2×10^−145, proving that the difference in ESG performance between "Safe" and "High Risk" groups is statistically indisputable.

🛠 Tech Stack
- Language: Python 3.12+ (Optimized for Apple Silicon M4)
- Libraries: * Data Manipulation: pandas, numpy
  Statistics & ML: scipy, scikit-learn, XGBoost
  Visualization: matplotlib, seaborn
- Infrastructure: Homebrew, Virtual Environments (venv)

🗺 Project Roadmap
- [x] Phase 0: Build a robust, correlated synthetic dataset (src/generate_dataset.py).
- [x] Phase 1: Perform rigorous Exploratory Data Analysis and statistical testing (notebooks/01_eda.ipynb).
- [x] Phase 2: Train Econometric (Logit) and Machine Learning (XGBoost) models (src/modeling.py).
- [ ] Phase 3: Implement SHAP values for model explainability (interpreting granular risk drivers).
- [ ] Phase 4: Publish a Final Risk Report and Interactive Dashboard.
