Integrated Market & Credit Risk: ESG as a Leading Indicator for Tail Risk Events

Project OverviewThis project investigates whether the integration of Environmental, Social, and Governance (ESG) data enhances the predictive power of traditional credit risk models. By simulating a sophisticated dataset of 1,000 industrial firms, the model analyzes the relationship between corporate sustainability and financial resilience.
The core thesis is that ESG factors, particularly Governance (G), serve as leading indicators for 12-month downside risk, providing "alpha" for risk managers beyond what is captured by standard financial ratios like leverage and interest coverage.

Goal
To simulate the workflow of a Risk Desk at a major investment bank (e.g., J.P. Morgan, Goldman Sachs) or Asset Manager. The objective is to identify firms with high "Tail Risk" by integrating:
- Financial Fundamentals: Solvency and liquidity metrics.
- Market Indicators: Historical volatility as a proxy for market sentiment.
- ESG Integration: E, S, and G scores as non-traditional risk drivers.

Data Strategy & Features
The dataset is generated using a Latent Risk Factor model, ensuring realistic correlations between governance quality, financial health, and tail risk events.
- Financial Features: ebitda_margin, debt_to_ebitda (Leverage), interest_coverage (Liquidity).
- Market Features: vol_90d (90-day realized volatility).
- ESG Scores: Individual $E$, $S$, and $G$ scores + a weighted Composite_ESG_Score.
- Target Variable (target_risk_12m): A binary flag (1 = High Risk) representing a probability of severe downside events (drawdown > 30%) or financial distress.

Preliminary Key Findings (EDA Phase)
The Exploratory Data Analysis (EDA) has already confirmed a high-conviction signal:
- Statistical Validation: An independent T-test on the ESG_score between "Safe" and "High Risk" groups yielded a p-value of $3.2 \times 10^{-145}$. This indicates that the difference in ESG performance is statistically indisputable.
- Governance Alpha: G_score shows a strong negative correlation with the risk target, confirming that firms with poor oversight are significantly more prone to financial distress.
- Target Distribution: The dataset maintains a realistic 24.2% default/risk rate, providing a robust framework for training imbalanced machine learning models.

Tech StackLanguage: 
- Python 3.x
- Libraries: pandas, numpy, seaborn, matplotlib, scipy (statistical testing)
- Modeling (Upcoming): scikit-learn, XGBoost
- Explainability: SHAP (Shapley Additive Explanations)

Project Roadmap
- [x] Phase 0: Build a robust, correlated synthetic dataset (src/generate_dataset.py).
- [x] Phase 1: Perform rigorous Exploratory Data Analysis and statistical testing (notebooks/01_eda.ipynb).
- [ ] Phase 2: Train Econometric (Logit/Probit) and Machine Learning (XGBoost) models.
- [ ] Phase 3: Implement SHAP values for model explainability (interpreting risk drivers).
- [ ] Phase 4: Publish a Final Risk Report and Dashboard.
