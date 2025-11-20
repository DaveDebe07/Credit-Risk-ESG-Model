# Notebook 01 — Exploratory Data Analysis (EDA)

## Objective
Understand the structure of the dataset and detect patterns related to credit risk.
This notebook will focus on:
- Summary statistics (financial & ESG metrics)
- Distribution plots
- Correlation analysis
- High-risk vs Low-risk comparison (based on target variable)

---

## Dataset to be loaded
From `/data/processed/`:
- `features_with_target.csv`  (to be created)
- (possible future) `company_ESG_KPIs.csv`  (real ESG data if available)

---

## Planned analysis

### 1. Overview
- Number of companies
- Missing values
- Basic stats: mean, std, min-max

### 2. Distributions
Variables to study:
- EBITDA margin
- Debt/EBITDA
- Interest coverage
- 90-day volatility
- Max drawdown
- ESG scores (E, S, G, composite)

### 3. Correlations
- Heatmap finance only
- Heatmap ESG only
- Heatmap finance + ESG
- Check for multicollinearity

### 4. Group comparison
Divide dataset into:
- High risk = target = 1
- Low risk = target = 0

Then compare groups with:
- t-test on ESG scores
- boxplot on leverage & ESG

---

## Possible insights to write later
- Are ESG scores lower in high-risk firms?
- Do financial ratios differ significantly across groups?
- Is there correlation between E/S/G?
- Which metrics show strongest predictive potential?

---

## 🧠 Next step (Notebook 02):
➡ Econometric models: logit, probit, OLS.
