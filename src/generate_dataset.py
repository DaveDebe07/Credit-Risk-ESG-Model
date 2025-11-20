import pandas as pd
import numpy as np

# Number of companies (can be expanded later)
n = 30  

np.random.seed(42)

data = {
    "company": [f"Firm_{i+1}" for i in range(n)],
    "ebitda_margin": np.random.uniform(0.05, 0.30, n),
    "debt_to_ebitda": np.random.uniform(0.5, 4.0, n),
    "interest_coverage": np.random.uniform(2, 20, n),
    "vol_90d": np.random.uniform(0.15, 0.50, n),
    "max_drawdown": np.random.uniform(-0.60, -0.10, n),
    "E_score": np.random.randint(40, 90, n),
    "S_score": np.random.randint(35, 85, n),
    "G_score": np.random.randint(30, 95, n),
}

df = pd.DataFrame(data)

# Create ESG composite score
df["ESG_score"] = (df["E_score"] * 0.4 +
                   df["S_score"] * 0.3 +
                   df["G_score"] * 0.3)

# Create binary target
df["target_risk_12m"] = (df["max_drawdown"] < -0.30).astype(int)

# Save file
output_path = "../data/processed/features_with_target.csv"
df.to_csv(output_path, index=False)

print("Dataset saved to:", output_path)
print(df.head())
