import pandas as pd
import numpy as np
import os

# --- CONFIGURAZIONE PERCORSO SPECIFICO ---
# Usiamo 'r' davanti alla stringa per gestire correttamente i backslash di Windows
output_path = r"C:\Users\utente\Desktop\Progetti_Finanza_Distinzione\Piano_Carriera_2030\Dataset\features_with_target.csv"

# Creiamo la cartella 'Dataset' se non esiste ancora
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Cartella creata: {output_dir}")

# --- LOGICA DI GENERAZIONE (Wall Street Style) ---
np.random.seed(42)
n = 1000 

# Generazione fattore di rischio latente
latent_risk = np.random.normal(0, 1, n)

data = {
    "company": [f"Firm_{i+1}" for i in range(n)],
    "ebitda_margin": 0.20 - (0.05 * latent_risk) + np.random.normal(0, 0.02, n),
    "debt_to_ebitda": 2.5 + (0.8 * latent_risk) + np.random.normal(0, 0.3, n),
    "interest_coverage": 10 - (3 * latent_risk) + np.random.normal(0, 1, n),
    "vol_90d": 0.25 + (0.10 * latent_risk) + np.random.normal(0, 0.05, n),
    "E_score": 65 - (10 * latent_risk) + np.random.normal(0, 5, n),
    "S_score": 60 - (8 * latent_risk) + np.random.normal(0, 5, n),
    "G_score": 70 - (15 * latent_risk) + np.random.normal(0, 5, n),
}

df = pd.DataFrame(data)

# Clipping e calcoli finali
df["ebitda_margin"] = df["ebitda_margin"].clip(0.01, 0.50)
df["debt_to_ebitda"] = df["debt_to_ebitda"].clip(0.1, 15.0)
df["interest_coverage"] = df["interest_coverage"].clip(0.5, 40.0)
df["E_score"] = df["E_score"].clip(0, 100).astype(int)
df["S_score"] = df["S_score"].clip(0, 100).astype(int)
df["G_score"] = df["G_score"].clip(0, 100).astype(int)
df["ESG_score"] = (df["E_score"] * 0.4 + df["S_score"] * 0.3 + df["G_score"] * 0.3)

# Calcolo Target (Rischio alto = 1)
prob_risk = 1 / (1 + np.exp(-(2 * latent_risk + np.random.normal(0, 0.5, n))))
df["target_risk_12m"] = (prob_risk > 0.8).astype(int)

# --- SALVATAGGIO ---
try:
    df.to_csv(output_path, index=False)
    print(f"✅ Successo! Dataset salvato in:\n{output_path}")
    print(f"\nDistribuzione Target (High Risk = 1):")
    print(df['target_risk_12m'].value_counts(normalize=True))
except Exception as e:
    print(f"❌ Errore durante il salvataggio: {e}")
