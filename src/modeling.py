import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier

# --- 1. CARICAMENTO DATI ---
current_script = Path(__file__).resolve()
project_root = current_script.parent.parent
data_path = project_root / "Dataset" / "features_with_target.csv"

print(f"📂 Caricamento dati da: {data_path}")
df = pd.read_csv(data_path)

# --- 2. PREPARAZIONE FEATURE ---
# Escludiamo l'ID dell'azienda e il target. 
# Escludiamo anche ESG_score per vedere se E, S, G pesano diversamente
X = df.drop(columns=['company', 'target_risk_12m', 'ESG_score'])
y = df['target_risk_12m']

# Suddivisione Training (80%) e Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling (necessario per la Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. TRAINING: LOGISTIC REGRESSION ---
print("⚙️ Training Logistic Regression...")
logit = LogisticRegression()
logit.fit(X_train_scaled, y_train)
y_pred_logit = logit.predict(X_test_scaled)
y_prob_logit = logit.predict_proba(X_test_scaled)[:, 1]

# --- 4. TRAINING: XGBOOST (Quant Standard) ---
print("⚙️ Training XGBoost...")
xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

# --- 5. VALUTAZIONE ---
print("\n" + "="*30)
print("📊 RISULTATI MODELLO")
print("="*30)
print(f"ROC-AUC Logistic Regression: {roc_auc_score(y_test, y_prob_logit):.4f}")
print(f"ROC-AUC XGBoost:             {roc_auc_score(y_test, y_prob_xgb):.4f}")
print("="*30)

# --- 6. VISUALIZZAZIONE: FEATURE IMPORTANCE ---
plt.figure(figsize=(10, 6))
# Prendiamo l'importanza delle feature da XGBoost
importances = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=True)
importances.plot(kind='barh', color='teal')
plt.title("Quali fattori guidano il Rischio? (XGBoost Feature Importance)")
plt.xlabel("Importanza relativa")
plt.tight_layout()

# Salviamo il grafico nella cartella reports/figures (creala se non esiste)
figures_dir = project_root / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(figures_dir / "feature_importance.png")
print(f"\n✅ Grafico salvato in: {figures_dir / 'feature_importance.png'}")

plt.show()