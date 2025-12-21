import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

# ==========================================
# 1. CONFIGURAZIONE E CARICAMENTO DATI
# ==========================================

# Percorso del file (Raw string per Windows)
file_path = r"C:\Users\utente\Desktop\Progetti_Finanza_Distinzione\Piano_Carriera_2030\Dataset\features_with_target.csv"

def run_eda():
    # Verifica se il file esiste
    if not os.path.exists(file_path):
        print(f"❌ ERRORE: Il file non è stato trovato in: {file_path}")
        print("Assicurati di aver fatto girare lo script 'generate_dataset.py' prima.")
        return

    # Caricamento dati
    df = pd.read_csv(file_path)
    print("✅ Dataset caricato correttamente.")
    print(f"Dimensioni: {df.shape[0]} righe, {df.shape[1]} colonne\n")

    # ==========================================
    # 2. ANALISI DESCRITTIVA
    # ==========================================
    print("--- Riepilogo Statistico ---")
    print(df.describe().T) # Trasposta per migliore leggibilità
    print("\n--- Distribuzione Target (0=Safe, 1=Risk) ---")
    print(df['target_risk_12m'].value_counts(normalize=True))

    # ==========================================
    # 3. VISUALIZZAZIONE PROFESSIONALE
    # ==========================================
    
    # Impostazioni stile Seaborn
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.dpi'] = 100

    # GRAFICO 1: Matrice di Correlazione (Heatmap)
    plt.figure(figsize=(12, 8))
    # Escludiamo la colonna 'company' per il calcolo
    corr = df.drop(columns=['company']).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool)) # Maschera la parte superiore
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='RdYlGn', center=0, linewidths=.5)
    plt.title("Correlation Matrix: ESG & Financials vs Risk Target", fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()

    # GRAFICO 2: Confronto Gruppi (Boxplots)
    # Analizziamo le 4 variabili più importanti
    features_to_compare = ['ESG_score', 'G_score', 'debt_to_ebitda', 'ebitda_margin']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, col in enumerate(features_to_compare):
        sns.boxplot(x='target_risk_12m', y=col, data=df, ax=axes[i], palette="Set2")
        axes[i].set_title(f'Distribution of {col} by Risk Group', fontsize=12)
        axes[i].set_xlabel("Risk Group (0=Low, 1=High)")

    plt.suptitle("Financial & ESG Differentiation for Risk Assessment", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # ==========================================
    # 4. TEST DI SIGNIFICATIVITÀ STATISTICA
    # ==========================================
    print("\n--- Analisi Inferenziale (T-Test) ---")
    
    # Gruppi
    high_risk_esg = df[df['target_risk_12m'] == 1]['ESG_score']
    low_risk_esg = df[df['target_risk_12m'] == 0]['ESG_score']

    # T-test a due campioni indipendenti
    t_stat, p_val = stats.ttest_ind(high_risk_esg, low_risk_esg)

    print(f"Test sul punteggio ESG:")
    print(f" - T-statistic: {t_stat:.4f}")
    print(f" - P-value: {p_val:.4e}")

    if p_val < 0.05:
        print(">>> RISULTATO: Statisticamente Significativo. L'ESG score influenza il profilo di rischio.")
    else:
        print(">>> RISULTATO: Non Significativo.")

if __name__ == "__main__":
    run_eda()
