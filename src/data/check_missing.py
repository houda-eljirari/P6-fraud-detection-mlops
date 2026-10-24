"""Liste les colonnes qui contiennent la valeur -1 (code de valeur manquante dans BAF)."""
import pandas as pd

df = pd.read_csv("data/raw/baf/Base.csv")
counts = (df == -1).sum()
counts = counts[counts > 0].sort_values(ascending=False)
print("Colonnes contenant -1 (nombre de lignes) :")
print(counts.to_string())
