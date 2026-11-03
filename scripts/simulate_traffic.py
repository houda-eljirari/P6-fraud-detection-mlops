"""Envoie des transactions du jeu de test a l'API pour alimenter la supervision.

Usage : python scripts/simulate_traffic.py --n 300 --rate 5
"""
import argparse
import time

import pandas as pd

# Colonnes du fichier de test qui ne sont pas des entrees de l'API
DROP_COLUMNS = ["fraud_bool", "month"]


def rows_to_payloads(df: pd.DataFrame) -> list[dict]:
    """Transforme des lignes du jeu de test en corps de requete /predict.

    Les valeurs manquantes (NaN) deviennent null, accepte par l'API.
    """
    clean = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    return [
        {key: (None if pd.isna(value) else value) for key, value in record.items()}
        for record in clean.to_dict(orient="records")
    ]


def main() -> None:
    import httpx

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--csv", default="data/processed/test.csv")
    parser.add_argument("--n", type=int, default=300, help="nombre de requetes")
    parser.add_argument("--rate", type=float, default=5.0, help="requetes par seconde")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sample = pd.read_csv(args.csv).sample(n=args.n, random_state=args.seed)
    payloads = rows_to_payloads(sample)

    sent = flagged = errors = 0
    try:
        with httpx.Client(base_url=args.url, timeout=10.0) as client:
            for payload in payloads:
                response = client.post("/predict", json=payload)
                sent += 1
                if response.status_code == 200:
                    flagged += int(response.json()["is_fraud"])
                else:
                    errors += 1
                time.sleep(1.0 / args.rate)
    except httpx.HTTPError as exc:
        print(f"Connexion impossible a {args.url} : {exc}")
        raise SystemExit(1) from exc

    print(f"{sent} requetes envoyees, {flagged} signalees comme fraude, {errors} erreurs")


if __name__ == "__main__":
    main()