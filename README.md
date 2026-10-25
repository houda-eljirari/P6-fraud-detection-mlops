# P6 — Détection de fraude avec supervision avancée

## Contexte

Projet réalisé dans le cadre du module **DevOps et MLOps**.

L'objectif est de mettre en place une chaîne reproductible, versionnée,
conteneurisée et automatisée pour la détection de fraude sur des données
tabulaires.

Le projet ne vise pas uniquement la performance du modèle. L'objectif
principal est de construire une chaîne MLOps complète pouvant être
reproduite par une autre personne.

## Objectifs

- Développer un modèle de détection de fraude.
- Exposer le modèle via une API de scoring.
- Suivre les expériences avec MLflow.
- Versionner les données et artefacts avec DVC.
- Mettre en place une supervision des métriques.
- Détecter les dérives de données (data drift).
- Mettre en place des alertes en cas de dégradation.
- Automatiser les contrôles avec CI.

## Technologies prévues

- Python
- scikit-learn
- Pandas
- MLflow
- DVC
- FastAPI
- Docker
- Prometheus
- Grafana
- GitHub Actions

## Architecture prévue

```
Dataset
   ↓
Préparation des données
   ↓
Modèle de détection de fraude
   ↓
MLflow
   ↓
FastAPI
   ↓
Prometheus
   ↓
Grafana
   ↓
Supervision avancée
   ├── Performance du modèle
   ├── Data Drift
   └── Alertes
```
## Structure du projet
```
fraud-detection-mlops/
├── .github/
│   └── workflows/
├── data/
│   ├── raw/
│   └── processed/
├── docker/
├── monitoring/
│   ├── grafana/
│   └── prometheus/
├── notebooks/
├── src/
│   ├── api/
│   ├── data/
│   ├── models/
│   └── monitoring/
├── tests/
├── .gitignore
├── README.md
├── requirements.txt
└── docker-compose.yml
```

## Reproductibilité

Le projet est conçu pour permettre la reproduction de la chaîne à partir
du dépôt Git.

Les données seront versionnées avec DVC et les dépendances seront
documentées dans requirements.txt.

## Supervision

La supervision avancée prévue comprend :

métriques de l'API ;
performance du modèle lorsque les labels sont disponibles ;
suivi du taux de fraude ;
détection de data drift ;
alertes en cas de dégradation.
## Données

| | |
|---|---|
| Jeu de données | Bank Account Fraud (BAF) — fichier `Base.csv` |
| Nature | Synthétique, généré à partir d'un jeu réel anonymisé (fraude à l'ouverture de compte) |
| Taille | 1 000 000 de lignes, 32 colonnes, environ 1,1 % de fraudes |
| Source | https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 |
| Licence | CC BY-NC-SA 4.0 (usage non commercial) — à confirmer sur la page Kaggle le jour de la remise |
| Référence | Jesus et al., « Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation », NeurIPS 2022 (Datasets and Benchmarks). DOI : *à compléter depuis la page Kaggle* |
| Modifications | Les valeurs `-1` de 5 colonnes sont traitées comme des valeurs manquantes ; découpage temporel par mois (mois 0 à 5 : entraînement, mois 6 et 7 : test). Aucune donnée n'est stockée dans Git. |

Les données sont versionnées avec DVC (`data/raw/baf/Base.csv.dvc`). Remote : Google Drive.

## Lancement

### Prérequis
- Git
- Docker Desktop démarré
- (optionnel, pour les tests en local) Python 3.12 et `pip install -r requirements.txt`

### 1. Récupérer le dépôt
```bash
git clone https://github.com/houda-eljirari/P6-fraud-detection-mlops.git
cd P6-fraud-detection-mlops
```

### 2. Récupérer les données (deux possibilités)

**A. Depuis le remote DVC** (nécessite l'accès au Google Drive du projet) :
```bash
dvc pull
```

**B. Depuis Kaggle** (sans accès au remote) : télécharger `Base.csv` depuis la page Kaggle ci-dessus,
le placer dans `data/raw/baf/Base.csv`, puis vérifier son intégrité :
```bash
dvc status
```
Le fichier est accepté si son empreinte correspond à celle de `data/raw/baf/Base.csv.dvc`.

### 3. Lancer le pipeline
```bash
docker compose up --build
```
Le conteneur exécute `dvc repro --force` : préparation des données, entraînement, évaluation.
Résultat attendu à la fin : les métriques `pr_auc` (≈ 0,156), `roc_auc` (≈ 0,877) et `recall_at_5pct_fpr` (≈ 0,49).

### 4. Lancer les tests
```bash
pip install -r requirements.txt
pytest
```

### 5. Consulter les expériences MLflow
Les expériences lancées dans Docker sont enregistrées dans `mlflow_docker.db` :
```bash
mlflow ui --backend-store-uri sqlite:///mlflow_docker.db
```
Puis ouvrir http://localhost:5000.

### Variables d'environnement
| Variable | Rôle | Valeur par défaut |
|---|---|---|
| `MLFLOW_TRACKING_URI` | Emplacement du suivi des expériences | `sqlite:///mlflow.db` (local), `sqlite:///mlflow_docker.db` (Docker) |

Aucun secret n'est stocké dans le dépôt. Les identifiants OAuth du remote DVC sont dans `.dvc/config.local`, ignoré par Git.

### Limites connues
- `dvc pull` exige l'accès au Google Drive du projet ; sur une autre machine, utiliser l'option B.
- Dans Docker, le tag `git_commit` des runs MLflow vaut `unknown` (Git n'est pas installé dans l'image).
- Le modèle est volontairement simple (régression logistique) : l'objectif est la chaîne reproductible.
