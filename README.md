# P6 — Détection de fraude avec supervision avancée

![CI](https://github.com/houda-eljirari/P6-fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)

Projet de fin de module DevOps / MLOps : une chaîne **reproductible, versionnée, conteneurisée et automatisée** pour la détection de fraude sur des données tabulaires.

## Contexte

Le projet ne vise pas uniquement la performance du modèle. L'objectif principal est de construire une chaîne MLOps complète, qu'une autre personne peut reproduire à partir du dépôt : données versionnées, entraînement reproductible, suivi des expériences, exécution dans un conteneur, tests automatiques, puis supervision du modèle en production.

## Objectifs

- [x] Versionner les données et artefacts avec DVC.
- [x] Construire un pipeline reproductible (préparation, entraînement).
- [x] Suivre les expériences avec MLflow.
- [x] Exécuter la chaîne dans un conteneur Docker.
- [x] Automatiser les contrôles avec une CI (tests unitaires, validation du modèle, lint).
- [ ] Exposer le modèle via une API de scoring (FastAPI).
- [ ] Mettre en place une supervision des métriques (Prometheus, Grafana).
- [ ] Détecter les dérives de données (data drift).
- [ ] Mettre en place des alertes en cas de dégradation.

## Technologies

| Rôle | Outils |
|---|---|
| Langage et ML | Python, pandas, scikit-learn |
| Versionnement du code | Git, GitHub |
| Versionnement des données | DVC (remote : Google Drive) |
| Suivi des expériences | MLflow |
| Conteneurisation | Docker, Docker Compose |
| Qualité et automatisation | pytest, ruff, GitHub Actions |
| À venir | FastAPI, Prometheus, Grafana |

## Architecture

État actuel :

```
GitHub (code, dvc.yaml, params.yaml, dvc.lock, CI)
   │
   ▼
Base.csv (versionné par DVC, remote Google Drive)
   │
   ▼
[prepare] ──► train.csv / test.csv
   │
   ▼
[train] ──► model.joblib + metrics.json
   │
   ▼
MLflow : runs, paramètres, métriques, modèle

L'ensemble s'exécute dans un conteneur Docker (docker compose up).
```

Cible (à venir) :

```
Modèle ──► FastAPI (/predict) ──► Prometheus ──► Grafana
                                                    ├── Performance du modèle
                                                    ├── Data drift
                                                    └── Alertes
```

## Structure du projet

```
P6-fraud-detection-mlops/
├── .github/workflows/ci.yml     # CI : lint, tests, build Docker
├── data/
│   ├── raw/baf/Base.csv.dvc     # pointeur DVC (le CSV n'est pas dans Git)
│   └── processed/               # train.csv, test.csv (générés, ignorés par Git)
├── docker/Dockerfile            # image multi-stage, utilisateur non-root
├── metrics/metrics.json         # métriques du dernier entraînement
├── models/                      # model.joblib (généré, versionné par DVC)
├── monitoring/                  # Prometheus, Grafana (à venir)
├── notebooks/
├── src/
│   ├── api/                     # API de scoring (à venir)
│   ├── data/                    # prepare.py, check_missing.py
│   ├── models/                  # train.py
│   └── monitoring/              # métriques, dérive (à venir)
├── tests/                       # test_prepare.py, test_model.py
├── docker-compose.yml
├── dvc.yaml                     # définition du pipeline
├── dvc.lock                     # empreintes de la dernière exécution
├── params.yaml                  # tous les réglages
├── pytest.ini
├── requirements.txt
└── README.md
```

## Données

| | |
|---|---|
| Jeu de données | Bank Account Fraud (BAF), fichier `Base.csv` |
| Nature | Synthétique, généré à partir d'un jeu réel anonymisé (fraude à l'ouverture de compte bancaire) |
| Taille | 1 000 000 de lignes, 32 colonnes, environ 1,1 % de fraudes |
| Cible | `fraud_bool` (1 = fraude, 0 = légitime) |
| Temps | colonne `month` (0 à 7), prévue pour la validation temporelle |
| Source | https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 |
| Ressources des auteurs | https://github.com/feedzai/bank-account-fraud · https://arxiv.org/abs/2211.13358 |
| Licence | CC BY-NC-SA 4.0 — https://creativecommons.org/licenses/by-nc-sa/4.0/ (consultée le 25/10/2026) |

**Modifications apportées aux données** : les valeurs `-1` de cinq colonnes (`prev_address_months_count`, `current_address_months_count`, `bank_months_count`, `session_length_in_minutes`, `device_distinct_emails_8w`) sont traitées comme des valeurs manquantes ; les données sont découpées par mois (mois 0 à 5 : entraînement, mois 6 et 7 : test). Aucune donnée n'est stockée dans Git.

Les données sont versionnées avec DVC (`data/raw/baf/Base.csv.dvc`). Le projet est strictement non commercial.

### Citation

Jesus, S., Pombal, J., Alves, D., Cruz, A., Saleiro, P., Ribeiro, R. P., Gama, J., & Bizarro, P. (2022).
*Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation.*
Advances in Neural Information Processing Systems.

## Pipeline

Défini dans `dvc.yaml`, réglé par `params.yaml` :

1. **`prepare`** (`src/data/prepare.py`) : remplace les `-1` par des valeurs manquantes puis découpe par mois. Sorties : `data/processed/train.csv` (794 989 lignes) et `test.csv` (205 011 lignes).
2. **`train`** (`src/models/train.py`) : pré-traitement (imputation par la médiane avec indicateur de manque, normalisation, encodage one-hot) puis régression logistique à classes équilibrées. Le modèle, les paramètres et les métriques sont enregistrés dans MLflow. Sorties : `models/model.joblib` et `metrics/metrics.json`.

Le split est **temporel** (on entraîne sur le passé, on teste sur le futur) et les métriques sont adaptées au fort déséquilibre de classes : PR-AUC, ROC-AUC et rappel à 5 % de faux positifs. L'accuracy n'est pas utilisée, car elle est trompeuse avec environ 1 % de fraudes.

## Résultats actuels (modèle de référence)

| Métrique | Valeur |
|---|---|
| PR-AUC | 0,156 (le hasard donnerait environ 0,014) |
| ROC-AUC | 0,877 |
| Rappel à 5 % de faux positifs | 0,491 |

Le modèle est volontairement simple : il sert de référence mesurée à améliorer. L'objectif du projet est la chaîne reproductible, pas la performance.

## Lancement

### Prérequis

- Git
- Docker Desktop **démarré**
- (optionnel, pour les tests en local) Python 3.12 et un environnement virtuel

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

**B. Depuis Kaggle** (sans accès au remote) : télécharger `Base.csv` depuis la page Kaggle ci-dessus, le placer dans `data/raw/baf/Base.csv`, puis vérifier son intégrité :

```bash
dvc status
```

Le fichier est accepté si son empreinte correspond à celle de `data/raw/baf/Base.csv.dvc`.

### 3. Lancer le pipeline

```bash
docker compose up --build
```

Le conteneur exécute `dvc repro --force` : préparation des données puis entraînement. Résultat attendu à la fin : `pr_auc` ≈ 0,156, `roc_auc` ≈ 0,877, `recall_at_5pct_fpr` ≈ 0,49.

### 4. Lancer les tests et le linter

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install ruff
pytest
ruff check src tests
```

Résultat attendu : `8 passed` et `All checks passed!`. Les tests utilisent des données synthétiques générées à la volée : ils n'ont pas besoin de `Base.csv`.

### 5. Consulter les expériences MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow_docker.db    # exécution dans Docker
mlflow ui --backend-store-uri sqlite:///mlflow.db           # exécution locale
```

Puis ouvrir http://localhost:5000.

### Variables d'environnement

| Variable | Rôle | Valeur par défaut |
|---|---|---|
| `MLFLOW_TRACKING_URI` | Emplacement du suivi des expériences | `sqlite:///mlflow.db` (local), `sqlite:///mlflow_docker.db` (Docker) |

Aucun secret n'est stocké dans le dépôt. Les identifiants OAuth du remote DVC sont dans `.dvc/config.local`, ignoré par Git.

### Dépannage

- **`dvc pull` ou `dvc push` échoue avec `module 'lib' has no attribute ...`** : versions incompatibles de `pyOpenSSL` et `cryptography`. Exécuter `pip install --upgrade pyOpenSSL cryptography`.
- **`dvc push` échoue avec `WinError 10054`** : coupure réseau, relancer la commande.
- **Google affiche « Cette application est bloquée »** : l'identifiant OAuth par défaut de DVC est bloqué. Créer son propre identifiant OAuth dans Google Cloud Console et le renseigner avec `dvc remote modify --local`.
- **`Base.csv.dvc` apparaît modifié (`isexec: true`) après un passage dans Docker** : artefact du montage de volume sous Windows, ne pas le commiter.

## Reproductibilité

Tout ce qui définit le résultat est versionné : le code, les réglages (`params.yaml`) et la définition du pipeline (`dvc.yaml`) dans Git, les données dans DVC, et l'environnement d'exécution est figé par Docker. Relancé depuis un clone vierge, le pipeline redonne exactement les mêmes métriques.

## Supervision (à venir)

La supervision avancée prévue comprend : métriques de l'API, performance du modèle lorsque les labels sont disponibles, suivi du taux de fraude, détection de data drift (PSI entre les données de référence et les données récentes) et alertes en cas de dégradation.

## État du projet

| Jalon | Contenu | État |
|---|---|---|
| J2 | Dépôt, structure, README, DVC initialisé | Terminé |
| J3 | Pipeline de bout en bout, conteneurisé | Terminé, validé depuis un clone vierge |
| J4 | Données versionnées, expériences tracées, tests et CI | En cours de validation |
| J5 | API, supervision, dérive, alertes, documentation | À faire |
| J6 | Démonstration et remise | À faire |

## Limites connues

- Le modèle est volontairement simple (régression logistique) : l'objectif est la chaîne reproductible.
- `dvc pull` exige l'accès au Google Drive du projet ; sur une autre machine, utiliser l'option B.
- Dans Docker, le tag `git_commit` des runs MLflow vaut `unknown` (Git n'est pas installé dans l'image).
- L'API, la supervision, la détection de dérive et les alertes ne sont pas encore implémentées.
