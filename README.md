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
## État du projet

J2 — Initialisation du projet

À ce stade :

dépôt Git initialisé ;
structure du projet créée ;
DVC initialisé ;
dépendances de base définies ;
architecture cible documentée.

Les composants ML, API, monitoring et CI seront développés dans les prochaines étapes.