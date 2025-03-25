# Backtest et Suivi d'un Portefeuille Répliquant un Indice

Ce projet vise à développer et évaluer des stratégies de réplication d'indices boursiers en comparant les méthodes physique et synthétique. L'objectif est d'optimiser la performance, minimiser les coûts et le tracking error, tout en analysant l'impact des frais de gestion et des dividendes.

## Objectifs

* Implémenter une stratégie de réplication d'un indice boursier selon les méthodes physique et synthétique
* Analyser l'impact des frais de gestion et des dividendes sur la performance du portefeuille
* Évaluer le tracking error et son évolution dans le temps
* Étudier différentes stratégies de rebalancement pour minimiser les coûts
* Automatiser le suivi et la mise à jour du portefeuille

## Structure du Projet

```
├── data/                # Données brutes et transformées
├── notebooks/           # Jupyter notebooks pour l'analyse exploratoire
├── src/                 # Code source principal
│   ├── data/            # Scripts pour le téléchargement et le traitement des données
│   ├── models/          # Implémentation des stratégies de réplication
│   ├── visualization/   # Fonctions de visualisation
│   └── utils/           # Fonctions utilitaires
├── tests/               # Tests unitaires et d'intégration
├── config/              # Fichiers de configuration
└── docs/                # Documentation
```

## Méthodologie

1. **Sélection des Indices**: Analyse et sélection d'un ou plusieurs indices boursiers de référence
2. **Réplication Physique**: Achat direct des composants de l'indice en respectant leur pondération
3. **Réplication Synthétique**: Utilisation de swaps ou d'autres instruments dérivés
4. **Backtesting**: Test des stratégies sur des données historiques
5. **Optimisation du Rebalancement**: Analyse de différentes fréquences et méthodes de rebalancement
6. **Analyse des Coûts**: Évaluation de l'impact des frais de gestion et de transaction
7. **Automatisation**: Développement d'outils pour le suivi et la mise à jour du portefeuille

## Technologies

* **Langages**: Python (Pandas, NumPy, Scikit-Learn, Backtrader)
* **Sources de Données**: Yahoo Finance, API d'indices
* **Visualisation**: Matplotlib, Plotly, Seaborn

## Installation et Utilisation

```bash
# Cloner le dépôt
git clone https://github.com/Kyac99/index-replication-backtest.git
cd index-replication-backtest

# Installer les dépendances
pip install -r requirements.txt

# Exécuter les notebooks ou scripts
jupyter notebook notebooks/
```

## Licence

MIT
