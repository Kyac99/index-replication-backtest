# Backtest et Suivi d'un Portefeuille Répliquant un Indice

Ce projet vise à développer et évaluer des stratégies de réplication d'indices boursiers en comparant les méthodes physique et synthétique. L'objectif est d'optimiser la performance, minimiser les coûts et le tracking error, tout en analysant l'impact des frais de gestion et des dividendes.

## Aperçu

Le framework permet de:
- Télécharger les données historiques des indices et de leurs composants
- Implémenter des stratégies de réplication physique et synthétique
- Comparer les performances des différentes approches
- Analyser l'impact des frais de gestion et de transaction
- Optimiser les stratégies de rebalancement

## Caractéristiques

- **Réplication Physique**: Achat direct des composants de l'indice en respectant leur pondération
- **Réplication Synthétique**: Utilisation de swaps pour reproduire la performance de l'indice
- **Analyse Comparative**: Comparaison détaillée des deux approches (tracking error, coûts, risques)
- **Visualisation**: Génération de graphiques et tableaux pour l'analyse des résultats
- **Tests de Sensibilité**: Analyse de l'impact des paramètres clés (frais, fréquence de rebalancement)

## Structure du Projet

```
├── config/               # Fichiers de configuration
├── data/                 # Données brutes et transformées
│   ├── raw/              # Données brutes téléchargées
│   └── processed/        # Données traitées pour les backtests
├── logs/                 # Fichiers de logs
├── notebooks/            # Jupyter notebooks pour l'analyse exploratoire
├── results/              # Résultats des backtests
├── src/                  # Code source principal
│   ├── data/             # Scripts pour le téléchargement et le traitement des données
│   ├── models/           # Implémentation des stratégies de réplication
│   ├── visualization/    # Fonctions de visualisation
│   └── utils/            # Fonctions utilitaires
├── tests/                # Tests unitaires et d'intégration
├── .gitignore            # Fichiers à ignorer par Git
├── main.py               # Script principal
├── setup.bat             # Script batch pour créer la structure des dossiers (Windows)
└── requirements.txt      # Dépendances Python
```

## Installation

1. Cloner le dépôt:
```bash
git clone https://github.com/Kyac99/index-replication-backtest.git
cd index-replication-backtest
```

2. Créer un environnement virtuel et l'activer:
```bash
# Linux/macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. Installer les dépendances:
```bash
pip install -r requirements.txt
```

4. Créer les dossiers nécessaires:

### Pour Windows (méthode facile):
```
setup.bat
```
Le fichier batch créera automatiquement toute la structure de dossiers nécessaire.

### Pour Linux/macOS:
```bash
mkdir -p data/raw/indices data/raw/components data/processed/indices data/processed/components data/processed/weights logs results
```

### Pour Windows (CMD) - méthode manuelle:
```cmd
mkdir data
mkdir data\raw
mkdir data\raw\indices
mkdir data\raw\components
mkdir data\processed
mkdir data\processed\indices
mkdir data\processed\components
mkdir data\processed\weights
mkdir logs
mkdir results
```

### Pour Windows (PowerShell):
```powershell
New-Item -Path "data\raw\indices","data\raw\components","data\processed\indices","data\processed\components","data\processed\weights","logs","results" -ItemType Directory -Force
```

## Utilisation

### Via la ligne de commande

#### 1. Télécharger les données

```bash
python main.py download --index CAC40 --years 5
```

#### 2. Traiter les données

```bash
python main.py process --index CAC40
```

#### 3. Exécuter le backtest (réplication physique)

```bash
python main.py physical --index CAC40 --rebalance-frequency quarterly
```

#### 4. Exécuter le backtest (réplication synthétique)

```bash
python main.py synthetic --index CAC40 --swap-reset-frequency monthly
```

#### 5. Comparer les stratégies

```bash
python main.py compare --index CAC40
```

### Via le notebook

Vous pouvez également exécuter le notebook Jupyter pour une analyse interactive:

```bash
jupyter notebook notebooks/index_replication_demo.ipynb
```

## Exemples de Résultats

La comparaison des deux stratégies de réplication produit des visualisations et métriques permettant d'évaluer leurs performances relatives:

1. **Évolution de la valeur du portefeuille** - Montre la croissance du capital au fil du temps
2. **Rendement cumulé** - Compare le rendement du portefeuille avec celui de l'indice
3. **Tracking Error** - Mesure la précision de la réplication 
4. **Coûts** - Analyse des frais de gestion et de transaction
5. **Métriques de risque** - Volatilité, drawdown, ratio de Sharpe, etc.

## Personnalisation

Le fichier de configuration `config/default_config.yaml` permet de personnaliser les paramètres principaux:

- Indice à répliquer
- Capital initial
- Période de backtest
- Frais de gestion et de transaction
- Fréquence de rebalancement
- Format des rapports générés

## Exécution des Tests

Pour exécuter les tests unitaires:

```bash
python -m unittest discover tests
```

## Déploiement sur GitHub Pages

Vous pouvez utiliser GitHub Pages pour présenter les résultats de vos analyses:

1. Créez un dossier `docs` à la racine du projet
2. Générez des rapports HTML et des visualisations dans ce dossier
3. Activez GitHub Pages dans les paramètres du dépôt en choisissant le dossier `/docs`

Votre site sera disponible à l'adresse: `https://username.github.io/index-replication-backtest/`

## Licence

MIT
