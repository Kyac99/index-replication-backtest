@echo off
echo Création des répertoires pour le projet de réplication d'indice...

mkdir data\raw\indices
mkdir data\raw\components
mkdir data\processed\indices
mkdir data\processed\components
mkdir data\processed\weights
mkdir logs
mkdir results

echo Répertoires créés avec succès !
echo.
echo Structure du projet:
echo - data\raw\indices: Données brutes des indices
echo - data\raw\components: Données brutes des composants
echo - data\processed\indices: Données traitées des indices
echo - data\processed\components: Données traitées des composants
echo - data\processed\weights: Données de pondération
echo - logs: Fichiers de journalisation
echo - results: Résultats des backtests
echo.
echo Vous pouvez maintenant lancer le téléchargement des données:
echo python main.py download --index CAC40 --years 5
echo.
pause
