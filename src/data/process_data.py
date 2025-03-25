#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour traiter les données téléchargées et les préparer pour le backtesting.
"""

import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'data_processing.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def create_directory_if_not_exists(directory):
    """
    Crée le répertoire s'il n'existe pas déjà.
    
    Parameters:
    -----------
    directory : str
        Chemin du répertoire à créer
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Répertoire créé: {directory}")


def load_index_data(index_name, data_dir='data/raw/indices'):
    """
    Charge les données d'un indice.
    
    Parameters:
    -----------
    index_name : str
        Nom de l'indice ('S&P500', 'CAC40', etc.)
    data_dir : str, optional
        Répertoire des données brutes
        
    Returns:
    --------
    pandas.DataFrame
        Données de l'indice
    """
    file_path = os.path.join(data_dir, f"{index_name}_index.parquet")
    
    try:
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            logger.info(f"Données de l'indice {index_name} chargées: {df.shape[0]} entrées")
            return df
        else:
            logger.error(f"Fichier non trouvé: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données de l'indice {index_name}: {e}")
        return None


def load_components_data(index_name, data_dir='data/raw/components'):
    """
    Charge les données des composants d'un indice.
    
    Parameters:
    -----------
    index_name : str
        Nom de l'indice ('S&P500', 'CAC40', etc.)
    data_dir : str, optional
        Répertoire des données brutes
        
    Returns:
    --------
    dict
        Dictionnaire de DataFrames avec les données de chaque composant
    """
    component_dir = os.path.join(data_dir, f"{index_name}_components")
    
    if not os.path.exists(component_dir):
        logger.error(f"Répertoire non trouvé: {component_dir}")
        return {}
    
    components_data = {}
    file_paths = glob.glob(os.path.join(component_dir, "*.parquet"))
    
    logger.info(f"Chargement des données pour {len(file_paths)} composants de {index_name}")
    
    for file_path in file_paths:
        try:
            ticker = os.path.basename(file_path).replace('.parquet', '').replace('_', '.')
            df = pd.read_parquet(file_path)
            components_data[ticker] = df
            logger.debug(f"Données chargées pour {ticker}: {df.shape[0]} entrées")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données de {file_path}: {e}")
    
    logger.info(f"{len(components_data)} composants chargés pour {index_name}")
    return components_data


def calculate_returns(df, price_column='Adj Close'):
    """
    Calcule les rendements quotidiens et cumulés.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données de prix
    price_column : str, optional
        Nom de la colonne de prix à utiliser
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame avec colonnes supplémentaires pour les rendements
    """
    if price_column not in df.columns:
        logger.error(f"Colonne {price_column} non trouvée dans le DataFrame")
        return df
    
    # Calculer les rendements quotidiens
    df['daily_return'] = df[price_column].pct_change()
    
    # Calculer les rendements cumulés
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
    
    # Supprimer la première ligne avec des NaN
    df = df.dropna(subset=['daily_return'])
    
    return df


def create_price_matrix(components_data, price_column='Adj Close'):
    """
    Crée une matrice de prix pour tous les composants.
    
    Parameters:
    -----------
    components_data : dict
        Dictionnaire de DataFrames avec les données des composants
    price_column : str, optional
        Nom de la colonne de prix à utiliser
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame avec les prix des composants en colonnes
    """
    price_dfs = []
    
    for ticker, df in components_data.items():
        if price_column in df.columns:
            # Renommer la colonne pour inclure le ticker
            price_series = df[price_column].rename(ticker)
            price_dfs.append(price_series)
    
    if not price_dfs:
        logger.error(f"Aucune donnée de prix valide trouvée")
        return None
    
    # Concaténer toutes les séries de prix
    price_matrix = pd.concat(price_dfs, axis=1)
    logger.info(f"Matrice de prix créée: {price_matrix.shape[0]} jours, {price_matrix.shape[1]} composants")
    
    return price_matrix


def create_return_matrix(components_data, price_column='Adj Close'):
    """
    Crée une matrice de rendements pour tous les composants.
    
    Parameters:
    -----------
    components_data : dict
        Dictionnaire de DataFrames avec les données des composants
    price_column : str, optional
        Nom de la colonne de prix à utiliser
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame avec les rendements des composants en colonnes
    """
    return_dfs = []
    
    for ticker, df in components_data.items():
        if price_column in df.columns:
            # Calculer les rendements quotidiens
            returns = df[price_column].pct_change().rename(ticker)
            return_dfs.append(returns)
    
    if not return_dfs:
        logger.error(f"Aucune donnée de rendements valide trouvée")
        return None
    
    # Concaténer toutes les séries de rendements
    return_matrix = pd.concat(return_dfs, axis=1)
    
    # Supprimer la première ligne avec des NaN
    return_matrix = return_matrix.dropna(how='all')
    
    logger.info(f"Matrice de rendements créée: {return_matrix.shape[0]} jours, {return_matrix.shape[1]} composants")
    
    return return_matrix


def process_weights_data(index_name, components_data):
    """
    Traite les données de pondération des composants dans l'indice.
    
    Pour le moment, nous utilisons des pondérations simplifiées basées sur la 
    capitalisation boursière ou des pondérations égales, mais dans un système
    réel, nous récupérerions les véritables pondérations historiques.
    
    Parameters:
    -----------
    index_name : str
        Nom de l'indice
    components_data : dict
        Dictionnaire de DataFrames avec les données des composants
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame avec l'évolution historique des pondérations
    """
    # Pour simplifier, nous utilisons des pondérations égales
    # Dans un cas réel, nous utiliserions les vraies pondérations
    
    # Obtenir les dates communes à tous les composants
    common_dates = None
    
    for ticker, df in components_data.items():
        if common_dates is None:
            common_dates = set(df.index)
        else:
            common_dates = common_dates.intersection(set(df.index))
    
    common_dates = sorted(list(common_dates))
    
    # Créer un DataFrame avec des pondérations égales
    num_components = len(components_data)
    weights_df = pd.DataFrame(index=common_dates)
    
    for ticker in components_data.keys():
        weights_df[ticker] = 1.0 / num_components
    
    logger.info(f"Données de pondération créées pour {index_name}: {weights_df.shape[0]} jours, {weights_df.shape[1]} composants")
    return weights_df


def save_processed_data(data, filename, directory='data/processed'):
    """
    Sauvegarde les données traitées dans un fichier.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Données à sauvegarder
    filename : str
        Nom du fichier (sans extension)
    directory : str, optional
        Répertoire de sauvegarde
    """
    create_directory_if_not_exists(directory)
    filepath = os.path.join(directory, f"{filename}.parquet")
    
    try:
        data.to_parquet(filepath)
        logger.info(f"Données traitées sauvegardées: {filepath}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des données traitées: {e}")


def main():
    """
    Fonction principale pour traiter toutes les données téléchargées.
    """
    # Créer le dossier de logs s'il n'existe pas
    create_directory_if_not_exists('logs')
    
    # Liste des indices à traiter
    indices = ['S&P500', 'CAC40', 'EUROSTOXX50']
    
    for index_name in indices:
        logger.info(f"Traitement des données pour l'indice {index_name}")
        
        # Charger les données de l'indice
        index_data = load_index_data(index_name)
        if index_data is not None:
            # Calculer les rendements de l'indice
            index_data = calculate_returns(index_data)
            save_processed_data(index_data, f"{index_name}_index_returns", 'data/processed/indices')
        
        # Charger les données des composants
        components_data = load_components_data(index_name)
        if components_data:
            # Créer et sauvegarder la matrice de prix
            price_matrix = create_price_matrix(components_data)
            if price_matrix is not None:
                save_processed_data(price_matrix, f"{index_name}_price_matrix", 'data/processed/components')
            
            # Créer et sauvegarder la matrice de rendements
            return_matrix = create_return_matrix(components_data)
            if return_matrix is not None:
                save_processed_data(return_matrix, f"{index_name}_return_matrix", 'data/processed/components')
            
            # Traiter et sauvegarder les données de pondération
            weights_df = process_weights_data(index_name, components_data)
            save_processed_data(weights_df, f"{index_name}_weights", 'data/processed/weights')


if __name__ == "__main__":
    main()
