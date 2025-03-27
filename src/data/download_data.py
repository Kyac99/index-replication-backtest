#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour télécharger les données historiques des indices et leurs composants
à partir de Yahoo Finance ou d'autres sources.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import time

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'data_download.log')),
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


def download_index_data(index_ticker, start_date=None, end_date=None, interval='1d'):
    """
    Télécharge les données historiques d'un indice.
    
    Parameters:
    -----------
    index_ticker : str
        Symbole de l'indice (ex: ^GSPC pour S&P 500)
    start_date : str, optional
        Date de début au format 'YYYY-MM-DD'
    end_date : str, optional
        Date de fin au format 'YYYY-MM-DD'
    interval : str, optional
        Intervalle de temps ('1d', '1wk', '1mo')
        
    Returns:
    --------
    pandas.DataFrame
        Données historiques de l'indice
    """
    # Dates par défaut: 10 ans jusqu'à aujourd'hui
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    logger.info(f"Téléchargement des données pour l'indice {index_ticker} de {start_date} à {end_date}")
    
    try:
        index_data = yf.download(index_ticker, start=start_date, end=end_date, interval=interval)
        logger.info(f"Données téléchargées avec succès: {index_data.shape[0]} entrées")
        return index_data
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement des données pour {index_ticker}: {e}")
        return None


def get_index_components(index_name):
    """
    Obtient les composants d'un indice spécifique.
    
    Parameters:
    -----------
    index_name : str
        Nom de l'indice ('S&P500', 'CAC40', etc.)
        
    Returns:
    --------
    list
        Liste des tickers des composants de l'indice
    """
    index_components = {
        'S&P500': '^GSPC',
        'CAC40': '^FCHI',
        'EUROSTOXX50': '^STOXX50E'
    }
    
    if index_name == 'S&P500':
        # Pour le S&P 500, nous pouvons utiliser Wikipedia
        try:
            tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            df = tables[0]
            tickers = df['Symbol'].tolist()
            logger.info(f"Composants du S&P 500 récupérés: {len(tickers)} éléments")
            return tickers
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des composants du S&P 500: {e}")
            return []
    
    elif index_name == 'CAC40':
        # Pour le CAC 40, nous pouvons utiliser une liste prédéfinie
        # Source: https://www.boursier.com/indices/composition/cac-40-FR0003500008,FR.html
        cac40_tickers = [
            'AI.PA', 'AIR.PA', 'ALO.PA', 'ATO.PA', 'BN.PA', 'CAP.PA', 'CA.PA', 
            'CS.PA', 'ENGI.PA', 'EN.PA', 'EL.PA', 'ERF.PA', 'HO.PA', 'KER.PA', 
            'LR.PA', 'LIN.PA', 'OR.PA', 'MC.PA', 'ML.PA', 'ORA.PA', 'RI.PA', 
            'PUB.PA', 'RMS.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAINT-GOBAIN.PA', 
            'SAN.PA', 'SU.PA', 'GLE.PA', 'STLA.PA', 'STM.PA', 'TEP.PA', 'TTE.PA', 
            'VIE.PA', 'DG.PA', 'VIV.PA', 'WLN.PA'
        ]
        logger.info(f"Composants du CAC 40 récupérés: {len(cac40_tickers)} éléments")
        return cac40_tickers
    
    # Pour EUROSTOXX50
    elif index_name == 'EUROSTOXX50':
        # Liste des composants de l'EURO STOXX 50
        eurostoxx50_tickers = [
            'ABI.BR', 'AD.AS', 'ADY.PA', 'AIR.PA', 'ALV.DE', 'ASML.AS', 'AXA.PA', 
            'BAS.DE', 'BAYN.DE', 'BBVA.MC', 'BMW.DE', 'BN.PA', 'BNP.PA', 'CRH.AS', 
            'CS.PA', 'DGE.PA', 'DPW.DE', 'DTE.DE', 'ENEL.MI', 'ENI.MI', 'EL.PA', 
            'FLTR.AS', 'FP.PA', 'IBE.MC', 'IFX.DE', 'ISP.MI', 'ITX.MC', 'KER.PA', 
            'LIN.DE', 'MC.PA', 'MUV2.DE', 'NOKIA.HE', 'OR.PA', 'ORA.PA', 'PHIA.AS', 
            'PRX.AS', 'SAF.PA', 'SAN.MC', 'SAP.DE', 'SHL.DE', 'SIE.DE', 'SU.PA',
            'TEF.MC', 'VIV.PA', 'VOW3.DE'
        ]
        logger.info(f"Composants de l'EURO STOXX 50 récupérés: {len(eurostoxx50_tickers)} éléments")
        return eurostoxx50_tickers
    
    # Ajouter d'autres indices au besoin
    logger.warning(f"Aucune méthode implémentée pour récupérer les composants de l'indice {index_name}")
    return []


def download_components_data(components, start_date=None, end_date=None, interval='1d'):
    """
    Télécharge les données historiques pour tous les composants d'un indice.
    
    Parameters:
    -----------
    components : list
        Liste des tickers des composants
    start_date : str, optional
        Date de début au format 'YYYY-MM-DD'
    end_date : str, optional
        Date de fin au format 'YYYY-MM-DD'
    interval : str, optional
        Intervalle de temps ('1d', '1wk', '1mo')
        
    Returns:
    --------
    dict
        Dictionnaire de DataFrames avec les données de chaque composant
    """
    # Dates par défaut: 10 ans jusqu'à aujourd'hui
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    components_data = {}
    failed_components = []
    
    logger.info(f"Téléchargement des données pour {len(components)} composants de {start_date} à {end_date}")
    
    for ticker in tqdm(components, desc="Téléchargement des composants"):
        try:
            # Ajout d'un délai pour éviter d'être bloqué par Yahoo Finance
            time.sleep(0.5)
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            if data.empty:
                logger.warning(f"Aucune donnée trouvée pour {ticker}")
                failed_components.append(ticker)
                continue
            components_data[ticker] = data
            logger.debug(f"Données téléchargées pour {ticker}: {data.shape[0]} entrées")
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement des données pour {ticker}: {e}")
            failed_components.append(ticker)
    
    logger.info(f"Téléchargement terminé. {len(components_data)} composants téléchargés, {len(failed_components)} échecs")
    if failed_components:
        logger.warning(f"Composants non téléchargés: {failed_components}")
    
    return components_data


def save_data(data, filename, directory='data/raw'):
    """
    Sauvegarde les données dans un fichier.
    
    Parameters:
    -----------
    data : pandas.DataFrame or dict
        Données à sauvegarder
    filename : str
        Nom du fichier (sans extension)
    directory : str, optional
        Répertoire de sauvegarde
    """
    create_directory_if_not_exists(directory)
    filepath = os.path.join(directory, f"{filename}.parquet")
    
    try:
        if isinstance(data, pd.DataFrame):
            data.to_parquet(filepath)
            logger.info(f"Données sauvegardées: {filepath}")
        elif isinstance(data, dict):
            # Cas pour les données des composants
            # On crée un sous-dossier pour les composants
            component_dir = os.path.join(directory, filename)
            create_directory_if_not_exists(component_dir)
            
            for ticker, df in data.items():
                # Remplacer les caractères spéciaux dans les noms de fichiers
                safe_ticker = ticker.replace('^', '').replace('.', '_')
                component_path = os.path.join(component_dir, f"{safe_ticker}.parquet")
                df.to_parquet(component_path)
            
            logger.info(f"Données des {len(data)} composants sauvegardées dans {component_dir}")
        else:
            logger.error(f"Type de données non pris en charge: {type(data)}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des données: {e}")


def main():
    """
    Fonction principale pour télécharger toutes les données nécessaires.
    """
    # Créer le dossier de logs s'il n'existe pas
    create_directory_if_not_exists('logs')
    
    # Configuration
    indices = {
        'S&P500': '^GSPC',
        'CAC40': '^FCHI',
        'EUROSTOXX50': '^STOXX50E'
    }
    
    start_date = '2015-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Télécharger les données des indices
    for index_name, index_ticker in indices.items():
        logger.info(f"Traitement de l'indice {index_name}")
        
        # Télécharger les données de l'indice
        index_data = download_index_data(index_ticker, start_date, end_date)
        if index_data is not None:
            save_data(index_data, f"{index_name}_index", 'data/raw/indices')
        
        # Télécharger les données des composants
        components = get_index_components(index_name)
        if components:
            logger.info(f"Téléchargement des données pour {len(components)} composants de {index_name}")
            components_data = download_components_data(components, start_date, end_date)
            save_data(components_data, f"{index_name}_components", 'data/raw/components')


if __name__ == "__main__":
    main()
