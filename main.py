#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal pour exécuter les différentes tâches du projet de réplication d'indice.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Ajouter le répertoire courant au chemin de recherche Python
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'main.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def download_data(args):
    """
    Télécharge les données des indices et de leurs composants.
    """
    from src.data.download_data import download_index_data, get_index_components, download_components_data, save_data
    
    logger.info(f"Téléchargement des données pour l'indice {args.index}")
    
    # Mappage des noms d'indices vers les tickers
    index_tickers = {
        'S&P500': '^GSPC',
        'CAC40': '^FCHI',
        'EUROSTOXX50': '^STOXX50E'
    }
    
    # Vérifier si l'indice est supporté
    if args.index not in index_tickers:
        logger.error(f"Indice non supporté: {args.index}. Options disponibles: {', '.join(index_tickers.keys())}")
        return 1
    
    # Configurer les dates
    if args.start_date is None:
        args.start_date = (datetime.now().replace(day=1) - timedelta(days=365*args.years)).strftime('%Y-%m-%d')
    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Période: {args.start_date} à {args.end_date}")
    
    # Créer les répertoires de données si nécessaire
    os.makedirs(os.path.join('data', 'raw', 'indices'), exist_ok=True)
    os.makedirs(os.path.join('data', 'raw', 'components'), exist_ok=True)
    
    # Télécharger les données de l'indice
    index_ticker = index_tickers[args.index]
    index_data = download_index_data(index_ticker, args.start_date, args.end_date)
    
    if index_data is not None:
        save_data(index_data, f"{args.index}_index", os.path.join('data', 'raw', 'indices'))
    
    # Télécharger les données des composants
    if not args.no_components:
        components = get_index_components(args.index)
        if components:
            logger.info(f"Téléchargement des données pour {len(components)} composants")
            
            # Limiter le nombre de composants si spécifié
            if args.limit > 0 and args.limit < len(components):
                components = components[:args.limit]
                logger.info(f"Limitation à {args.limit} composants pour la démo")
            
            components_data = download_components_data(components, args.start_date, args.end_date)
            save_data(components_data, f"{args.index}_components", os.path.join('data', 'raw', 'components'))
    
    logger.info("Téléchargement des données terminé")
    return 0


def process_data(args):
    """
    Traite les données brutes pour préparer le backtest.
    """
    from src.data.process_data import load_index_data, load_components_data, calculate_returns, create_price_matrix, create_return_matrix, process_weights_data, save_processed_data
    
    logger.info(f"Traitement des données pour l'indice {args.index}")
    
    # Créer les répertoires de données traitées si nécessaire
    os.makedirs(os.path.join('data', 'processed', 'indices'), exist_ok=True)
    os.makedirs(os.path.join('data', 'processed', 'components'), exist_ok=True)
    os.makedirs(os.path.join('data', 'processed', 'weights'), exist_ok=True)
    
    # Charger les données brutes
    raw_index_data = load_index_data(args.index, os.path.join('data', 'raw', 'indices'))
    raw_components_data = load_components_data(args.index, os.path.join('data', 'raw', 'components'))
    
    # Traiter les données de l'indice
    if raw_index_data is not None:
        index_returns = calculate_returns(raw_index_data)
        save_processed_data(index_returns, f"{args.index}_index_returns", os.path.join('data', 'processed', 'indices'))
    
    # Traiter les données des composants
    if raw_components_data:
        # Créer et sauvegarder la matrice de prix
        price_matrix = create_price_matrix(raw_components_data)
        if price_matrix is not None:
            save_processed_data(price_matrix, f"{args.index}_price_matrix", os.path.join('data', 'processed', 'components'))
        
        # Créer et sauvegarder la matrice de rendements
        return_matrix = create_return_matrix(raw_components_data)
        if return_matrix is not None:
            save_processed_data(return_matrix, f"{args.index}_return_matrix", os.path.join('data', 'processed', 'components'))
        
        # Créer et sauvegarder les données de pondération
        weights_df = process_weights_data(args.index, raw_components_data)
        save_processed_data(weights_df, f"{args.index}_weights", os.path.join('data', 'processed', 'weights'))
    
    logger.info("Traitement des données terminé")
    return 0


def backtest_physical(args):
    """
    Exécute le backtest de la stratégie de réplication physique.
    """
    from src.models.physical_replication import PhysicalReplication
    
    logger.info(f"Backtest de réplication physique pour l'indice {args.index}")
    
    # Créer le répertoire de résultats si nécessaire
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialiser la réplication physique
    physical = PhysicalReplication(
        index_name=args.index,
        start_date=args.start_date,
        end_date=args.end_date,
        data_dir=args.data_dir,
        initial_capital=args.capital,
        management_fee=args.management_fee,
        transaction_cost=args.transaction_cost,
        rebalance_frequency=args.rebalance_frequency
    )
    
    # Exécuter le backtest
    results = physical.run_backtest()
    
    # Calculer les métriques de performance
    metrics = physical.calculate_performance_metrics(results)
    
    # Afficher les métriques principales
    logger.info(f"Tracking Error: {metrics['tracking_error']:.4%}")
    logger.info(f"Information Ratio: {metrics['information_ratio']:.4f}")
    logger.info(f"Coûts de transaction totaux: {metrics['total_transaction_costs']:.2f} €")
    
    # Sauvegarder les graphiques
    plot_path = os.path.join(args.output_dir, f"{args.index}_physical_plot.png")
    physical.plot_results(results, save_path=plot_path)
    
    logger.info(f"Graphiques sauvegardés: {plot_path}")
    return 0


def backtest_synthetic(args):
    """
    Exécute le backtest de la stratégie de réplication synthétique.
    """
    from src.models.synthetic_replication import SyntheticReplication
    
    logger.info(f"Backtest de réplication synthétique pour l'indice {args.index}")
    
    # Créer le répertoire de résultats si nécessaire
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialiser la réplication synthétique
    synthetic = SyntheticReplication(
        index_name=args.index,
        start_date=args.start_date,
        end_date=args.end_date,
        data_dir=args.data_dir,
        initial_capital=args.capital,
        management_fee=args.management_fee,
        swap_fee=args.swap_fee,
        swap_reset_frequency=args.swap_reset_frequency,
        risk_free_rate=args.risk_free_rate
    )
    
    # Exécuter le backtest
    results = synthetic.run_backtest()
    
    # Calculer les métriques de performance
    metrics = synthetic.calculate_performance_metrics(results)
    
    # Afficher les métriques principales
    logger.info(f"Tracking Error: {metrics['tracking_error']:.4%}")
    logger.info(f"Information Ratio: {metrics['information_ratio']:.4f}")
    logger.info(f"Coûts de swap totaux: {metrics['total_swap_costs']:.2f} €")
    
    # Sauvegarder les graphiques
    plot_path = os.path.join(args.output_dir, f"{args.index}_synthetic_plot.png")
    synthetic.plot_results(results, save_path=plot_path)
    
    logger.info(f"Graphiques sauvegardés: {plot_path}")
    return 0


def compare_strategies(args):
    """
    Compare les stratégies de réplication physique et synthétique.
    """
    from src.models.comparison import ReplicationComparison
    
    logger.info(f"Comparaison des stratégies pour l'indice {args.index}")
    
    # Créer le répertoire de résultats si nécessaire
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Paramètres pour la réplication physique
    physical_params = {
        'management_fee': args.physical_mgmt_fee,
        'transaction_cost': args.physical_txn_cost,
        'rebalance_frequency': args.physical_rebalance
    }
    
    # Paramètres pour la réplication synthétique
    synthetic_params = {
        'management_fee': args.synthetic_mgmt_fee,
        'swap_fee': args.synthetic_swap_fee,
        'swap_reset_frequency': args.synthetic_reset,
        'risk_free_rate': args.risk_free_rate
    }
    
    # Initialiser la comparaison
    comparison = ReplicationComparison(
        index_name=args.index,
        start_date=args.start_date,
        end_date=args.end_date,
        data_dir=args.data_dir,
        initial_capital=args.capital,
        physical_params=physical_params,
        synthetic_params=synthetic_params
    )
    
    # Exécuter la comparaison
    results = comparison.run_comparison()
    
    # Générer le rapport
    report_dir = comparison.generate_report(results, args.output_dir)
    
    logger.info(f"Rapport de comparaison généré: {report_dir}")
    return 0


def parse_args():
    """
    Parse les arguments de la ligne de commande.
    
    Returns:
    --------
    argparse.Namespace
        Arguments parsés
    """
    parser = argparse.ArgumentParser(description='Backtest et suivi d\'un portefeuille répliquant un indice')
    subparsers = parser.add_subparsers(dest='command', help='Commande à exécuter')
    
    # Parser pour la commande 'download'
    download_parser = subparsers.add_parser('download', help='Télécharge les données des indices et de leurs composants')
    download_parser.add_argument('--index', type=str, default='CAC40',
                                help='Indice à répliquer (S&P500, CAC40, EUROSTOXX50)')
    download_parser.add_argument('--start-date', type=str, default=None,
                                help='Date de début (YYYY-MM-DD)')
    download_parser.add_argument('--end-date', type=str, default=None,
                                help='Date de fin (YYYY-MM-DD)')
    download_parser.add_argument('--years', type=int, default=5,
                                help='Nombre d\'années de données à télécharger (si start-date non spécifiée)')
    download_parser.add_argument('--no-components', action='store_true',
                                help='Ne pas télécharger les données des composants')
    download_parser.add_argument('--limit', type=int, default=0,
                                help='Limite le nombre de composants à télécharger (0 = tous)')
    
    # Parser pour la commande 'process'
    process_parser = subparsers.add_parser('process', help='Traite les données brutes pour préparer le backtest')
    process_parser.add_argument('--index', type=str, default='CAC40',
                              help='Indice à répliquer (S&P500, CAC40, EUROSTOXX50)')
    
    # Parser pour la commande 'physical'
    physical_parser = subparsers.add_parser('physical', help='Exécute le backtest de la réplication physique')
    physical_parser.add_argument('--index', type=str, default='CAC40',
                                help='Indice à répliquer (S&P500, CAC40, EUROSTOXX50)')
    physical_parser.add_argument('--start-date', type=str, default=None,
                                help='Date de début (YYYY-MM-DD)')
    physical_parser.add_argument('--end-date', type=str, default=None,
                                help='Date de fin (YYYY-MM-DD)')
    physical_parser.add_argument('--data-dir', type=str, default='data/processed',
                                help='Répertoire des données traitées')
    physical_parser.add_argument('--capital', type=float, default=1000000.0,
                                help='Capital initial')
    physical_parser.add_argument('--management-fee', type=float, default=0.0035,
                                help='Frais de gestion annuels (en pourcentage)')
    physical_parser.add_argument('--transaction-cost', type=float, default=0.0020,
                                help='Coûts de transaction (en pourcentage)')
    physical_parser.add_argument('--rebalance-frequency', type=str, default='quarterly',
                                choices=['daily', 'weekly', 'monthly', 'quarterly', 'yearly'],
                                help='Fréquence de rebalancement')
    physical_parser.add_argument('--output-dir', type=str, default='results',
                                help='Répertoire pour les résultats')
    
    # Parser pour la commande 'synthetic'
    synthetic_parser = subparsers.add_parser('synthetic', help='Exécute le backtest de la réplication synthétique')
    synthetic_parser.add_argument('--index', type=str, default='CAC40',
                                 help='Indice à répliquer (S&P500, CAC40, EUROSTOXX50)')
    synthetic_parser.add_argument('--start-date', type=str, default=None,
                                 help='Date de début (YYYY-MM-DD)')
    synthetic_parser.add_argument('--end-date', type=str, default=None,
                                 help='Date de fin (YYYY-MM-DD)')
    synthetic_parser.add_argument('--data-dir', type=str, default='data/processed',
                                 help='Répertoire des données traitées')
    synthetic_parser.add_argument('--capital', type=float, default=1000000.0,
                                 help='Capital initial')
    synthetic_parser.add_argument('--management-fee', type=float, default=0.0025,
                                 help='Frais de gestion annuels (en pourcentage)')
    synthetic_parser.add_argument('--swap-fee', type=float, default=0.0015,
                                 help='Frais du swap annuels (en pourcentage)')
    synthetic_parser.add_argument('--swap-reset-frequency', type=str, default='monthly',
                                 choices=['daily', 'weekly', 'monthly', 'quarterly', 'yearly'],
                                 help='Fréquence de reset du swap')
    synthetic_parser.add_argument('--risk-free-rate', type=float, default=0.02,
                                 help='Taux sans risque annuel')
    synthetic_parser.add_argument('--output-dir', type=str, default='results',
                                 help='Répertoire pour les résultats')
    
    # Parser pour la commande 'compare'
    compare_parser = subparsers.add_parser('compare', help='Compare les stratégies de réplication')
    compare_parser.add_argument('--index', type=str, default='CAC40',
                               help='Indice à répliquer (S&P500, CAC40, EUROSTOXX50)')
    compare_parser.add_argument('--start-date', type=str, default=None,
                               help='Date de début (YYYY-MM-DD)')
    compare_parser.add_argument('--end-date', type=str, default=None,
                               help='Date de fin (YYYY-MM-DD)')
    compare_parser.add_argument('--data-dir', type=str, default='data/processed',
                               help='Répertoire des données traitées')
    compare_parser.add_argument('--capital', type=float, default=1000000.0,
                               help='Capital initial')
    compare_parser.add_argument('--output-dir', type=str, default='results',
                               help='Répertoire pour les résultats')
    
    # Paramètres pour la réplication physique
    compare_parser.add_argument('--physical-mgmt-fee', type=float, default=0.0035,
                               help='Frais de gestion annuels pour la réplication physique')
    compare_parser.add_argument('--physical-txn-cost', type=float, default=0.0020,
                               help='Coûts de transaction pour la réplication physique')
    compare_parser.add_argument('--physical-rebalance', type=str, default='quarterly',
                               choices=['daily', 'weekly', 'monthly', 'quarterly', 'yearly'],
                               help='Fréquence de rebalancement pour la réplication physique')
    
    # Paramètres pour la réplication synthétique
    compare_parser.add_argument('--synthetic-mgmt-fee', type=float, default=0.0025,
                               help='Frais de gestion annuels pour la réplication synthétique')
    compare_parser.add_argument('--synthetic-swap-fee', type=float, default=0.0015,
                               help='Frais du swap annuels pour la réplication synthétique')
    compare_parser.add_argument('--synthetic-reset', type=str, default='monthly',
                               choices=['daily', 'weekly', 'monthly', 'quarterly', 'yearly'],
                               help='Fréquence de reset du swap pour la réplication synthétique')
    compare_parser.add_argument('--risk-free-rate', type=float, default=0.02,
                               help='Taux sans risque annuel')
    
    return parser.parse_args()


def main():
    """
    Fonction principale.
    """
    # Créer le dossier de logs s'il n'existe pas
    os.makedirs('logs', exist_ok=True)
    
    args = parse_args()
    
    if args.command == 'download':
        return download_data(args)
    elif args.command == 'process':
        return process_data(args)
    elif args.command == 'physical':
        return backtest_physical(args)
    elif args.command == 'synthetic':
        return backtest_synthetic(args)
    elif args.command == 'compare':
        return compare_strategies(args)
    else:
        logger.error(f"Commande inconnue: {args.command}")
        return 1


if __name__ == "__main__":
    exit(main())
