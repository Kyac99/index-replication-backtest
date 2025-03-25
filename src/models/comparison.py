#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour comparer les stratégies de réplication physique et synthétique d'un indice.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import logging
import argparse
from pathlib import Path

from physical_replication import PhysicalReplication
from synthetic_replication import SyntheticReplication

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'comparison.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ReplicationComparison:
    """
    Classe pour comparer les stratégies de réplication physique et synthétique d'un indice.
    """
    
    def __init__(self, index_name, start_date=None, end_date=None, 
                 data_dir='data/processed', initial_capital=1000000.0,
                 physical_params=None, synthetic_params=None):
        """
        Initialise la comparaison des stratégies de réplication.
        
        Parameters:
        -----------
        index_name : str
            Nom de l'indice à répliquer ('S&P500', 'CAC40', etc.)
        start_date : str, optional
            Date de début au format 'YYYY-MM-DD'
        end_date : str, optional
            Date de fin au format 'YYYY-MM-DD'
        data_dir : str, optional
            Répertoire des données traitées
        initial_capital : float, optional
            Capital initial du portefeuille
        physical_params : dict, optional
            Paramètres spécifiques pour la réplication physique
        synthetic_params : dict, optional
            Paramètres spécifiques pour la réplication synthétique
        """
        self.index_name = index_name
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir
        self.initial_capital = initial_capital
        
        # Paramètres par défaut pour la réplication physique
        self.physical_params = {
            'management_fee': 0.0035,
            'transaction_cost': 0.0020,
            'rebalance_frequency': 'quarterly'
        }
        # Mettre à jour avec les paramètres fournis
        if physical_params:
            self.physical_params.update(physical_params)
        
        # Paramètres par défaut pour la réplication synthétique
        self.synthetic_params = {
            'management_fee': 0.0025,
            'swap_fee': 0.0015,
            'swap_reset_frequency': 'monthly',
            'risk_free_rate': 0.02
        }
        # Mettre à jour avec les paramètres fournis
        if synthetic_params:
            self.synthetic_params.update(synthetic_params)
        
        # Initialiser les objets de réplication
        self.physical = PhysicalReplication(
            index_name=index_name,
            start_date=start_date,
            end_date=end_date,
            data_dir=data_dir,
            initial_capital=initial_capital,
            **self.physical_params
        )
        
        self.synthetic = SyntheticReplication(
            index_name=index_name,
            start_date=start_date,
            end_date=end_date,
            data_dir=data_dir,
            initial_capital=initial_capital,
            **self.synthetic_params
        )
        
        # Résultats
        self.physical_results = None
        self.synthetic_results = None
        self.physical_metrics = None
        self.synthetic_metrics = None
    
    def run_comparison(self):
        """
        Exécute les deux stratégies et les compare.
        
        Returns:
        --------
        dict
            Résultats de la comparaison
        """
        logger.info(f"Exécution de la comparaison pour l'indice {self.index_name}")
        
        # Exécuter la réplication physique
        logger.info("Exécution de la réplication physique...")
        self.physical_results = self.physical.run_backtest()
        self.physical_metrics = self.physical.calculate_performance_metrics(self.physical_results)
        
        # Exécuter la réplication synthétique
        logger.info("Exécution de la réplication synthétique...")
        self.synthetic_results = self.synthetic.run_backtest()
        self.synthetic_metrics = self.synthetic.calculate_performance_metrics(self.synthetic_results)
        
        # Combiner les résultats
        comparison_results = {
            'physical': {
                'results': self.physical_results,
                'metrics': self.physical_metrics
            },
            'synthetic': {
                'results': self.synthetic_results,
                'metrics': self.synthetic_metrics
            }
        }
        
        logger.info("Comparaison terminée")
        return comparison_results
    
    def plot_comparison(self, results=None, figsize=(16, 12), save_path=None):
        """
        Affiche les résultats de la comparaison sous forme de graphiques.
        
        Parameters:
        -----------
        results : dict, optional
            Résultats de la comparaison. Si None, utilise self.run_comparison()
        figsize : tuple, optional
            Taille des figures
        save_path : str, optional
            Chemin pour sauvegarder les graphiques. Si None, affiche les graphiques
        """
        if results is None:
            results = self.run_comparison()
        
        physical_results = results['physical']['results']
        physical_metrics = results['physical']['metrics']
        synthetic_results = results['synthetic']['results']
        synthetic_metrics = results['synthetic']['metrics']
        
        # Création de la figure avec GridSpec pour un meilleur contrôle de la mise en page
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(4, 2, figure=fig)
        
        # 1. Évolution de la valeur des portefeuilles
        ax1 = fig.add_subplot(gs[0, :])
        physical_portfolio = physical_results['portfolio_value']
        synthetic_portfolio = synthetic_results['portfolio_value']
        index_dates = physical_results['index_returns'].index
        
        # Aligner les dates
        common_dates = physical_portfolio.index.intersection(synthetic_portfolio.index)
        
        ax1.plot(common_dates, physical_portfolio.loc[common_dates], label='Réplication Physique')
        ax1.plot(common_dates, synthetic_portfolio.loc[common_dates], label='Réplication Synthétique')
        ax1.set_title('Évolution de la valeur des portefeuilles', fontsize=14)
        ax1.set_ylabel('Valeur (€)')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Rendement cumulé des deux stratégies vs indice
        ax2 = fig.add_subplot(gs[1, :])
        physical_returns = physical_results['portfolio_returns']['cumulative_return']
        synthetic_returns = synthetic_results['portfolio_returns']['cumulative_return']
        index_returns = physical_results['index_returns']['cumulative_return']  # Les deux stratégies utilisent le même indice
        
        # Aligner les dates
        common_dates = physical_returns.index.intersection(synthetic_returns.index).intersection(index_returns.index)
        
        ax2.plot(common_dates, physical_returns.loc[common_dates] * 100, label='Réplication Physique')
        ax2.plot(common_dates, synthetic_returns.loc[common_dates] * 100, label='Réplication Synthétique')
        ax2.plot(common_dates, index_returns.loc[common_dates] * 100, label='Indice', linestyle='--')
        ax2.set_title('Rendement cumulé (%)', fontsize=14)
        ax2.set_ylabel('Rendement (%)')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Tracking error quotidien
        ax3 = fig.add_subplot(gs[2, 0])
        physical_daily_returns = physical_results['portfolio_returns']['daily_return']
        synthetic_daily_returns = synthetic_results['portfolio_returns']['daily_return']
        index_daily_returns = physical_results['index_returns']['daily_return']
        
        # Aligner les dates
        common_dates = physical_daily_returns.index.intersection(synthetic_daily_returns.index).intersection(index_daily_returns.index)
        
        physical_tracking_error = (physical_daily_returns.loc[common_dates] - index_daily_returns.loc[common_dates]) * 100
        synthetic_tracking_error = (synthetic_daily_returns.loc[common_dates] - index_daily_returns.loc[common_dates]) * 100
        
        ax3.plot(common_dates, physical_tracking_error, label='Réplication Physique', alpha=0.7)
        ax3.plot(common_dates, synthetic_tracking_error, label='Réplication Synthétique', alpha=0.7)
        ax3.set_title('Tracking Error Quotidien (pp)', fontsize=14)
        ax3.set_ylabel('Écart (%)')
        ax3.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax3.legend()
        ax3.grid(True)
        
        # 4. Histogramme des tracking errors
        ax4 = fig.add_subplot(gs[2, 1])
        sns.histplot(physical_tracking_error, bins=50, alpha=0.5, label='Réplication Physique', ax=ax4)
        sns.histplot(synthetic_tracking_error, bins=50, alpha=0.5, label='Réplication Synthétique', ax=ax4)
        ax4.set_title('Distribution du Tracking Error Quotidien', fontsize=14)
        ax4.set_xlabel('Écart (%)')
        ax4.set_ylabel('Fréquence')
        ax4.legend()
        ax4.grid(True)
        
        # 5. Tableau comparatif des métriques
        ax5 = fig.add_subplot(gs[3, :])
        ax5.axis('tight')
        ax5.axis('off')
        
        metrics_data = [
            ['Métrique', 'Réplication Physique', 'Réplication Synthétique', 'Indice'],
            ['Rendement Cumulé', f"{physical_metrics['portfolio']['cumulative_return']:.2%}", f"{synthetic_metrics['portfolio']['cumulative_return']:.2%}", f"{physical_metrics['index']['cumulative_return']:.2%}"],
            ['Rendement Annualisé', f"{physical_metrics['portfolio']['annualized_return']:.2%}", f"{synthetic_metrics['portfolio']['annualized_return']:.2%}", f"{physical_metrics['index']['annualized_return']:.2%}"],
            ['Volatilité', f"{physical_metrics['portfolio']['volatility']:.2%}", f"{synthetic_metrics['portfolio']['volatility']:.2%}", f"{physical_metrics['index']['volatility']:.2%}"],
            ['Ratio de Sharpe', f"{physical_metrics['portfolio']['sharpe_ratio']:.2f}", f"{synthetic_metrics['portfolio']['sharpe_ratio']:.2f}", f"{physical_metrics['index']['sharpe_ratio']:.2f}"],
            ['Drawdown Max', f"{physical_metrics['portfolio']['max_drawdown']:.2%}", f"{synthetic_metrics['portfolio']['max_drawdown']:.2%}", f"{physical_metrics['index']['max_drawdown']:.2%}"],
            ['Tracking Error', f"{physical_metrics['tracking_error']:.2%}", f"{synthetic_metrics['tracking_error']:.2%}", "N/A"],
            ['Information Ratio', f"{physical_metrics['information_ratio']:.2f}", f"{synthetic_metrics['information_ratio']:.2f}", "N/A"],
            ['Coûts Totaux', f"{physical_metrics['total_transaction_costs']:.2f} €", f"{synthetic_metrics['total_swap_costs']:.2f} €", "N/A"],
            ['Nb Transactions / Resets', f"{physical_metrics['number_of_transactions']}", f"{synthetic_metrics['number_of_swap_resets']}", "N/A"]
        ]
        
        table = ax5.table(cellText=metrics_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Mise en forme du tableau
        for i in range(len(metrics_data)):
            for j in range(len(metrics_data[0])):
                cell = table[(i, j)]
                if i == 0:  # Entêtes
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(color='white', fontweight='bold')
                elif j == 0:  # Première colonne
                    cell.set_text_props(fontweight='bold')
                    cell.set_facecolor('#D9E1F2')
                else:
                    if i % 2 == 0:  # Lignes paires
                        cell.set_facecolor('#E9EDF4')
                    else:  # Lignes impaires
                        cell.set_facecolor('#FFFFFF')
        
        # Ajuster la mise en page
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.suptitle(f'Comparaison des Stratégies de Réplication pour {self.index_name}', fontsize=16, y=0.995)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Graphiques sauvegardés: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, results=None, output_dir='results'):
        """
        Génère un rapport complet de la comparaison.
        
        Parameters:
        -----------
        results : dict, optional
            Résultats de la comparaison. Si None, utilise self.run_comparison()
        output_dir : str, optional
            Répertoire pour les résultats
        
        Returns:
        --------
        str
            Chemin du rapport généré
        """
        if results is None:
            results = self.run_comparison()
        
        # Créer le répertoire de sortie s'il n'existe pas
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer un sous-répertoire pour cette comparaison
        report_dir = output_dir / f"{self.index_name}_comparison"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder les graphiques de comparaison
        plot_path = report_dir / f"{self.index_name}_comparison_plot.png"
        self.plot_comparison(results, save_path=plot_path)
        
        # Sauvegarder les métriques comparatives dans un fichier CSV
        physical_metrics = results['physical']['metrics']
        synthetic_metrics = results['synthetic']['metrics']
        
        comparison_df = pd.DataFrame({
            'Métrique': [
                'Rendement Cumulé (Physique)',
                'Rendement Cumulé (Synthétique)',
                'Rendement Cumulé (Indice)',
                'Rendement Annualisé (Physique)',
                'Rendement Annualisé (Synthétique)',
                'Rendement Annualisé (Indice)',
                'Volatilité (Physique)',
                'Volatilité (Synthétique)',
                'Volatilité (Indice)',
                'Ratio de Sharpe (Physique)',
                'Ratio de Sharpe (Synthétique)',
                'Ratio de Sharpe (Indice)',
                'Drawdown Max (Physique)',
                'Drawdown Max (Synthétique)',
                'Drawdown Max (Indice)',
                'Tracking Error (Physique)',
                'Tracking Error (Synthétique)',
                'Information Ratio (Physique)',
                'Information Ratio (Synthétique)',
                'Coûts Totaux (Physique)',
                'Coûts Totaux (Synthétique)',
                'Nombre de Transactions (Physique)',
                'Nombre de Resets (Synthétique)'
            ],
            'Valeur': [
                f"{physical_metrics['portfolio']['cumulative_return']:.4%}",
                f"{synthetic_metrics['portfolio']['cumulative_return']:.4%}",
                f"{physical_metrics['index']['cumulative_return']:.4%}",
                f"{physical_metrics['portfolio']['annualized_return']:.4%}",
                f"{synthetic_metrics['portfolio']['annualized_return']:.4%}",
                f"{physical_metrics['index']['annualized_return']:.4%}",
                f"{physical_metrics['portfolio']['volatility']:.4%}",
                f"{synthetic_metrics['portfolio']['volatility']:.4%}",
                f"{physical_metrics['index']['volatility']:.4%}",
                f"{physical_metrics['portfolio']['sharpe_ratio']:.4f}",
                f"{synthetic_metrics['portfolio']['sharpe_ratio']:.4f}",
                f"{physical_metrics['index']['sharpe_ratio']:.4f}",
                f"{physical_metrics['portfolio']['max_drawdown']:.4%}",
                f"{synthetic_metrics['portfolio']['max_drawdown']:.4%}",
                f"{physical_metrics['index']['max_drawdown']:.4%}",
                f"{physical_metrics['tracking_error']:.4%}",
                f"{synthetic_metrics['tracking_error']:.4%}",
                f"{physical_metrics['information_ratio']:.4f}",
                f"{synthetic_metrics['information_ratio']:.4f}",
                f"{physical_metrics['total_transaction_costs']:.2f} €",
                f"{synthetic_metrics['total_swap_costs']:.2f} €",
                f"{physical_metrics['number_of_transactions']}",
                f"{synthetic_metrics['number_of_swap_resets']}"
            ]
        })
        
        # Sauvegarder les métriques
        metrics_path = report_dir / f"{self.index_name}_comparison_metrics.csv"
        comparison_df.to_csv(metrics_path, index=False)
        
        # Sauvegarder les valeurs des portefeuilles en CSV
        physical_portfolio = results['physical']['results']['portfolio_value']
        synthetic_portfolio = results['synthetic']['results']['portfolio_value']
        index_returns = results['physical']['results']['index_returns']['cumulative_return']
        
        # Aligner les dates
        common_dates = physical_portfolio.index.intersection(synthetic_portfolio.index).intersection(index_returns.index)
        
        portfolios_df = pd.DataFrame({
            'Date': common_dates,
            'Réplication Physique': physical_portfolio.loc[common_dates].values,
            'Réplication Synthétique': synthetic_portfolio.loc[common_dates].values,
            'Indice (Rendement Cumulé)': (1 + index_returns.loc[common_dates]).values * self.initial_capital
        })
        
        portfolios_path = report_dir / f"{self.index_name}_comparison_portfolios.csv"
        portfolios_df.to_csv(portfolios_path, index=False)
        
        # Sauvegarder les paramètres utilisés
        params_df = pd.DataFrame({
            'Paramètre': [
                'Indice', 
                'Date de début', 
                'Date de fin',
                'Capital initial',
                'Frais de gestion (Physique)',
                'Frais de transaction (Physique)',
                'Fréquence de rebalancement (Physique)',
                'Frais de gestion (Synthétique)',
                'Frais de swap (Synthétique)',
                'Fréquence de reset (Synthétique)',
                'Taux sans risque (Synthétique)'
            ],
            'Valeur': [
                self.index_name,
                self.start_date if self.start_date else 'Auto',
                self.end_date if self.end_date else 'Auto',
                f"{self.initial_capital:.2f} €",
                f"{self.physical_params['management_fee']:.4%}",
                f"{self.physical_params['transaction_cost']:.4%}",
                self.physical_params['rebalance_frequency'],
                f"{self.synthetic_params['management_fee']:.4%}",
                f"{self.synthetic_params['swap_fee']:.4%}",
                self.synthetic_params['swap_reset_frequency'],
                f"{self.synthetic_params['risk_free_rate']:.2%}"
            ]
        })
        
        params_path = report_dir / f"{self.index_name}_comparison_params.csv"
        params_df.to_csv(params_path, index=False)
        
        logger.info(f"Rapport généré dans {report_dir}")
        return str(report_dir)


def parse_args():
    """
    Parse les arguments de la ligne de commande.
    
    Returns:
    --------
    argparse.Namespace
        Arguments parsés
    """
    parser = argparse.ArgumentParser(description='Comparaison des stratégies de réplication d\'un indice')
    
    parser.add_argument('--index', type=str, default='CAC40',
                        help='Indice à répliquer (S&P500, CAC40, EUROSTOXX50)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Date de début (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='Date de fin (YYYY-MM-DD)')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Répertoire des données traitées')
    parser.add_argument('--capital', type=float, default=1000000.0,
                        help='Capital initial')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Répertoire pour les résultats')
    
    # Paramètres pour la réplication physique
    parser.add_argument('--physical-mgmt-fee', type=float, default=0.0035,
                        help='Frais de gestion annuels pour la réplication physique')
    parser.add_argument('--physical-txn-cost', type=float, default=0.0020,
                        help='Coûts de transaction pour la réplication physique')
    parser.add_argument('--physical-rebalance', type=str, default='quarterly',
                        choices=['daily', 'weekly', 'monthly', 'quarterly', 'yearly'],
                        help='Fréquence de rebalancement pour la réplication physique')
    
    # Paramètres pour la réplication synthétique
    parser.add_argument('--synthetic-mgmt-fee', type=float, default=0.0025,
                        help='Frais de gestion annuels pour la réplication synthétique')
    parser.add_argument('--synthetic-swap-fee', type=float, default=0.0015,
                        help='Frais du swap annuels pour la réplication synthétique')
    parser.add_argument('--synthetic-reset', type=str, default='monthly',
                        choices=['daily', 'weekly', 'monthly', 'quarterly', 'yearly'],
                        help='Fréquence de reset du swap pour la réplication synthétique')
    parser.add_argument('--risk-free-rate', type=float, default=0.02,
                        help='Taux sans risque annuel')
    
    return parser.parse_args()


def main():
    """
    Fonction principale pour exécuter la comparaison depuis la ligne de commande.
    """
    args = parse_args()
    
    # Extraire les paramètres physiques
    physical_params = {
        'management_fee': args.physical_mgmt_fee,
        'transaction_cost': args.physical_txn_cost,
        'rebalance_frequency': args.physical_rebalance
    }
    
    # Extraire les paramètres synthétiques
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
    
    # Exécuter la comparaison et générer le rapport
    results = comparison.run_comparison()
    report_dir = comparison.generate_report(results, args.output_dir)
    
    logger.info(f"Rapport complet disponible dans {report_dir}")


if __name__ == "__main__":
    main()
