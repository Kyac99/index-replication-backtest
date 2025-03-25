#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour implémenter et tester la stratégie de réplication synthétique d'un indice.
La réplication synthétique utilise des swaps pour reproduire le rendement de l'indice.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import argparse
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'synthetic_replication.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class SyntheticReplication:
    """
    Classe pour implémenter la stratégie de réplication synthétique d'un indice à l'aide de swaps.
    """
    
    def __init__(self, index_name, start_date=None, end_date=None, 
                 data_dir='data/processed', initial_capital=1000000.0,
                 management_fee=0.0025, swap_fee=0.0015, 
                 swap_reset_frequency='monthly',
                 risk_free_rate=0.02):
        """
        Initialise la stratégie de réplication synthétique.
        
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
        management_fee : float, optional
            Frais de gestion annuels (en pourcentage)
        swap_fee : float, optional
            Frais du swap annuels (en pourcentage)
        swap_reset_frequency : str, optional
            Fréquence de reset du swap ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')
        risk_free_rate : float, optional
            Taux sans risque annuel pour les placements monétaires
        """
        self.index_name = index_name
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.data_dir = data_dir
        self.initial_capital = initial_capital
        self.management_fee = management_fee
        self.swap_fee = swap_fee
        self.swap_reset_frequency = swap_reset_frequency
        self.risk_free_rate = risk_free_rate
        
        # Initialisation des données
        self.index_returns = None
        
        # Initialisation des résultats
        self.portfolio_value = None
        self.portfolio_returns = None
        self.cash_position = None
        self.swap_position = None
        self.swap_pnl = None
        self.swap_costs = []
        self.tracking_error = None
        
        # Chargement des données
        self._load_data()
        
        # Filtrer les données selon les dates
        self._filter_dates()
    
    def _load_data(self):
        """
        Charge les données nécessaires pour la réplication.
        """
        try:
            # Charger les rendements de l'indice
            index_path = os.path.join(self.data_dir, 'indices', f'{self.index_name}_index_returns.parquet')
            if os.path.exists(index_path):
                index_data = pd.read_parquet(index_path)
                self.index_returns = index_data[['daily_return', 'cumulative_return']]
                logger.info(f"Rendements de l'indice chargés: {self.index_returns.shape}")
            else:
                logger.error(f"Fichier non trouvé: {index_path}")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
    
    def _filter_dates(self):
        """
        Filtre les données selon les dates de début et de fin spécifiées.
        """
        if self.start_date is None and self.end_date is None:
            return
        
        # Filtrer les rendements de l'indice
        if self.index_returns is not None:
            mask = True
            if self.start_date:
                mask = mask & (self.index_returns.index >= self.start_date)
            if self.end_date:
                mask = mask & (self.index_returns.index <= self.end_date)
            self.index_returns = self.index_returns[mask]
        
        logger.info(f"Données filtrées de {self.index_returns.index.min()} à {self.index_returns.index.max()}")
    
    def _get_swap_reset_dates(self):
        """
        Détermine les dates de reset du swap en fonction de la fréquence spécifiée.
        
        Returns:
        --------
        list
            Liste des dates de reset du swap
        """
        if self.index_returns is None:
            logger.error("Données de l'indice non chargées")
            return []
        
        dates = self.index_returns.index
        first_date = dates.min()
        last_date = dates.max()
        
        reset_dates = [first_date]  # Toujours inclure la première date
        
        if self.swap_reset_frequency == 'daily':
            reset_dates = dates
        
        elif self.swap_reset_frequency == 'weekly':
            # Reset chaque lundi
            current_date = first_date
            while current_date <= last_date:
                current_date += pd.Timedelta(days=7)
                # Trouver la première date disponible après current_date
                future_dates = dates[dates > current_date]
                if not future_dates.empty:
                    reset_dates.append(future_dates.min())
        
        elif self.swap_reset_frequency == 'monthly':
            # Reset au premier jour de chaque mois
            current_month = first_date.month
            current_year = first_date.year
            while True:
                current_month += 1
                if current_month > 12:
                    current_month = 1
                    current_year += 1
                
                next_month_date = pd.Timestamp(year=current_year, month=current_month, day=1)
                if next_month_date > last_date:
                    break
                
                # Trouver la première date disponible après next_month_date
                future_dates = dates[dates >= next_month_date]
                if not future_dates.empty:
                    reset_dates.append(future_dates.min())
        
        elif self.swap_reset_frequency == 'quarterly':
            # Reset au premier jour de chaque trimestre
            current_month = first_date.month
            current_year = first_date.year
            # Trouver le début du prochain trimestre
            next_quarter_month = ((current_month - 1) // 3 + 1) * 3 + 1
            if next_quarter_month > 12:
                next_quarter_month -= 12
                current_year += 1
            
            while True:
                next_quarter_date = pd.Timestamp(year=current_year, month=next_quarter_month, day=1)
                if next_quarter_date > last_date:
                    break
                
                # Trouver la première date disponible après next_quarter_date
                future_dates = dates[dates >= next_quarter_date]
                if not future_dates.empty:
                    reset_dates.append(future_dates.min())
                
                # Passer au trimestre suivant
                next_quarter_month += 3
                if next_quarter_month > 12:
                    next_quarter_month -= 12
                    current_year += 1
        
        elif self.swap_reset_frequency == 'yearly':
            # Reset au premier jour de chaque année
            current_year = first_date.year
            while True:
                current_year += 1
                next_year_date = pd.Timestamp(year=current_year, month=1, day=1)
                if next_year_date > last_date:
                    break
                
                # Trouver la première date disponible après next_year_date
                future_dates = dates[dates >= next_year_date]
                if not future_dates.empty:
                    reset_dates.append(future_dates.min())
        
        logger.info(f"Dates de reset du swap générées: {len(reset_dates)} dates")
        return reset_dates
    
    def run_backtest(self):
        """
        Exécute le backtest de la stratégie de réplication synthétique.
        
        Returns:
        --------
        dict
            Résultats du backtest
        """
        if self.index_returns is None:
            logger.error("Données de l'indice manquantes pour le backtest")
            return {}
        
        # Obtenir les dates de reset du swap
        reset_dates = self._get_swap_reset_dates()
        
        # Calculer les taux quotidiens
        daily_mgmt_fee = (1 + self.management_fee) ** (1/252) - 1
        daily_swap_fee = (1 + self.swap_fee) ** (1/252) - 1
        daily_rf_rate = (1 + self.risk_free_rate) ** (1/252) - 1
        
        # Initialiser le portefeuille
        dates = self.index_returns.index
        self.portfolio_value = pd.Series(index=dates, dtype=float)
        self.portfolio_value.iloc[0] = self.initial_capital
        
        # Initialiser les positions monétaires et du swap
        self.cash_position = pd.Series(index=dates, dtype=float)
        self.cash_position.iloc[0] = self.initial_capital
        
        self.swap_position = pd.Series(index=dates, dtype=float)
        self.swap_position.iloc[0] = self.initial_capital
        
        self.swap_pnl = pd.Series(index=dates, dtype=float)
        self.swap_pnl.iloc[0] = 0
        
        # Initialiser le swap
        current_swap_notional = self.initial_capital
        current_swap_reference = 100.0  # Valeur de référence de l'indice
        
        # Simuler le backtest jour par jour
        for i, date in enumerate(dates):
            if i == 0:
                # Premier jour: initialisation du portefeuille
                continue
            
            # Obtenir la date précédente
            prev_date = dates[i-1]
            
            # Récupérer le rendement de l'indice pour ce jour
            index_return = self.index_returns.loc[date, 'daily_return']
            
            # Calculer la nouvelle valeur du swap (avant les frais)
            swap_value_before_fee = self.swap_position.loc[prev_date] * (1 + index_return)
            
            # Appliquer les frais du swap
            swap_value_after_fee = swap_value_before_fee * (1 - daily_swap_fee)
            self.swap_position.loc[date] = swap_value_after_fee
            
            # Calculer le P&L du swap pour ce jour
            swap_pnl_today = swap_value_after_fee - self.swap_position.loc[prev_date]
            self.swap_pnl.loc[date] = swap_pnl_today
            
            # Faire croître la position en cash avec le taux sans risque
            cash_value_before_fee = self.cash_position.loc[prev_date] * (1 + daily_rf_rate)
            
            # Appliquer les frais de gestion
            cash_value_after_fee = cash_value_before_fee * (1 - daily_mgmt_fee)
            self.cash_position.loc[date] = cash_value_after_fee
            
            # Calculer la valeur totale du portefeuille
            self.portfolio_value.loc[date] = self.cash_position.loc[date]
            
            # Reset du swap si nécessaire
            if date in reset_dates and i > 0:
                # Enregistrer le coût du reset du swap
                swap_reset_cost = self.swap_position.loc[date] * 0.0001  # Coût de 1 bp pour le reset
                
                # Enregistrer les informations du reset
                self.swap_costs.append({
                    'date': date,
                    'swap_value': self.swap_position.loc[date],
                    'cost': swap_reset_cost
                })
                
                # Appliquer le coût du reset
                self.cash_position.loc[date] -= swap_reset_cost
                self.portfolio_value.loc[date] -= swap_reset_cost
                
                # Mettre à jour les références du swap
                current_swap_notional = self.swap_position.loc[date]
                current_swap_reference = 100.0 * (1 + self.index_returns.loc[date, 'cumulative_return'])
        
        # Calculer les rendements du portefeuille
        self.portfolio_returns = pd.DataFrame(index=dates)
        self.portfolio_returns['daily_return'] = self.portfolio_value.pct_change()
        self.portfolio_returns['cumulative_return'] = (1 + self.portfolio_returns['daily_return']).cumprod() - 1
        
        # Supprimer la première ligne avec des NaN
        self.portfolio_returns = self.portfolio_returns.dropna()
        
        # Calculer le tracking error
        self._calculate_tracking_error()
        
        # Préparer les résultats du backtest
        results = {
            'portfolio_value': self.portfolio_value,
            'portfolio_returns': self.portfolio_returns,
            'index_returns': self.index_returns,
            'tracking_error': self.tracking_error,
            'cash_position': self.cash_position,
            'swap_position': self.swap_position,
            'swap_pnl': self.swap_pnl,
            'swap_costs': pd.DataFrame(self.swap_costs)
        }
        
        logger.info(f"Backtest terminé: rendement cumulé du portefeuille de {self.portfolio_returns['cumulative_return'].iloc[-1]:.2%}")
        return results
    
    def _calculate_tracking_error(self):
        """
        Calcule le tracking error entre le portefeuille et l'indice.
        """
        if self.portfolio_returns is None or self.index_returns is None:
            logger.error("Données manquantes pour le calcul du tracking error")
            return
        
        # Aligner les dates
        common_dates = self.portfolio_returns.index.intersection(self.index_returns.index)
        portfolio_daily_returns = self.portfolio_returns.loc[common_dates, 'daily_return']
        index_daily_returns = self.index_returns.loc[common_dates, 'daily_return']
        
        # Calculer la différence de rendement
        return_difference = portfolio_daily_returns - index_daily_returns
        
        # Calculer le tracking error (écart-type annualisé de la différence de rendement)
        tracking_error = return_difference.std() * np.sqrt(252)  # Annualisation
        
        self.tracking_error = tracking_error
        logger.info(f"Tracking error: {tracking_error:.4%}")
    
    def calculate_performance_metrics(self, results=None):
        """
        Calcule les métriques de performance du portefeuille et de l'indice.
        
        Parameters:
        -----------
        results : dict, optional
            Résultats du backtest. Si None, utilise self.run_backtest()
            
        Returns:
        --------
        dict
            Métriques de performance
        """
        if results is None:
            results = self.run_backtest()
        
        if not results:
            logger.error("Aucun résultat disponible pour calculer les métriques de performance")
            return {}
        
        portfolio_returns = results['portfolio_returns']
        index_returns = results['index_returns']
        
        # Aligner les dates
        common_dates = portfolio_returns.index.intersection(index_returns.index)
        portfolio_daily_returns = portfolio_returns.loc[common_dates, 'daily_return']
        index_daily_returns = index_returns.loc[common_dates, 'daily_return']
        
        # Rendement cumulé
        portfolio_cum_return = portfolio_returns['cumulative_return'].iloc[-1]
        index_cum_return = index_returns['cumulative_return'].iloc[-1]
        
        # Rendement annualisé
        days = (common_dates.max() - common_dates.min()).days
        years = days / 365.25
        portfolio_annual_return = (1 + portfolio_cum_return) ** (1 / years) - 1
        index_annual_return = (1 + index_cum_return) ** (1 / years) - 1
        
        # Volatilité annualisée
        portfolio_volatility = portfolio_daily_returns.std() * np.sqrt(252)
        index_volatility = index_daily_returns.std() * np.sqrt(252)
        
        # Ratio de Sharpe
        portfolio_sharpe = (portfolio_annual_return - self.risk_free_rate) / portfolio_volatility
        index_sharpe = (index_annual_return - self.risk_free_rate) / index_volatility
        
        # Maximum drawdown
        portfolio_cum_returns = (1 + portfolio_daily_returns).cumprod()
        index_cum_returns = (1 + index_daily_returns).cumprod()
        
        portfolio_drawdowns = 1 - portfolio_cum_returns / portfolio_cum_returns.cummax()
        index_drawdowns = 1 - index_cum_returns / index_cum_returns.cummax()
        
        portfolio_max_drawdown = portfolio_drawdowns.max()
        index_max_drawdown = index_drawdowns.max()
        
        # Information ratio
        return_difference = portfolio_daily_returns - index_daily_returns
        tracking_error = return_difference.std() * np.sqrt(252)
        information_ratio = (portfolio_annual_return - index_annual_return) / tracking_error if tracking_error > 0 else np.nan
        
        # Résultats
        metrics = {
            'portfolio': {
                'cumulative_return': portfolio_cum_return,
                'annualized_return': portfolio_annual_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': portfolio_sharpe,
                'max_drawdown': portfolio_max_drawdown
            },
            'index': {
                'cumulative_return': index_cum_return,
                'annualized_return': index_annual_return,
                'volatility': index_volatility,
                'sharpe_ratio': index_sharpe,
                'max_drawdown': index_max_drawdown
            },
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'total_swap_costs': sum(cost['cost'] for cost in self.swap_costs) if 'swap_costs' in results else 0,
            'number_of_swap_resets': len(self.swap_costs) if 'swap_costs' in results else 0
        }
        
        return metrics
    
    def plot_results(self, results=None, figsize=(12, 10), save_path=None):
        """
        Affiche les résultats du backtest sous forme de graphiques.
        
        Parameters:
        -----------
        results : dict, optional
            Résultats du backtest. Si None, utilise self.run_backtest()
        figsize : tuple, optional
            Taille des figures
        save_path : str, optional
            Chemin pour sauvegarder les graphiques. Si None, affiche les graphiques
        """
        if results is None:
            results = self.run_backtest()
        
        if not results:
            logger.error("Aucun résultat disponible pour l'affichage")
            return
        
        portfolio_value = results['portfolio_value']
        portfolio_returns = results['portfolio_returns']
        index_returns = results['index_returns']
        swap_position = results['swap_position']
        cash_position = results['cash_position']
        
        # Calculer les métriques de performance
        metrics = self.calculate_performance_metrics(results)
        
        # Création de 3 sous-graphiques
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # 1. Évolution de la valeur du portefeuille et de ses composants
        axes[0].plot(portfolio_value.index, portfolio_value.values, label='Portefeuille Total')
        axes[0].plot(swap_position.index, swap_position.values, label='Position Swap', linestyle='--')
        axes[0].plot(cash_position.index, cash_position.values, label='Position Cash', linestyle=':')
        axes[0].set_title('Évolution de la valeur du portefeuille et de ses composants')
        axes[0].set_ylabel('Valeur (€)')
        axes[0].legend()
        axes[0].grid(True)
        
        # 2. Rendement cumulé (portefeuille vs indice)
        common_dates = portfolio_returns.index.intersection(index_returns.index)
        axes[1].plot(common_dates, portfolio_returns.loc[common_dates, 'cumulative_return'] * 100, label='Portefeuille')
        axes[1].plot(common_dates, index_returns.loc[common_dates, 'cumulative_return'] * 100, label='Indice')
        axes[1].set_title('Rendement cumulé (%)')
        axes[1].set_ylabel('Rendement (%)')
        axes[1].legend()
        axes[1].grid(True)
        
        # 3. Tracking error (différence de rendement quotidien)
        return_difference = portfolio_returns.loc[common_dates, 'daily_return'] - index_returns.loc[common_dates, 'daily_return']
        axes[2].plot(common_dates, return_difference * 100)
        axes[2].set_title('Différence de rendement quotidien (portefeuille - indice)')
        axes[2].set_ylabel('Différence (%)')
        axes[2].axhline(y=0, color='r', linestyle='-', alpha=0.3)
        axes[2].grid(True)
        
        # Ajouter les dates de reset du swap si disponibles
        if len(self.swap_costs) > 0:
            reset_dates = [cost['date'] for cost in self.swap_costs]
            for ax in axes:
                for date in reset_dates:
                    ax.axvline(x=date, color='g', linestyle='--', alpha=0.3)
        
        # Ajout de métriques de performance
        textstr = '\n'.join([
            f"Rendement cumulé: {metrics['portfolio']['cumulative_return']:.2%} (Ptf) vs {metrics['index']['cumulative_return']:.2%} (Idx)",
            f"Rendement annualisé: {metrics['portfolio']['annualized_return']:.2%} (Ptf) vs {metrics['index']['annualized_return']:.2%} (Idx)",
            f"Volatilité: {metrics['portfolio']['volatility']:.2%} (Ptf) vs {metrics['index']['volatility']:.2%} (Idx)",
            f"Tracking Error: {metrics['tracking_error']:.2%}",
            f"Information Ratio: {metrics['information_ratio']:.2f}",
            f"Coûts de swap totaux: {metrics['total_swap_costs']:.2f} €",
            f"Nombre de resets du swap: {metrics['number_of_swap_resets']}"
        ])
        
        fig.text(0.15, 0.01, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Graphiques sauvegardés: {save_path}")
        else:
            plt.show()
        
        plt.close()


def parse_args():
    """
    Parse les arguments de la ligne de commande.
    
    Returns:
    --------
    argparse.Namespace
        Arguments parsés
    """
    parser = argparse.ArgumentParser(description='Backtest de réplication synthétique d\'un indice')
    
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
    parser.add_argument('--management-fee', type=float, default=0.0025,
                        help='Frais de gestion annuels (en pourcentage)')
    parser.add_argument('--swap-fee', type=float, default=0.0015,
                        help='Frais du swap annuels (en pourcentage)')
    parser.add_argument('--swap-reset-frequency', type=str, default='monthly',
                        choices=['daily', 'weekly', 'monthly', 'quarterly', 'yearly'],
                        help='Fréquence de reset du swap')
    parser.add_argument('--risk-free-rate', type=float, default=0.02,
                        help='Taux sans risque annuel')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Répertoire pour les résultats')
    
    return parser.parse_args()


def main():
    """
    Fonction principale pour exécuter le backtest depuis la ligne de commande.
    """
    args = parse_args()
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialiser la stratégie de réplication synthétique
    replication = SyntheticReplication(
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
    results = replication.run_backtest()
    
    # Calculer les métriques de performance
    metrics = replication.calculate_performance_metrics(results)
    
    # Enregistrer les métriques dans un fichier CSV
    metrics_df = pd.DataFrame({
        'Metric': [
            'Rendement cumulé (Portefeuille)',
            'Rendement cumulé (Indice)',
            'Rendement annualisé (Portefeuille)',
            'Rendement annualisé (Indice)',
            'Volatilité (Portefeuille)',
            'Volatilité (Indice)',
            'Ratio de Sharpe (Portefeuille)',
            'Ratio de Sharpe (Indice)',
            'Drawdown maximal (Portefeuille)',
            'Drawdown maximal (Indice)',
            'Tracking Error',
            'Information Ratio',
            'Coûts de swap totaux',
            'Nombre de resets du swap'
        ],
        'Value': [
            f"{metrics['portfolio']['cumulative_return']:.4%}",
            f"{metrics['index']['cumulative_return']:.4%}",
            f"{metrics['portfolio']['annualized_return']:.4%}",
            f"{metrics['index']['annualized_return']:.4%}",
            f"{metrics['portfolio']['volatility']:.4%}",
            f"{metrics['index']['volatility']:.4%}",
            f"{metrics['portfolio']['sharpe_ratio']:.4f}",
            f"{metrics['index']['sharpe_ratio']:.4f}",
            f"{metrics['portfolio']['max_drawdown']:.4%}",
            f"{metrics['index']['max_drawdown']:.4%}",
            f"{metrics['tracking_error']:.4%}",
            f"{metrics['information_ratio']:.4f}",
            f"{metrics['total_swap_costs']:.2f} €",
            f"{metrics['number_of_swap_resets']}"
        ]
    })
    
    # Sauvegarder les métriques
    metrics_path = output_dir / f"{args.index}_synthetic_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    
    # Sauvegarder les résultats du portefeuille
    portfolio_path = output_dir / f"{args.index}_synthetic_portfolio.parquet"
    results['portfolio_value'].to_frame('portfolio_value').to_parquet(portfolio_path)
    
    # Sauvegarder les coûts du swap
    if 'swap_costs' in results and not results['swap_costs'].empty:
        swap_costs_path = output_dir / f"{args.index}_synthetic_swap_costs.csv"
        results['swap_costs'].to_csv(swap_costs_path, index=False)
    
    # Tracer et sauvegarder les graphiques
    plot_path = output_dir / f"{args.index}_synthetic_plot.png"
    replication.plot_results(results, save_path=plot_path)
    
    logger.info(f"Résultats sauvegardés dans {output_dir}")
    logger.info(f"Tracking Error: {metrics['tracking_error']:.4%}")


if __name__ == "__main__":
    main()
