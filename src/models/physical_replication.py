#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour implémenter et tester la stratégie de réplication physique d'un indice.
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
        logging.FileHandler(os.path.join('logs', 'physical_replication.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class PhysicalReplication:
    """
    Classe pour implémenter la stratégie de réplication physique d'un indice.
    """
    
    def __init__(self, index_name, start_date=None, end_date=None, 
                 data_dir='data/processed', initial_capital=1000000.0,
                 management_fee=0.0035, transaction_cost=0.0020,
                 rebalance_frequency='quarterly'):
        """
        Initialise la stratégie de réplication physique.
        
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
        transaction_cost : float, optional
            Coûts de transaction (en pourcentage)
        rebalance_frequency : str, optional
            Fréquence de rebalancement ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')
        """
        self.index_name = index_name
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.data_dir = data_dir
        self.initial_capital = initial_capital
        self.management_fee = management_fee
        self.transaction_cost = transaction_cost
        self.rebalance_frequency = rebalance_frequency
        
        # Initialisation des données
        self.price_matrix = None
        self.return_matrix = None
        self.weights_df = None
        self.index_returns = None
        
        # Initialisation des résultats
        self.portfolio_value = None
        self.portfolio_returns = None
        self.positions = None
        self.transactions = []
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
            # Charger la matrice de prix
            price_path = os.path.join(self.data_dir, 'components', f'{self.index_name}_price_matrix.parquet')
            if os.path.exists(price_path):
                self.price_matrix = pd.read_parquet(price_path)
                logger.info(f"Matrice de prix chargée: {self.price_matrix.shape}")
            else:
                logger.error(f"Fichier non trouvé: {price_path}")
            
            # Charger la matrice de rendements
            return_path = os.path.join(self.data_dir, 'components', f'{self.index_name}_return_matrix.parquet')
            if os.path.exists(return_path):
                self.return_matrix = pd.read_parquet(return_path)
                logger.info(f"Matrice de rendements chargée: {self.return_matrix.shape}")
            else:
                logger.error(f"Fichier non trouvé: {return_path}")
            
            # Charger les pondérations
            weights_path = os.path.join(self.data_dir, 'weights', f'{self.index_name}_weights.parquet')
            if os.path.exists(weights_path):
                self.weights_df = pd.read_parquet(weights_path)
                logger.info(f"Données de pondération chargées: {self.weights_df.shape}")
            else:
                logger.error(f"Fichier non trouvé: {weights_path}")
            
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
        
        # Filtrer la matrice de prix
        if self.price_matrix is not None:
            mask = True
            if self.start_date:
                mask = mask & (self.price_matrix.index >= self.start_date)
            if self.end_date:
                mask = mask & (self.price_matrix.index <= self.end_date)
            self.price_matrix = self.price_matrix[mask]
        
        # Filtrer la matrice de rendements
        if self.return_matrix is not None:
            mask = True
            if self.start_date:
                mask = mask & (self.return_matrix.index >= self.start_date)
            if self.end_date:
                mask = mask & (self.return_matrix.index <= self.end_date)
            self.return_matrix = self.return_matrix[mask]
        
        # Filtrer les données de pondération
        if self.weights_df is not None:
            mask = True
            if self.start_date:
                mask = mask & (self.weights_df.index >= self.start_date)
            if self.end_date:
                mask = mask & (self.weights_df.index <= self.end_date)
            self.weights_df = self.weights_df[mask]
        
        # Filtrer les rendements de l'indice
        if self.index_returns is not None:
            mask = True
            if self.start_date:
                mask = mask & (self.index_returns.index >= self.start_date)
            if self.end_date:
                mask = mask & (self.index_returns.index <= self.end_date)
            self.index_returns = self.index_returns[mask]
        
        logger.info(f"Données filtrées de {self.price_matrix.index.min()} à {self.price_matrix.index.max()}")
    
    def _get_rebalance_dates(self):
        """
        Détermine les dates de rebalancement en fonction de la fréquence spécifiée.
        
        Returns:
        --------
        list
            Liste des dates de rebalancement
        """
        if self.price_matrix is None:
            logger.error("Matrice de prix non chargée")
            return []
        
        dates = self.price_matrix.index
        first_date = dates.min()
        last_date = dates.max()
        
        rebalance_dates = [first_date]  # Toujours inclure la première date
        
        if self.rebalance_frequency == 'daily':
            rebalance_dates = dates
        
        elif self.rebalance_frequency == 'weekly':
            # Rebalancement chaque lundi
            current_date = first_date
            while current_date <= last_date:
                current_date += pd.Timedelta(days=7)
                # Trouver la première date disponible après current_date
                future_dates = dates[dates > current_date]
                if not future_dates.empty:
                    rebalance_dates.append(future_dates.min())
        
        elif self.rebalance_frequency == 'monthly':
            # Rebalancement au premier jour de chaque mois
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
                    rebalance_dates.append(future_dates.min())
        
        elif self.rebalance_frequency == 'quarterly':
            # Rebalancement au premier jour de chaque trimestre
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
                    rebalance_dates.append(future_dates.min())
                
                # Passer au trimestre suivant
                next_quarter_month += 3
                if next_quarter_month > 12:
                    next_quarter_month -= 12
                    current_year += 1
        
        elif self.rebalance_frequency == 'yearly':
            # Rebalancement au premier jour de chaque année
            current_year = first_date.year
            while True:
                current_year += 1
                next_year_date = pd.Timestamp(year=current_year, month=1, day=1)
                if next_year_date > last_date:
                    break
                
                # Trouver la première date disponible après next_year_date
                future_dates = dates[dates >= next_year_date]
                if not future_dates.empty:
                    rebalance_dates.append(future_dates.min())
        
        logger.info(f"Dates de rebalancement générées: {len(rebalance_dates)} dates")
        return rebalance_dates
    
    def run_backtest(self):
        """
        Exécute le backtest de la stratégie de réplication physique.
        
        Returns:
        --------
        dict
            Résultats du backtest
        """
        if (self.price_matrix is None or self.weights_df is None or 
            self.return_matrix is None or self.index_returns is None):
            logger.error("Données manquantes pour le backtest")
            return {}
        
        # Obtenir les dates de rebalancement
        rebalance_dates = self._get_rebalance_dates()
        
        # Initialiser le portefeuille
        dates = self.price_matrix.index
        self.portfolio_value = pd.Series(index=dates, dtype=float)
        self.portfolio_value.iloc[0] = self.initial_capital
        
        # Initialiser les positions (nombre d'actions détenues)
        self.positions = pd.DataFrame(0, index=dates, columns=self.price_matrix.columns)
        
        # Calculer les frais de gestion quotidiens
        daily_fee = (1 + self.management_fee) ** (1/252) - 1
        
        # Initialiser la liste des transactions
        self.transactions = []
        
        # Simuler le rebalancement et suivre la performance
        for i, date in enumerate(dates):
            if i == 0:
                # Premier jour: initialisation du portefeuille
                self._rebalance_portfolio(date)
            else:
                # Mise à jour de la valeur du portefeuille avec les prix du jour
                prev_date = dates[i-1]
                self.positions.loc[date] = self.positions.loc[prev_date]  # Copier les positions du jour précédent
                
                # Calculer la nouvelle valeur du portefeuille
                portfolio_value_before_fees = np.sum(self.positions.loc[date] * self.price_matrix.loc[date])
                
                # Appliquer les frais de gestion quotidiens
                portfolio_value_after_fees = portfolio_value_before_fees * (1 - daily_fee)
                self.portfolio_value.loc[date] = portfolio_value_after_fees
                
                # Rebalancement si nécessaire
                if date in rebalance_dates:
                    self._rebalance_portfolio(date)
        
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
            'positions': self.positions,
            'transactions': pd.DataFrame(self.transactions)
        }
        
        logger.info(f"Backtest terminé: rendement cumulé du portefeuille de {self.portfolio_returns['cumulative_return'].iloc[-1]:.2%}")
        return results
    
    def _rebalance_portfolio(self, date):
        """
        Rebalance le portefeuille à une date donnée.
        
        Parameters:
        -----------
        date : pandas.Timestamp
            Date de rebalancement
        """
        # Obtenir les pondérations cibles
        target_weights = self.weights_df.loc[date]
        
        # Obtenir les prix actuels
        current_prices = self.price_matrix.loc[date]
        
        # Calculer la valeur actuelle du portefeuille
        if date == self.price_matrix.index[0]:
            current_value = self.initial_capital
        else:
            current_value = self.portfolio_value.loc[date]
        
        # Calculer les positions cibles (en nombre d'actions)
        target_positions = {}
        for ticker in self.price_matrix.columns:
            if pd.isna(current_prices[ticker]) or current_prices[ticker] == 0:
                target_positions[ticker] = 0
            else:
                # Calculer la position cible (arrondie au nombre d'actions entier)
                target_positions[ticker] = int((target_weights[ticker] * current_value) / current_prices[ticker])
        
        # Calculer les transactions nécessaires
        total_transaction_cost = 0
        for ticker in self.price_matrix.columns:
            if date == self.price_matrix.index[0]:
                current_position = 0
            else:
                current_position = self.positions.loc[date, ticker]
            
            # Calculer le nombre d'actions à acheter/vendre
            shares_to_trade = target_positions[ticker] - current_position
            
            if shares_to_trade != 0 and not pd.isna(current_prices[ticker]) and current_prices[ticker] > 0:
                # Calculer le coût de la transaction
                transaction_value = abs(shares_to_trade * current_prices[ticker])
                transaction_cost = transaction_value * self.transaction_cost
                total_transaction_cost += transaction_cost
                
                # Enregistrer la transaction
                self.transactions.append({
                    'date': date,
                    'ticker': ticker,
                    'shares': shares_to_trade,
                    'price': current_prices[ticker],
                    'value': transaction_value,
                    'cost': transaction_cost
                })
        
        # Mettre à jour les positions
        for ticker in self.price_matrix.columns:
            self.positions.loc[date, ticker] = target_positions[ticker]
        
        # Mettre à jour la valeur du portefeuille après les coûts de transaction
        actual_value = np.sum(self.positions.loc[date] * current_prices) - total_transaction_cost
        self.portfolio_value.loc[date] = actual_value
        
        logger.debug(f"Rebalancement effectué à {date}: valeur du portefeuille = {actual_value:.2f}, coûts de transaction = {total_transaction_cost:.2f}")
    
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
        
        # Ratio de Sharpe (supposons un taux sans risque de 0.02 ou 2%)
        risk_free_rate = 0.02
        portfolio_sharpe = (portfolio_annual_return - risk_free_rate) / portfolio_volatility
        index_sharpe = (index_annual_return - risk_free_rate) / index_volatility
        
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
            'total_transaction_costs': sum(transaction['cost'] for transaction in self.transactions),
            'number_of_transactions': len(self.transactions)
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
        
        # Calculer les métriques de performance
        metrics = self.calculate_performance_metrics(results)
        
        # Création de 3 sous-graphiques
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # 1. Évolution de la valeur du portefeuille
        axes[0].plot(portfolio_value.index, portfolio_value.values)
        axes[0].set_title('Évolution de la valeur du portefeuille')
        axes[0].set_ylabel('Valeur (€)')
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
        
        # Ajout de métriques de performance
        textstr = '\n'.join([
            f"Rendement cumulé: {metrics['portfolio']['cumulative_return']:.2%} (Ptf) vs {metrics['index']['cumulative_return']:.2%} (Idx)",
            f"Rendement annualisé: {metrics['portfolio']['annualized_return']:.2%} (Ptf) vs {metrics['index']['annualized_return']:.2%} (Idx)",
            f"Volatilité: {metrics['portfolio']['volatility']:.2%} (Ptf) vs {metrics['index']['volatility']:.2%} (Idx)",
            f"Tracking Error: {metrics['tracking_error']:.2%}",
            f"Information Ratio: {metrics['information_ratio']:.2f}",
            f"Coûts de transaction totaux: {metrics['total_transaction_costs']:.2f} €"
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
    parser = argparse.ArgumentParser(description='Backtest de réplication physique d\'un indice')
    
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
    parser.add_argument('--management-fee', type=float, default=0.0035,
                        help='Frais de gestion annuels (en pourcentage)')
    parser.add_argument('--transaction-cost', type=float, default=0.0020,
                        help='Coûts de transaction (en pourcentage)')
    parser.add_argument('--rebalance-frequency', type=str, default='quarterly',
                        choices=['daily', 'weekly', 'monthly', 'quarterly', 'yearly'],
                        help='Fréquence de rebalancement')
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
    
    # Initialiser la stratégie de réplication physique
    replication = PhysicalReplication(
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
            'Coûts de transaction totaux',
            'Nombre de transactions'
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
            f"{metrics['total_transaction_costs']:.2f} €",
            f"{metrics['number_of_transactions']}"
        ]
    })
    
    # Sauvegarder les métriques
    metrics_path = output_dir / f"{args.index}_physical_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    
    # Sauvegarder les résultats du portefeuille
    portfolio_path = output_dir / f"{args.index}_physical_portfolio.parquet"
    results['portfolio_value'].to_frame('portfolio_value').to_parquet(portfolio_path)
    
    # Sauvegarder les transactions
    transactions_path = output_dir / f"{args.index}_physical_transactions.csv"
    results['transactions'].to_csv(transactions_path, index=False)
    
    # Tracer et sauvegarder les graphiques
    plot_path = output_dir / f"{args.index}_physical_plot.png"
    replication.plot_results(results, save_path=plot_path)
    
    logger.info(f"Résultats sauvegardés dans {output_dir}")
    logger.info(f"Tracking Error: {metrics['tracking_error']:.4%}")


if __name__ == "__main__":
    main()
