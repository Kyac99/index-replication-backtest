#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests unitaires pour les modèles de réplication d'indice.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ajouter le dossier src au path pour importer les modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.physical_replication import PhysicalReplication
from src.models.synthetic_replication import SyntheticReplication
from src.models.comparison import ReplicationComparison


class TestPhysicalReplication(unittest.TestCase):
    """
    Tests pour la classe PhysicalReplication.
    """
    
    def setUp(self):
        """
        Prépare les données de test.
        """
        # Créer un jeu de données synthétique
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='B')
        
        # Simuler les rendements de l'indice
        np.random.seed(42)
        index_returns = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)
        cum_returns = (1 + index_returns).cumprod() - 1
        
        self.index_returns = pd.DataFrame({
            'daily_return': index_returns,
            'cumulative_return': cum_returns
        })
        
        # Simuler les prix des composants
        components = ['A', 'B', 'C', 'D', 'E']
        
        # Matrice de prix
        prices = np.zeros((len(dates), len(components)))
        prices[0, :] = np.array([100.0, 50.0, 200.0, 75.0, 150.0])
        
        for i in range(1, len(dates)):
            # Corrélation avec l'indice + bruit spécifique
            component_returns = index_returns.iloc[i] + np.random.normal(0, 0.005, len(components))
            prices[i, :] = prices[i-1, :] * (1 + component_returns)
        
        self.price_matrix = pd.DataFrame(prices, index=dates, columns=components)
        
        # Matrice de rendements
        returns = np.zeros((len(dates), len(components)))
        returns[0, :] = 0.0
        
        for i in range(1, len(dates)):
            returns[i, :] = (prices[i, :] - prices[i-1, :]) / prices[i-1, :]
        
        self.return_matrix = pd.DataFrame(returns, index=dates, columns=components)
        
        # Pondérations égales
        self.weights_df = pd.DataFrame(np.ones((len(dates), len(components))) / len(components),
                                     index=dates, columns=components)
        
        # Créer les répertoires temporaires
        os.makedirs('tests/temp/processed/indices', exist_ok=True)
        os.makedirs('tests/temp/processed/components', exist_ok=True)
        os.makedirs('tests/temp/processed/weights', exist_ok=True)
        
        # Sauvegarder les données
        self.index_returns.to_parquet('tests/temp/processed/indices/TEST_index_returns.parquet')
        self.price_matrix.to_parquet('tests/temp/processed/components/TEST_price_matrix.parquet')
        self.return_matrix.to_parquet('tests/temp/processed/components/TEST_return_matrix.parquet')
        self.weights_df.to_parquet('tests/temp/processed/weights/TEST_weights.parquet')
    
    def tearDown(self):
        """
        Nettoie les fichiers temporaires après les tests.
        """
        # Supprimer les fichiers temporaires
        import shutil
        if os.path.exists('tests/temp'):
            shutil.rmtree('tests/temp')
    
    def test_initialization(self):
        """
        Teste l'initialisation de la classe PhysicalReplication.
        """
        replication = PhysicalReplication(
            index_name='TEST',
            data_dir='tests/temp/processed',
            initial_capital=1000000.0
        )
        
        self.assertEqual(replication.index_name, 'TEST')
        self.assertEqual(replication.initial_capital, 1000000.0)
        self.assertEqual(replication.management_fee, 0.0035)  # Valeur par défaut
        self.assertEqual(replication.transaction_cost, 0.0020)  # Valeur par défaut
        self.assertEqual(replication.rebalance_frequency, 'quarterly')  # Valeur par défaut
    
    def test_run_backtest(self):
        """
        Teste l'exécution du backtest.
        """
        replication = PhysicalReplication(
            index_name='TEST',
            data_dir='tests/temp/processed',
            initial_capital=1000000.0,
            rebalance_frequency='monthly'  # Pour avoir plusieurs rebalancements dans la période
        )
        
        results = replication.run_backtest()
        
        # Vérifier que les résultats contiennent les clés attendues
        self.assertIn('portfolio_value', results)
        self.assertIn('portfolio_returns', results)
        self.assertIn('index_returns', results)
        self.assertIn('tracking_error', results)
        self.assertIn('positions', results)
        self.assertIn('transactions', results)
        
        # Vérifier que la valeur initiale du portefeuille est correcte
        self.assertAlmostEqual(results['portfolio_value'].iloc[0], 1000000.0, delta=100)
        
        # Vérifier qu'il y a des transactions
        self.assertGreater(len(results['transactions']), 0)
    
    def test_calculate_performance_metrics(self):
        """
        Teste le calcul des métriques de performance.
        """
        replication = PhysicalReplication(
            index_name='TEST',
            data_dir='tests/temp/processed',
            initial_capital=1000000.0
        )
        
        results = replication.run_backtest()
        metrics = replication.calculate_performance_metrics(results)
        
        # Vérifier que les métriques contiennent les clés attendues
        self.assertIn('portfolio', metrics)
        self.assertIn('index', metrics)
        self.assertIn('tracking_error', metrics)
        self.assertIn('information_ratio', metrics)
        
        # Vérifier que les métriques du portefeuille sont calculées
        self.assertIn('cumulative_return', metrics['portfolio'])
        self.assertIn('annualized_return', metrics['portfolio'])
        self.assertIn('volatility', metrics['portfolio'])
        self.assertIn('sharpe_ratio', metrics['portfolio'])
        self.assertIn('max_drawdown', metrics['portfolio'])


class TestSyntheticReplication(unittest.TestCase):
    """
    Tests pour la classe SyntheticReplication.
    """
    
    def setUp(self):
        """
        Prépare les données de test.
        """
        # Créer un jeu de données synthétique
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='B')
        
        # Simuler les rendements de l'indice
        np.random.seed(42)
        index_returns = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)
        cum_returns = (1 + index_returns).cumprod() - 1
        
        self.index_returns = pd.DataFrame({
            'daily_return': index_returns,
            'cumulative_return': cum_returns
        })
        
        # Créer les répertoires temporaires
        os.makedirs('tests/temp/processed/indices', exist_ok=True)
        
        # Sauvegarder les données
        self.index_returns.to_parquet('tests/temp/processed/indices/TEST_index_returns.parquet')
    
    def tearDown(self):
        """
        Nettoie les fichiers temporaires après les tests.
        """
        # Supprimer les fichiers temporaires
        import shutil
        if os.path.exists('tests/temp'):
            shutil.rmtree('tests/temp')
    
    def test_initialization(self):
        """
        Teste l'initialisation de la classe SyntheticReplication.
        """
        replication = SyntheticReplication(
            index_name='TEST',
            data_dir='tests/temp/processed',
            initial_capital=1000000.0
        )
        
        self.assertEqual(replication.index_name, 'TEST')
        self.assertEqual(replication.initial_capital, 1000000.0)
        self.assertEqual(replication.management_fee, 0.0025)  # Valeur par défaut
        self.assertEqual(replication.swap_fee, 0.0015)  # Valeur par défaut
        self.assertEqual(replication.swap_reset_frequency, 'monthly')  # Valeur par défaut
        self.assertEqual(replication.risk_free_rate, 0.02)  # Valeur par défaut
    
    def test_run_backtest(self):
        """
        Teste l'exécution du backtest.
        """
        replication = SyntheticReplication(
            index_name='TEST',
            data_dir='tests/temp/processed',
            initial_capital=1000000.0,
            swap_reset_frequency='monthly'  # Pour avoir plusieurs resets dans la période
        )
        
        results = replication.run_backtest()
        
        # Vérifier que les résultats contiennent les clés attendues
        self.assertIn('portfolio_value', results)
        self.assertIn('portfolio_returns', results)
        self.assertIn('index_returns', results)
        self.assertIn('tracking_error', results)
        self.assertIn('cash_position', results)
        self.assertIn('swap_position', results)
        
        # Vérifier que la valeur initiale du portefeuille est correcte
        self.assertAlmostEqual(results['portfolio_value'].iloc[0], 1000000.0, delta=100)
    
    def test_calculate_performance_metrics(self):
        """
        Teste le calcul des métriques de performance.
        """
        replication = SyntheticReplication(
            index_name='TEST',
            data_dir='tests/temp/processed',
            initial_capital=1000000.0
        )
        
        results = replication.run_backtest()
        metrics = replication.calculate_performance_metrics(results)
        
        # Vérifier que les métriques contiennent les clés attendues
        self.assertIn('portfolio', metrics)
        self.assertIn('index', metrics)
        self.assertIn('tracking_error', metrics)
        self.assertIn('information_ratio', metrics)
        
        # Vérifier que les métriques du portefeuille sont calculées
        self.assertIn('cumulative_return', metrics['portfolio'])
        self.assertIn('annualized_return', metrics['portfolio'])
        self.assertIn('volatility', metrics['portfolio'])
        self.assertIn('sharpe_ratio', metrics['portfolio'])
        self.assertIn('max_drawdown', metrics['portfolio'])


class TestReplicationComparison(unittest.TestCase):
    """
    Tests pour la classe ReplicationComparison.
    """
    
    def setUp(self):
        """
        Prépare les données de test.
        """
        # Créer un jeu de données synthétique
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='B')
        
        # Simuler les rendements de l'indice
        np.random.seed(42)
        index_returns = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)
        cum_returns = (1 + index_returns).cumprod() - 1
        
        self.index_returns = pd.DataFrame({
            'daily_return': index_returns,
            'cumulative_return': cum_returns
        })
        
        # Simuler les prix des composants
        components = ['A', 'B', 'C', 'D', 'E']
        
        # Matrice de prix
        prices = np.zeros((len(dates), len(components)))
        prices[0, :] = np.array([100.0, 50.0, 200.0, 75.0, 150.0])
        
        for i in range(1, len(dates)):
            # Corrélation avec l'indice + bruit spécifique
            component_returns = index_returns.iloc[i] + np.random.normal(0, 0.005, len(components))
            prices[i, :] = prices[i-1, :] * (1 + component_returns)
        
        self.price_matrix = pd.DataFrame(prices, index=dates, columns=components)
        
        # Matrice de rendements
        returns = np.zeros((len(dates), len(components)))
        returns[0, :] = 0.0
        
        for i in range(1, len(dates)):
            returns[i, :] = (prices[i, :] - prices[i-1, :]) / prices[i-1, :]
        
        self.return_matrix = pd.DataFrame(returns, index=dates, columns=components)
        
        # Pondérations égales
        self.weights_df = pd.DataFrame(np.ones((len(dates), len(components))) / len(components),
                                     index=dates, columns=components)
        
        # Créer les répertoires temporaires
        os.makedirs('tests/temp/processed/indices', exist_ok=True)
        os.makedirs('tests/temp/processed/components', exist_ok=True)
        os.makedirs('tests/temp/processed/weights', exist_ok=True)
        os.makedirs('tests/temp/results', exist_ok=True)
        
        # Sauvegarder les données
        self.index_returns.to_parquet('tests/temp/processed/indices/TEST_index_returns.parquet')
        self.price_matrix.to_parquet('tests/temp/processed/components/TEST_price_matrix.parquet')
        self.return_matrix.to_parquet('tests/temp/processed/components/TEST_return_matrix.parquet')
        self.weights_df.to_parquet('tests/temp/processed/weights/TEST_weights.parquet')
    
    def tearDown(self):
        """
        Nettoie les fichiers temporaires après les tests.
        """
        # Supprimer les fichiers temporaires
        import shutil
        if os.path.exists('tests/temp'):
            shutil.rmtree('tests/temp')
    
    def test_initialization(self):
        """
        Teste l'initialisation de la classe ReplicationComparison.
        """
        comparison = ReplicationComparison(
            index_name='TEST',
            data_dir='tests/temp/processed',
            initial_capital=1000000.0
        )
        
        self.assertEqual(comparison.index_name, 'TEST')
        self.assertEqual(comparison.initial_capital, 1000000.0)
        self.assertIsNotNone(comparison.physical)
        self.assertIsNotNone(comparison.synthetic)
    
    def test_run_comparison(self):
        """
        Teste l'exécution de la comparaison.
        """
        comparison = ReplicationComparison(
            index_name='TEST',
            data_dir='tests/temp/processed',
            initial_capital=1000000.0
        )
        
        results = comparison.run_comparison()
        
        # Vérifier que les résultats contiennent les clés attendues
        self.assertIn('physical', results)
        self.assertIn('synthetic', results)
        self.assertIn('results', results['physical'])
        self.assertIn('metrics', results['physical'])
        self.assertIn('results', results['synthetic'])
        self.assertIn('metrics', results['synthetic'])
    
    def test_generate_report(self):
        """
        Teste la génération du rapport.
        """
        comparison = ReplicationComparison(
            index_name='TEST',
            data_dir='tests/temp/processed',
            initial_capital=1000000.0
        )
        
        results = comparison.run_comparison()
        report_dir = comparison.generate_report(results, 'tests/temp/results')
        
        # Vérifier que le répertoire du rapport existe
        self.assertTrue(os.path.exists(report_dir))
        
        # Vérifier que les fichiers principaux ont été créés
        self.assertTrue(os.path.exists(os.path.join(report_dir, 'TEST_comparison_plot.png')))
        self.assertTrue(os.path.exists(os.path.join(report_dir, 'TEST_comparison_metrics.csv')))
        self.assertTrue(os.path.exists(os.path.join(report_dir, 'TEST_comparison_params.csv')))


if __name__ == '__main__':
    unittest.main()
