#!/usr/bin/env python3
"""
test 42 - Risk_adjusted_momentum, mean_reversion Trading Strategy

Strategy Type: risk_adjusted_momentum, mean_reversion
Description: test 42
Created: 2025-06-24T23:37:43.973Z

WARNING: This is a template implementation. Thoroughly backtest before live trading.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class test42Strategy:
    """
    test 42 Implementation
    
    Strategy Type: risk_adjusted_momentum, mean_reversion
    Risk Level: Monitor drawdowns and position sizes carefully
    """
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.positions = {}
        self.performance_metrics = {}
        logger.info(f"Initialized test 42 strategy")
        
    def get_default_config(self):
        """Default configuration parameters"""
        return {
            'max_position_size': 0.05,  # 5% max position size
            'stop_loss_pct': 0.05,      # 5% stop loss
            'lookback_period': 20,       # 20-day lookback
            'rebalance_freq': 'daily',   # Rebalancing frequency
            'transaction_costs': 0.001,  # 0.1% transaction costs
        }
    
    def load_data(self, symbols, start_date, end_date):
        """Load market data for analysis"""
        try:
            import yfinance as yf
            data = yf.download(symbols, start=start_date, end=end_date)
            logger.info(f"Loaded data for {len(symbols)} symbols")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

# =============================================================================
# USER'S STRATEGY IMPLEMENTATION
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Strategy class definition
class Test42Strategy:
    def __init__(self, data, momentum_window=20, mean_rev_window=5, risk_free_rate=0.01, vol_window=20, max_position=1.0, risk_per_trade=0.02):
        # Initialize parameters
        self.data = data.copy()
        self.momentum_window = momentum_window
        self.mean_rev_window = mean_rev_window
        self.risk_free_rate = risk_free_rate
        self.vol_window = vol_window
        self.max_position = max_position
        self.risk_per_trade = risk_per_trade
        self.signals = pd.Series(index=self.data.index, dtype=float)
        self.positions = pd.Series(index=self.data.index, dtype=float)
        self.returns = pd.Series(index=self.data.index, dtype=float)
        self.equity_curve = pd.Series(index=self.data.index, dtype=float)
        self._prepare_data()
    
    def _prepare_data(self):
        # Calculate log returns
        self.data['log_ret'] = np.log(self.data['Close']).diff()
        # Calculate rolling volatility
        self.data['vol'] = self.data['log_ret'].rolling(self.vol_window).std()
        # Calculate momentum (risk-adjusted)
        self.data['momentum'] = (self.data['Close'].pct_change(self.momentum_window)) / (self.data['vol'])
        # Calculate mean reversion z-score
        self.data['mean_rev'] = (self.data['Close'] - self.data['Close'].rolling(self.mean_rev_window).mean()) / self.data['Close'].rolling(self.mean_rev_window).std()
        # Handle NaNs
        self.data.fillna(0, inplace=True)
    
    def generate_signals(self, momentum_thresh=1.0, mean_rev_thresh=1.0):
        # Risk-adjusted momentum signal
        momentum_signal = np.where(self.data['momentum'] > momentum_thresh, 1,
                            np.where(self.data['momentum'] < -momentum_thresh, -1, 0))
        # Mean reversion signal (contrarian)
        mean_rev_signal = np.where(self.data['mean_rev'] > mean_rev_thresh, -1,
                            np.where(self.data['mean_rev'] < -mean_rev_thresh, 1, 0))
        # Combine signals (average)
        combined_signal = (momentum_signal + mean_rev_signal) / 2.0
        # Clip to [-1, 1]
        combined_signal = np.clip(combined_signal, -1, 1)
        self.signals = pd.Series(combined_signal, index=self.data.index)
        logging.info("Signals generated.")
    
    def position_sizing(self):
        # Volatility targeting position sizing
        dollar_vol = self.data['vol'] * self.data['Close']
        with np.errstate(divide='ignore', invalid='ignore'):
            pos_size = self.risk_per_trade / (dollar_vol + 1e-8)
        pos_size = np.minimum(pos_size, self.max_position)
        pos_size = np.maximum(pos_size, 0)
        self.positions = self.signals * pos_size
        logging.info("Position sizing completed.")
    
    def backtest(self):
        # Calculate strategy returns
        self.returns = self.positions.shift(1) * self.data['log_ret']
        self.returns.fillna(0, inplace=True)
        self.equity_curve = (1 + self.returns).cumprod()
        logging.info("Backtest completed.")
    
    def calculate_performance(self):
        # Sharpe Ratio
        ann_factor = 252
        mean_ret = self.returns.mean() * ann_factor
        std_ret = self.returns.std() * np.sqrt(ann_factor)
        if std_ret != 0:
            sharpe = (mean_ret - self.risk_free_rate) / std_ret
        else:
            sharpe = 0
        # Max Drawdown
        cum_returns = (1 + self.returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        # Total Return
        total_return = cum_returns.iloc[-1] - 1
        # Win Rate
        win_rate = (self.returns > 0).sum() / len(self.returns)
        # Output
        print("Sharpe Ratio:", sharpe)
        print("Max Drawdown:", max_drawdown)
        print("Total Return:", total_return)
        print("Win Rate:", win_rate)
        return {'sharpe': sharpe, 'max_drawdown': max_drawdown, 'total_return': total_return, 'win_rate': win_rate}
    
    def plot_results(self):
        plt.figure(figsize=(12,6))
        plt.plot(self.equity_curve, label='Equity Curve')
        plt.title('Test 42 Strategy Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.legend()
        plt.grid(True)
        plt.show()

# Sample data generation
def generate_sample_data(length=500, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start='2020-01-01', periods=length, freq='B')
    # Simulate a random walk with drift
    drift = 0.0002
    vol = 0.01
    returns = np.random.normal(loc=drift, scale=vol, size=length)
    price = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({'Close': price}, index=dates)
    return df

# Main execution block
if __name__ == "__main__":
    try:
        # Generate sample data
        data = generate_sample_data()
        # Initialize strategy
        strategy = Test42Strategy(data)
        # Generate signals
        strategy.generate_signals(momentum_thresh=1.0, mean_rev_thresh=1.0)
        # Position sizing
        strategy.position_sizing()
        # Backtest
        strategy.backtest()
        # Performance metrics
        perf = strategy.calculate_performance()
        # Plot results
        strategy.plot_results()
    except Exception as e:
        logging.error("Error running strategy: %s" % str(e))

# =============================================================================
# STRATEGY EXECUTION AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Example usage and testing
    strategy = test42Strategy()
    print(f"Strategy '{strategyName}' initialized successfully!")
    
    # Example data loading
    symbols = ['SPY', 'QQQ', 'IWM']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    print(f"Loading data for symbols: {symbols}")
    data = strategy.load_data(symbols, start_date, end_date)
    
    if data is not None:
        print(f"Data loaded successfully. Shape: {data.shape}")
        print("Strategy ready for backtesting!")
    else:
        print("Failed to load data. Check your internet connection.")
