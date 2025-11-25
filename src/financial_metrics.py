# Create this file: src/financial_metrics.py
import pandas as pd
import numpy as np


class FinancialMetrics:
    """
    A class for calculating financial performance metrics
    """

    def __init__(self, data, price_column="Close"):
        self.data = data
        self.price_column = price_column
        self.metrics = {}

    def calculate_returns(self):
        """Calculate daily and cumulative returns"""
        self.data["Daily_Return"] = self.data[self.price_column].pct_change()
        self.data["Cumulative_Return"] = (1 + self.data["Daily_Return"]).cumprod()
        return self.data["Daily_Return"], self.data["Cumulative_Return"]

    def calculate_volatility(self, window=30):
        """Calculate rolling volatility"""
        returns = self.data["Daily_Return"].dropna()
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        self.metrics[f"Volatility_{window}d"] = rolling_vol
        return rolling_vol

    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        returns = self.data["Daily_Return"].dropna()
        excess_returns = returns - (risk_free_rate / 252)
        sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252)
        self.metrics["Sharpe_Ratio"] = sharpe
        return sharpe

    def calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        cumulative_returns = (1 + self.data["Daily_Return"].dropna()).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_dd = drawdown.min()
        self.metrics["Max_Drawdown"] = max_dd
        return max_dd

    def calculate_all_metrics(self):
        """Calculate all financial metrics"""
        print("Calculating financial metrics...")
        self.calculate_returns()
        self.calculate_volatility()
        sharpe = self.calculate_sharpe_ratio()
        max_dd = self.calculate_max_drawdown()

        print(f"✅ Sharpe Ratio: {sharpe:.4f}")
        print(f"✅ Max Drawdown: {max_dd:.2%}")

        return self.metrics
