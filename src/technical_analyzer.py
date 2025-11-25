# Create this file: src/technical_analyzer.py
import pandas as pd
import numpy as np


class TechnicalAnalyzer:
    """
    A class for calculating technical indicators for stock price analysis
    """

    def __init__(self, data):
        self.data = data
        self.indicators = {}

    def calculate_rsi(self, period=14, column="Close"):
        """Calculate Relative Strength Index"""
        delta = self.data[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        self.indicators["RSI"] = rsi
        return rsi

    def calculate_moving_averages(self, windows=[20, 50], column="Close"):
        """Calculate multiple moving averages"""
        for window in windows:
            ma_name = f"MA_{window}"
            self.indicators[ma_name] = self.data[column].rolling(window=window).mean()
        return {f"MA_{window}": self.indicators[f"MA_{window}"] for window in windows}

    def calculate_macd(self, column="Close"):
        """Calculate MACD indicator"""
        exp1 = self.data[column].ewm(span=12).mean()
        exp2 = self.data[column].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()

        self.indicators["MACD"] = macd
        self.indicators["MACD_Signal"] = signal
        self.indicators["MACD_Histogram"] = macd - signal

        return {"MACD": macd, "MACD_Signal": signal, "MACD_Histogram": macd - signal}

    def calculate_bollinger_bands(self, window=20, column="Close"):
        """Calculate Bollinger Bands"""
        middle_band = self.data[column].rolling(window=window).mean()
        std = self.data[column].rolling(window=window).std()

        upper_band = middle_band + (std * 2)
        lower_band = middle_band - (std * 2)

        self.indicators["BB_Upper"] = upper_band
        self.indicators["BB_Middle"] = middle_band
        self.indicators["BB_Lower"] = lower_band

        return {
            "BB_Upper": upper_band,
            "BB_Middle": middle_band,
            "BB_Lower": lower_band,
        }

    def calculate_all_indicators(self):
        """Calculate all technical indicators"""
        print("Calculating all technical indicators...")
        self.calculate_rsi()
        self.calculate_moving_averages()
        self.calculate_macd()
        self.calculate_bollinger_bands()

        # Add all indicators to dataframe
        for name, values in self.indicators.items():
            self.data[name] = values

        print(f"✅ Added {len(self.indicators)} technical indicators")
        return self.data
