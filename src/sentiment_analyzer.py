# src/sentiment_analyzer.py
import pandas as pd
import re


class SentimentAnalyzer:
    """
    Basic sentiment analysis for financial news
    """

    def __init__(self):
        # Financial sentiment dictionaries
        self.positive_words = {
            "profit",
            "growth",
            "gain",
            "rise",
            "up",
            "bullish",
            "positive",
            "strong",
            "beat",
            "surge",
            "rally",
            "outperform",
            "buy",
            "upgrade",
            "earnings",
            "revenue",
            "success",
            "win",
            "opportunity",
        }

        self.negative_words = {
            "loss",
            "fall",
            "drop",
            "down",
            "bearish",
            "negative",
            "weak",
            "miss",
            "plunge",
            "decline",
            "underperform",
            "sell",
            "downgrade",
            "risk",
            "warning",
            "failure",
            "crisis",
            "volatility",
        }

    def basic_sentiment_score(self, text):
        """Calculate basic sentiment score based on keyword matching"""
        if pd.isna(text):
            return 0

        text = str(text).lower()
        words = re.findall(r"\b\w+\b", text)

        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)

        total_words = len(words)
        if total_words == 0:
            return 0

        # Simple sentiment score: (positive - negative) / total_words
        sentiment = (positive_count - negative_count) / total_words
        return sentiment
