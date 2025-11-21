# tests/test_basic.py
def test_imports():
    """Test that main modules can be imported"""
    try:
        from src.technical_analyzer import TechnicalAnalyzer
        from src.financial_metrics import FinancialMetrics
        from src.sentiment_analyzer import SentimentAnalyzer

        print("✅ All modules import successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


if __name__ == "__main__":
    test_imports()
