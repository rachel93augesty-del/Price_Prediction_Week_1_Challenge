# scripts/task2/stock_data_manager.py
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")


class StockDataManager:
    def __init__(self):
        self.raw_data = {}
        self.cleaned_data = {}
        self.data_reports = {}
        self.tech_tickers = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA"]

    def load_and_prepare_stocks(self, data_dir="data/raw/stock_prices_raw/"):
        """
        Single function to load, validate, clean, and prepare all stock data
        """
        print("🚀 STOCK DATA: LOAD & PREPARE")
        print("=" * 50)

        if not os.path.exists(data_dir):
            print(f"❌ Directory not found: {data_dir}")
            return False

        successful = 0

        for ticker in self.tech_tickers:
            file_path = os.path.join(data_dir, f"{ticker}.csv")

            if not os.path.exists(file_path):
                print(f"❌ {ticker}: File not found")
                continue

            try:
                print(f"\n📊 {ticker}:")

                # Load data
                raw_df = pd.read_csv(file_path)
                print(f"   Loaded: {len(raw_df)} records")

                # Clean and prepare
                cleaned_df = self.clean_data(raw_df, ticker)

                if cleaned_df is not None:
                    self.cleaned_data[ticker] = cleaned_df
                    successful += 1
                    print(f"   ✅ Ready: {len(cleaned_df)} clean records")
                else:
                    print("   ❌ Failed to clean")

            except Exception as e:
                print(f"   ❌ Error: {e}")

        print(f"\n🎯 COMPLETE: {successful}/6 stocks ready for TA-Lib")
        return successful > 0

    def clean_data(self, df, ticker):
        """
        Clean and prepare individual stock data
        """
        df_clean = df.copy()

        # 1. Standardize columns
        df_clean = self.standardize_columns(df_clean)

        # 2. Check required columns
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df_clean.columns for col in required):
            missing = [col for col in required if col not in df_clean.columns]
            print(f"   Missing columns: {missing}")
            return None

        # 3. Handle dates
        if "Date" in df_clean.columns:
            df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")
            df_clean = df_clean.dropna(subset=["Date"])
            df_clean = df_clean.set_index("Date").sort_index()

        # 4. Clean data quality
        initial = len(df_clean)
        df_clean = self.clean_quality(df_clean)
        final = len(df_clean)

        if initial != final:
            print(f"   Removed {initial - final} invalid records")

        # 5. Add basic metrics
        df_clean = self.add_metrics(df_clean)

        # Store report
        self.data_reports[ticker] = {
            "initial": initial,
            "final": final,
            "date_range": f"{df_clean.index.min()} to {df_clean.index.max()}"
            if not df_clean.empty
            else "No data",
        }

        return df_clean

    def standardize_columns(self, df):
        """Standardize column names"""
        mapping = {}
        for col in df.columns:
            low = col.lower()
            if "open" in low:
                mapping[col] = "Open"
            elif "high" in low:
                mapping[col] = "High"
            elif "low" in low:
                mapping[col] = "Low"
            elif any(x in low for x in ["close", "price"]):
                mapping[col] = "Close"
            elif "volume" in low:
                mapping[col] = "Volume"
            elif any(x in low for x in ["date", "time"]):
                mapping[col] = "Date"

        if mapping:
            df = df.rename(columns=mapping)
        return df

    def clean_quality(self, df):
        """Clean data quality issues"""
        # Remove missing values
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

        # Remove invalid prices
        for col in ["Open", "High", "Low", "Close"]:
            df = df[df[col] > 0]

        # Remove inconsistencies
        df = df[df["High"] >= df["Low"]]
        df = df[df["High"] >= df["Open"]]
        df = df[df["High"] >= df["Close"]]
        df = df[df["Low"] <= df["Open"]]
        df = df[df["Low"] <= df["Close"]]

        return df

    def add_metrics(self, df):
        """Add basic pre-analysis metrics"""
        df["Daily_Return"] = df["Close"].pct_change() * 100
        df["Price_Range"] = ((df["High"] - df["Low"]) / df["Low"]) * 100
        return df

    def show_summary(self):
        """Show loading summary"""
        print("\n" + "=" * 60)
        print("📋 DATA SUMMARY")
        print("=" * 60)

        print(f"\n{'Ticker':<8} {'Initial':<8} {'Final':<8} {'Date Range':<30}")
        print("-" * 60)

        for ticker in self.tech_tickers:
            if ticker in self.data_reports:
                report = self.data_reports[ticker]
                print(
                    f"{ticker:<8} {report['initial']:<8} {report['final']:<8} {report['date_range']:<30}"
                )
            else:
                print(f"{ticker:<8} {'N/A':<8} {'N/A':<8} {'MISSING':<30}")

    def get_clean_data(self, ticker):
        """Get cleaned data for a ticker"""
        return self.cleaned_data.get(ticker)

    def get_all_clean_data(self):
        """Get all cleaned data"""
        return self.cleaned_data

    def save_clean_data(self, output_dir="data/processed/technical_indicators/"):
        """Save cleaned data"""
        os.makedirs(output_dir, exist_ok=True)

        for ticker, df in self.cleaned_data.items():
            path = os.path.join(output_dir, f"{ticker}_cleaned.csv")
            df.to_csv(path)
            print(f"💾 Saved: {path}")


# Main function
def prepare_stock_data():
    """Prepare all stock data for TA-Lib analysis"""
    manager = StockDataManager()

    print("🔧 PREPARING STOCK DATA FOR TECHNICAL ANALYSIS")
    print("-" * 50)

    success = manager.load_and_prepare_stocks()

    if success:
        manager.show_summary()
        manager.save_clean_data()
        print("\n✅ READY FOR TA-LIB ANALYSIS!")
        return manager
    else:
        print("\n❌ Preparation failed")
        return None


if __name__ == "__main__":
    manager = prepare_stock_data()
