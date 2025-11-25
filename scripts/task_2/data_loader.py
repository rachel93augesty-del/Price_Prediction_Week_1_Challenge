# scripts/task2/data_loader.py
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")


class StockDataLoader:
    def __init__(self):
        self.stock_data = {}
        self.data_info = {}
        self.tech_tickers = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA"]

    def load_tech_stocks(self, data_dir="data/raw/stock_prices_raw/"):
        """
        Load the 6 major tech stocks into pandas DataFrames
        """
        print("LOADING TECH STOCK DATA: AAPL, AMZN, GOOG, META, MSFT, NVDA")
        print("=" * 60)

        # Check if directory exists
        if not os.path.exists(data_dir):
            print("Directory not found:", data_dir)
            return False

        successful_loads = 0
        for ticker in self.tech_tickers:
            file_path = os.path.join(data_dir, f"{ticker}.csv")

            if not os.path.exists(file_path):
                print("File not found:", f"{ticker}.csv")
                continue

            try:
                # Load the CSV file
                df = pd.read_csv(file_path)
                print(f"\nLoading: {ticker}.csv")
                print("Initial shape:", df.shape)

                # Prepare the data
                df_cleaned = self.prepare_stock_data(df, ticker)

                if df_cleaned is not None:
                    self.stock_data[ticker] = df_cleaned
                    successful_loads += 1
                    print("Successfully loaded and prepared")

            except Exception as e:
                print("Error loading", ticker + ":", e)

        print(f"\nSUCCESSFULLY LOADED: {successful_loads}/6 tech stocks")
        return successful_loads > 0

    def prepare_stock_data(self, df, ticker):
        """
        Prepare and clean individual stock data with OHLCV columns
        """
        print("Preparing", ticker, "data...")

        # Create a copy to avoid modifying original
        df_clean = df.copy()

        # 1. Display available columns for debugging
        print("Available columns:", list(df_clean.columns))

        # 2. Standardize column names (handle different naming conventions)
        column_mapping = {}
        for col in df_clean.columns:
            col_lower = col.lower()
            if "open" in col_lower:
                column_mapping[col] = "Open"
            elif "high" in col_lower:
                column_mapping[col] = "High"
            elif "low" in col_lower:
                column_mapping[col] = "Low"
            elif "close" in col_lower or "price" in col_lower:
                column_mapping[col] = "Close"
            elif "volume" in col_lower:
                column_mapping[col] = "Volume"
            elif "date" in col_lower or "time" in col_lower:
                column_mapping[col] = "Date"

        if column_mapping:
            df_clean = df_clean.rename(columns=column_mapping)
            print("Standardized columns:", column_mapping)

        # 3. Check for required OHLCV columns
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [
            col for col in required_columns if col not in df_clean.columns
        ]

        if missing_columns:
            print("Missing required columns:", missing_columns)
            print("Current columns:", list(df_clean.columns))
            return None

        # 4. Handle date column
        if "Date" in df_clean.columns:
            # Convert to datetime
            df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")
            # Remove rows with invalid dates
            df_clean = df_clean.dropna(subset=["Date"])
            df_clean.set_index("Date", inplace=True)
            print("Set 'Date' as datetime index")
        else:
            print("No date column found, using default index")

        # 5. Data cleaning
        initial_rows = len(df_clean)

        # Remove rows with missing values in OHLCV
        df_clean = df_clean.dropna(subset=required_columns)

        # Remove rows with zero or negative prices using numpy
        price_columns = ["Open", "High", "Low", "Close"]
        for col in price_columns:
            df_clean = df_clean[df_clean[col] > 0]

        # Remove rows where High < Low (data inconsistency)
        df_clean = df_clean[df_clean["High"] >= df_clean["Low"]]

        cleaned_rows = len(df_clean)
        removed_rows = initial_rows - cleaned_rows
        if removed_rows > 0:
            print("Removed", removed_rows, "invalid rows")

        # 6. Sort by date (if available)
        if "Date" in df_clean.columns or df_clean.index.name == "Date":
            df_clean = df_clean.sort_index()
            print("Sorted by date")

        # 7. Calculate additional metrics using numpy
        df_clean["Daily_Return"] = df_clean["Close"].pct_change() * 100
        df_clean["Price_Range"] = (
            (df_clean["High"] - df_clean["Low"]) / df_clean["Low"]
        ) * 100
        df_clean["Log_Return"] = (
            np.log(df_clean["Close"] / df_clean["Close"].shift(1)) * 100
        )

        # Calculate volatility using numpy std
        df_clean["Volatility_5d"] = df_clean["Daily_Return"].rolling(window=5).std()
        df_clean["Volatility_20d"] = df_clean["Daily_Return"].rolling(window=20).std()

        # 8. Store data info
        if not df_clean.empty:
            total_return_pct = (
                df_clean["Close"].iloc[-1] / df_clean["Close"].iloc[0] - 1
            ) * 100
            avg_volume = df_clean["Volume"].mean()

            self.data_info[ticker] = {
                "original_rows": len(df),
                "cleaned_rows": len(df_clean),
                "date_range": str(df_clean.index.min())
                + " to "
                + str(df_clean.index.max()),
                "latest_close": "$" + f"{df_clean['Close'].iloc[-1]:.2f}",
                "data_points": len(df_clean),
                "total_return": f"{total_return_pct:.2f}%",
                "avg_daily_volume": f"{avg_volume:,.0f}",
            }
        else:
            self.data_info[ticker] = {
                "original_rows": len(df),
                "cleaned_rows": 0,
                "date_range": "No data",
                "latest_close": "N/A",
                "data_points": 0,
                "total_return": "N/A",
                "avg_daily_volume": "N/A",
            }

        return df_clean

    def get_tech_stocks_summary(self):
        """
        Generate a comprehensive summary of all 6 tech stocks
        """
        print("\n" + "=" * 80)
        print("TECH STOCKS DATA SUMMARY")
        print("=" * 80)

        header = f"\n{'Ticker':<8} {'Records':<10} {'Date Range':<25} {'Latest Close':<12} {'Total Return':<12} {'Avg Volume':<15} {'Status':<10}"
        print(header)
        print("-" * 80)

        for ticker in self.tech_tickers:
            if ticker in self.data_info:
                info = self.data_info[ticker]
                status = "LOADED" if ticker in self.stock_data else "FAILED"
                row = f"{ticker:<8} {info['cleaned_rows']:<10} {info['date_range']:<25} {info['latest_close']:<12} {info['total_return']:<12} {info['avg_daily_volume']:<15} {status:<10}"
                print(row)
            else:
                print(
                    f"{ticker:<8} {'N/A':<10} {'N/A':<25} {'N/A':<12} {'N/A':<12} {'N/A':<15} MISSING"
                )

    def get_stock_data(self, ticker):
        """
        Get data for a specific ticker
        """
        return self.stock_data.get(ticker)

    def get_all_data(self):
        """
        Get all loaded stock data
        """
        return self.stock_data

    def display_sample_data(self, ticker, n=3):
        """
        Display sample data for a specific ticker
        """
        if ticker in self.stock_data:
            df = self.stock_data[ticker]
            print(f"\nSAMPLE DATA: {ticker} (First {n} and Last {n} rows)")
            print("First rows:")
            print(
                df[["Open", "High", "Low", "Close", "Volume", "Daily_Return"]]
                .head(n)
                .round(2)
                .to_string()
            )
            print("\nLast rows:")
            print(
                df[["Open", "High", "Low", "Close", "Volume", "Daily_Return"]]
                .tail(n)
                .round(2)
                .to_string()
            )
        else:
            print("No data available for", ticker)


# Main execution function
def load_and_prepare_tech_stocks():
    """
    Main function to load and prepare all 6 tech stocks
    """
    loader = StockDataLoader()

    print("LOADING AND PREPARING STOCK DATA...")
    print("Required columns: Open, High, Low, Close, Volume")
    print("-" * 60)

    # Load all tech stocks
    success = loader.load_tech_stocks()

    if success and loader.stock_data:
        # Show comprehensive summary
        loader.get_tech_stocks_summary()

        # Show sample data for first successfully loaded stock
        first_loaded = list(loader.stock_data.keys())[0]
        loader.display_sample_data(first_loaded)

        print("\nDATA LOADING COMPLETE!")
        print(f"Loaded {len(loader.stock_data)} stocks ready for technical analysis")
        return loader
    else:
        print("\nFailed to load data. Please check:")
        print("   - File paths and naming")
        print("   - Required columns (Open, High, Low, Close, Volume)")
        print("   - File formats (should be CSV)")
        return None


if __name__ == "__main__":
    loader = load_and_prepare_tech_stocks()
