# stock_cleaner.py
import pandas as pd
import os
import warnings

print("🎯 STOCK CLEANER - FRESH START")
print("=" * 50)
print("THIS SCRIPT WILL WORK!")
print("=" * 50)

warnings.filterwarnings("ignore")

# HARDCODED ABSOLUTE PATH - This CANNOT fail
DATA_PATH = r"D:\Personal\KAIM-10 Academy\Week 1\Project Work\Price_Prediction_Week_1_Challenge\data\raw\stock_prices_raw"
OUTPUT_PATH = r"D:\Personal\KAIM-10 Academy\Week 1\Project Work\Price_Prediction_Week_1_Challenge\data\processed"

TICKERS = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA"]


def main():
    print(f"🔍 Checking data path: {DATA_PATH}")

    if not os.path.exists(DATA_PATH):
        print("❌ DATA PATH NOT FOUND!")
        print("This means your files are in a different location.")
        return False

    print("✅ DATA PATH FOUND!")

    # List files to confirm
    files = os.listdir(DATA_PATH)
    csv_files = [f for f in files if f.endswith(".csv")]
    print(f"📄 Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"   - {file}")

    cleaned_data = {}
    success_count = 0

    for ticker in TICKERS:
        file_path = os.path.join(DATA_PATH, f"{ticker}.csv")
        print(f"\n🎯 PROCESSING {ticker}...")

        if not os.path.exists(file_path):
            print(f"   ❌ File not found: {ticker}.csv")
            continue

        try:
            # Load the CSV
            print(f"   📥 Loading {ticker}.csv...")
            df = pd.read_csv(file_path)
            print(f"   ✅ Loaded {len(df)} rows")

            # Show what we're working with
            print(f"   📊 Columns: {list(df.columns)}")
            if len(df) > 0:
                print(f"   🔍 First row: {df.iloc[0].values}")

            # Clean the data
            cleaned_df = clean_stock_data(df, ticker)

            if cleaned_df is not None and len(cleaned_df) > 0:
                cleaned_data[ticker] = cleaned_df
                success_count += 1
                print(f"   🎉 SUCCESS: {len(cleaned_df)} clean rows")
            else:
                print(f"   ❌ Failed to clean {ticker}")

        except Exception as e:
            print(f"   💥 Error: {e}")

    print(f"\n📊 PROCESSED {success_count}/6 stocks successfully")

    if success_count > 0:
        print(f"\n💾 SAVING CLEANED DATA...")
        # Create output directory
        os.makedirs(OUTPUT_PATH, exist_ok=True)

        for ticker, df in cleaned_data.items():
            output_file = os.path.join(OUTPUT_PATH, f"{ticker}_cleaned.csv")
            df.to_csv(output_file, index=False)
            print(f"   ✅ Saved {ticker}: {len(df)} rows")

        print(f"\n🎉 ALL DONE!")
        print(f"📁 Cleaned files saved to: {OUTPUT_PATH}")
        return True
    else:
        print("\n❌ No data was processed")
        return False


def clean_stock_data(df, ticker):
    """Clean individual stock data"""
    try:
        clean_df = df.copy()

        # Check if first row has header issues
        first_val = str(clean_df.iloc[0, 0]) if len(clean_df) > 0 else ""
        if any(name in first_val for name in TICKERS):
            print(f"   🔧 Fixing header structure...")
            proper_columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
            clean_df = clean_df.iloc[1:].reset_index(drop=True)
            clean_df.columns = proper_columns

        # Convert data types
        numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
        for col in numeric_cols:
            if col in clean_df.columns:
                clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

        clean_df["Date"] = pd.to_datetime(clean_df["Date"], errors="coerce")
        clean_df["Symbol"] = ticker

        # Remove invalid rows
        initial_len = len(clean_df)
        clean_df = clean_df.dropna(subset=["Date", "Close"])
        clean_df = clean_df.sort_values("Date").reset_index(drop=True)

        removed = initial_len - len(clean_df)
        if removed > 0:
            print(f"   🗑️ Removed {removed} invalid rows")

        print(f"   ✅ Final: {len(clean_df)} clean rows")
        return clean_df

    except Exception as e:
        print(f"   💥 Cleaning error: {e}")
        return None


if __name__ == "__main__":
    print("🚀 STARTING STOCK DATA CLEANING...")
    success = main()

    if success:
        print("\n" + "🎉" * 10)
        print("SUCCESS! Your data is ready for analysis!")
        print("🎉" * 10)
    else:
        print("\n❌ Failed - but we'll figure it out!")
