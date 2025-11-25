# Add this at the VERY TOP of your file
print("=== TECHNICAL ANALYSIS SCRIPT STARTING ===")
import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime, timedelta

print("All imports successful!")


def calculate_technical_indicators(df):
    """Calculate various technical indicators"""
    print("Calculating technical indicators...")

    # Convert ALL numeric columns to proper numeric types
    numeric_columns = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    for col in numeric_columns:
        if col in df.columns:
            print(f"Converting {col} to numeric...")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Check for any NaN values after conversion
    nan_count = df["Close"].isna().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found in Close price after conversion")
        print("Dropping rows with NaN values...")
        df = df.dropna(subset=["Close"])
        print(f"Data shape after dropping NaN: {df.shape}")

    # RSI calculation
    print("Calculating RSI...")
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Moving Averages
    print("Calculating Moving Averages...")
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()

    # MACD
    print("Calculating MACD...")
    exp1 = df["Close"].ewm(span=12).mean()
    exp2 = df["Close"].ewm(span=26).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()

    # Bollinger Bands
    print("Calculating Bollinger Bands...")
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

    print("Technical indicators calculated successfully")
    return df


def main():
    print("Main function started")

    # File paths
    input_file = "../data/processed/cleaned_stock_data.csv"
    output_file = "../data/processed/technical_indicators.csv"

    print(f"Looking for input file: {input_file}")

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        print("Trying to download sample data...")

        # Download sample data
        try:
            print("Downloading AAPL stock data...")
            data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
            data = data.reset_index()
            print(f"Downloaded {len(data)} rows of data")

            # Create directory if it doesn't exist
            os.makedirs("../data/processed/", exist_ok=True)
            data.to_csv(input_file, index=False)
            print(f"Sample data saved to: {input_file}")
            df = data
        except Exception as e:
            print(f"Error downloading data: {e}")
            return
    else:
        # Load existing data
        print(f"Input file found: {input_file}")
        df = pd.read_csv(input_file)
        print(f"Data loaded successfully. Shape: {df.shape}")

        # DEBUG: Show detailed info about the data
        print(f"\n=== DATA DEBUG INFO ===")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Data types:\n{df.dtypes}")
        print(f"First 3 rows:")
        print(df.head(3))
        print(f"Close column sample values: {df['Close'].head(5).tolist()}")
        print(f"Close column type: {type(df['Close'].iloc[0])}")
        print("=======================\n")

    # Calculate technical indicators
    df = calculate_technical_indicators(df)

    # Display results
    print("\nTechnical Indicators Summary:")
    result_cols = ["Close", "RSI", "MA_20", "MA_50", "MACD"]
    available_cols = [col for col in result_cols if col in df.columns]
    print(df[available_cols].tail())

    # Save results
    df.to_csv(output_file, index=False)
    print(f"Technical indicators saved to: {output_file}")

    # Verify file creation
    if os.path.exists(output_file):
        print("SUCCESS: technical_indicators.csv created!")
        # Show file info
        file_size = os.path.getsize(output_file)
        print(f"File size: {file_size} bytes")
    else:
        print("ERROR: File was not created!")


# Make sure this is at the bottom of your file
if __name__ == "__main__":
    main()
    print("=== Script Finished ===")
