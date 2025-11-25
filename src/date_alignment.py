import pandas as pd
import os

print("🚀 STARTING TASK 3 - DATE ALIGNMENT")
print("=" * 50)


def load_news_data():
    """Load financial news data from Task 1"""
    print("📰 Loading news data...")
    news_path = "data/processed/financial_data_cleaned.csv"
    news_df = pd.read_csv(news_path)
    print(f"   ✅ News data: {news_df.shape[0]} articles, {news_df.shape[1]} columns")
    print(f"   📊 Sample stocks: {news_df['stock'].value_counts().head()}")
    return news_df


def load_stock_data():
    """Load all stock data from Task 2"""
    print("\n📈 Loading stock data...")
    stock_files = {
        "AAPL": "AAPL_cleaned.csv",
        "AMZN": "AMZN_cleaned.csv",
        "GOOG": "GOOG_cleaned.csv",
        "META": "META_cleaned.csv",
        "MSFT": "MSFT_cleaned.csv",
        "NVDA": "NVDA_cleaned.csv",
    }

    all_stocks = []
    for symbol, filename in stock_files.items():
        stock_df = pd.read_csv(f"data/processed/{filename}")
        stock_df["symbol"] = symbol
        all_stocks.append(stock_df)
        print(f"   ✅ {symbol}: {stock_df.shape[0]} records")

    combined_stocks = pd.concat(all_stocks, ignore_index=True)
    print(f"   📦 Total stock records: {combined_stocks.shape[0]}")
    return combined_stocks


def normalize_dates(news_df, stock_df):
    """Normalize dates for alignment"""
    print("\n🔄 Normalizing dates...")

    # News data uses 'date' column
    news_df["trading_date"] = pd.to_datetime(news_df["date"]).dt.date

    # Stock data uses 'Date' column (capital D)
    stock_df["trading_date"] = pd.to_datetime(stock_df["Date"]).dt.date

    print(
        f"   📅 News date range: {news_df['trading_date'].min()} to {news_df['trading_date'].max()}"
    )
    print(
        f"   📅 Stock date range: {stock_df['trading_date'].min()} to {stock_df['trading_date'].max()}"
    )

    return news_df, stock_df


def align_datasets(news_df, stock_df):
    """Align news and stock datasets by date and symbol"""
    print("\n🔗 Aligning datasets...")

    # Merge on trading date and stock symbol
    merged_df = pd.merge(
        news_df,
        stock_df,
        left_on=["stock", "trading_date"],
        right_on=["symbol", "trading_date"],
        how="inner",
        suffixes=("_news", "_stock"),
    )

    print(f"   ✅ Merged dataset: {merged_df.shape[0]} records")
    return merged_df


def analyze_alignment(merged_df):
    """Analyze the alignment results"""
    print("\n📊 Analyzing alignment results...")

    print(f"   🏢 Unique stocks: {merged_df['stock'].nunique()}")
    print(f"   📅 Unique trading days: {merged_df['trading_date'].nunique()}")

    # Show alignment by stock
    alignment_by_stock = merged_df.groupby("stock").size().sort_values(ascending=False)
    print(f"\n   📈 News articles per stock:")
    for stock, count in alignment_by_stock.items():
        print(f"      {stock}: {count} articles")

    # Show date coverage
    date_coverage = (
        merged_df.groupby("trading_date").size().sort_values(ascending=False)
    )
    print(f"\n   🗓️  Top dates by article count:")
    for date, count in date_coverage.head(5).items():
        print(f"      {date}: {count} articles")


def save_results(merged_df):
    """Save the aligned dataset"""
    print("\n💾 Saving results...")
    output_path = "data/processed/aligned_news_stock_data.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"   ✅ Saved to: {output_path}")
    print(f"   💾 File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    return output_path


def main():
    """Main execution function"""
    try:
        # Load data
        news_data = load_news_data()
        stock_data = load_stock_data()

        # Normalize dates
        news_normalized, stock_normalized = normalize_dates(news_data, stock_data)

        # Align datasets
        aligned_data = align_datasets(news_normalized, stock_normalized)

        if aligned_data.shape[0] > 0:
            # Analyze results
            analyze_alignment(aligned_data)

            # Save results
            save_results(aligned_data)

            print(f"\n🎉 TASK 3 - DATE ALIGNMENT COMPLETED SUCCESSFULLY!")
            print(f"   📈 Ready for sentiment analysis and correlation!")
            print(f"   🔗 {aligned_data.shape[0]} news-stock pairs aligned")
        else:
            print(f"\n❌ ALIGNMENT FAILED: No matching records found")
            print("   Check if news and stock data have overlapping dates and symbols")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("   Please check your data files and file paths")


if __name__ == "__main__":
    main()
