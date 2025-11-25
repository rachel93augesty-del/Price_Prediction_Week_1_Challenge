import pandas as pd
import os
from datetime import datetime, time, timedelta

print("🕐 PROPER TIMESTAMP NORMALIZATION")
print("=" * 50)


def normalize_news_timestamps(news_df):
    """
    Properly normalize news timestamps from UTC-4 to trading day alignment
    """
    print("📰 Normalizing news timestamps...")

    # Convert to datetime
    news_df["datetime_utc4"] = pd.to_datetime(news_df["date"])

    # Extract components for analysis
    news_df["original_date"] = news_df["datetime_utc4"].dt.date
    news_df["hour_utc4"] = news_df["datetime_utc4"].dt.hour
    news_df["day_of_week"] = news_df["datetime_utc4"].dt.day_name()

    print(
        f"   Original date range: {news_df['original_date'].min()} to {news_df['original_date'].max()}"
    )
    print(f"   Hour distribution (UTC-4):")
    print(news_df["hour_utc4"].value_counts().sort_index().head(10))

    # Trading day alignment logic
    def assign_trading_day(row):
        """
        Align news to proper trading day based on publication time
        - Market hours: 9:30 AM - 4:00 PM ET (13:30-20:00 UTC-4)
        - News after 4 PM ET → next trading day
        - Handle weekends (no trading Sat/Sun)
        """
        news_time = row["datetime_utc4"]
        hour = news_time.hour
        weekday = news_time.weekday()  # Monday=0, Sunday=6

        # If after market close (4 PM ET = 20:00 UTC-4)
        if hour >= 20:
            next_day = news_time + timedelta(days=1)
            # Skip weekends - find next trading day
            while next_day.weekday() >= 5:  # Saturday=5, Sunday=6
                next_day += timedelta(days=1)
            return next_day.date()

        # If before market open, check if it's a weekend
        elif hour < 13:  # Before 9:30 AM ET = 13:30 UTC-4
            if weekday >= 5:  # Weekend
                next_monday = news_time + timedelta(days=(7 - weekday))
                return next_monday.date()
            else:
                return news_time.date()

        else:  # During market hours
            if weekday >= 5:  # Weekend (shouldn't happen but just in case)
                next_monday = news_time + timedelta(days=(7 - weekday))
                return next_monday.date()
            else:
                return news_time.date()

    news_df["trading_date"] = news_df.apply(assign_trading_day, axis=1)

    # Analyze the normalization impact
    date_changes = (news_df["original_date"] != news_df["trading_date"]).sum()
    print(f"   📅 Date changes due to normalization: {date_changes} articles")
    print(
        f"   🗓️  Final trading date range: {news_df['trading_date'].min()} to {news_df['trading_date'].max()}"
    )

    return news_df


def normalize_stock_dates(stock_df):
    """
    Ensure stock dates are properly aligned to trading days
    """
    print("\n📈 Normalizing stock dates...")

    stock_df["datetime"] = pd.to_datetime(stock_df["Date"])
    stock_df["trading_date"] = stock_df["datetime"].dt.date

    # Verify these are actual trading days (no weekends)
    stock_df["day_of_week"] = stock_df["datetime"].dt.day_name()
    weekend_stocks = stock_df[stock_df["datetime"].dt.weekday >= 5]

    if len(weekend_stocks) > 0:
        print(f"   ⚠️  Found {len(weekend_stocks)} weekend records in stock data")
    else:
        print(f"   ✅ All stock dates are weekdays (trading days)")

    print(
        f"   📅 Stock trading date range: {stock_df['trading_date'].min()} to {stock_df['trading_date'].max()}"
    )

    return stock_df


def analyze_normalization_impact(news_df, stock_df):
    """
    Analyze how normalization affected alignment
    """
    print("\n🔍 Analyzing normalization impact...")

    # Before normalization (simple date matching)
    simple_alignment = pd.merge(
        news_df,
        stock_df,
        left_on=["stock", "original_date"],
        right_on=["symbol", "trading_date"],
        how="inner",
    )

    # After normalization (proper trading day alignment)
    normalized_alignment = pd.merge(
        news_df,
        stock_df,
        left_on=["stock", "trading_date"],
        right_on=["symbol", "trading_date"],
        how="inner",
    )

    print(f"   📊 ALIGNMENT COMPARISON:")
    print(f"      Simple date matching: {len(simple_alignment)} records")
    print(f"      Normalized trading day: {len(normalized_alignment)} records")
    print(
        f"      Improvement: {len(normalized_alignment) - len(simple_alignment)} records"
    )

    # Show timing analysis
    timing_analysis = news_df.groupby("hour_utc4").size()
    print(f"\n   🕐 NEWS TIMING DISTRIBUTION (UTC-4):")
    for hour, count in timing_analysis.sort_index().items():
        if count > 0:
            market_status = (
                "After Hours"
                if hour >= 20
                else "Market Hours"
                if hour >= 13
                else "Pre-Market"
            )
            print(f"      {hour:02d}:00 - {count:3d} articles ({market_status})")

    return normalized_alignment


def main():
    """Main execution with proper timestamp normalization"""
    print("🔄 Loading data for proper timestamp normalization...")

    # Load data
    news_df = pd.read_csv("data/processed/financial_data_cleaned.csv")
    stock_files = [
        "AAPL_cleaned.csv",
        "AMZN_cleaned.csv",
        "GOOG_cleaned.csv",
        "META_cleaned.csv",
        "MSFT_cleaned.csv",
        "NVDA_cleaned.csv",
    ]

    all_stocks = []
    for filename in stock_files:
        stock_df = pd.read_csv(f"data/processed/{filename}")
        stock_df["symbol"] = filename.replace("_cleaned.csv", "")
        all_stocks.append(stock_df)

    stock_df = pd.concat(all_stocks, ignore_index=True)

    # Filter news to available stocks
    available_stocks = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA"]
    filtered_news = news_df[news_df["stock"].isin(available_stocks)].copy()

    print(f"📊 Starting with {len(filtered_news)} news articles for available stocks")

    # Perform proper normalization
    normalized_news = normalize_news_timestamps(filtered_news)
    normalized_stocks = normalize_stock_dates(stock_df)

    # Analyze impact
    final_alignment = analyze_normalization_impact(normalized_news, normalized_stocks)

    # Save results
    if len(final_alignment) > 0:
        output_path = "data/processed/properly_normalized_alignment.csv"
        final_alignment.to_csv(output_path, index=False)
        print(f"\n💾 Saved properly normalized data to: {output_path}")
        print(f"🎯 Final aligned records: {len(final_alignment)}")

        # Show sample with timing info
        print(f"\n📝 SAMPLE WITH NORMALIZED TIMING:")
        sample_cols = [
            "stock",
            "original_date",
            "trading_date",
            "hour_utc4",
            "headline",
        ]
        print(final_alignment[sample_cols].head(5))
    else:
        print("\n❌ No records aligned after normalization")


if __name__ == "__main__":
    main()
