import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import os

print("📊 TASK 3 - CORRELATION WITH AGGREGATED DAILY SENTIMENTS")
print("=" * 60)


def load_aggregated_data():
    """Load aggregated daily sentiments and stock returns"""
    print("📥 Loading aggregated data...")

    # Load aggregated daily sentiments
    aggregated_path = "data/processed/daily_aggregated_sentiments.csv"
    if os.path.exists(aggregated_path):
        daily_sentiments = pd.read_csv(aggregated_path)
        print(f"✅ Loaded aggregated sentiments: {len(daily_sentiments)} daily records")
    else:
        print("❌ No aggregated sentiments found. Please run sentiment analysis first.")
        return None, None

    # Load stock data to calculate daily returns
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
        stock_df["trading_date"] = pd.to_datetime(stock_df["Date"]).dt.date
        all_stocks.append(stock_df)

    stock_data = pd.concat(all_stocks, ignore_index=True)

    # Calculate daily returns for each stock
    stock_data = stock_data.sort_values(["symbol", "trading_date"])
    stock_data["daily_return_pct"] = (
        stock_data.groupby("symbol")["Close"].pct_change() * 100
    )

    print(f"✅ Loaded stock data: {stock_data['symbol'].nunique()} stocks")

    return daily_sentiments, stock_data


def merge_aggregated_sentiments_with_returns(daily_sentiments, stock_data):
    """Merge aggregated sentiments with daily stock returns"""
    print("\n🔗 Merging aggregated sentiments with daily returns...")

    # Merge on stock symbol and trading date
    merged_data = pd.merge(
        daily_sentiments,
        stock_data[["symbol", "trading_date", "daily_return_pct", "Close"]],
        left_on=["stock", "date"],
        right_on=["symbol", "trading_date"],
        how="inner",
    )

    # Remove records with NaN returns (first day of each stock)
    valid_data = merged_data.dropna(subset=["daily_return_pct"])

    print(f"✅ Merged dataset: {len(valid_data)} daily records")
    print(f"📊 Records by stock:")
    for stock in valid_data["stock"].unique():
        stock_records = valid_data[valid_data["stock"] == stock]
        print(f"   {stock}: {len(stock_records)} days")

    return valid_data


def calculate_aggregated_correlation(merged_data):
    """Calculate correlation using AGGREGATED daily sentiments"""
    print("\n📈 CALCULATING CORRELATION WITH AGGREGATED SENTIMENTS")
    print("-" * 55)

    if len(merged_data) == 0:
        print("❌ No data available for correlation analysis")
        return

    print(f"📊 Analyzing {len(merged_data)} daily aggregated records...")

    # 1. OVERALL CORRELATION (All stocks combined)
    print("\n🎯 OVERALL CORRELATION - AGGREGATED DAILY SENTIMENTS:")

    # Pearson correlation between AVERAGE daily sentiment and daily returns
    pearson_corr, pearson_p = pearsonr(
        merged_data["avg_vader_compound"], merged_data["daily_return_pct"]
    )

    print(f"   📊 Pearson Correlation:")
    print(f"      r = {pearson_corr:.3f}")
    print(f"      p-value = {pearson_p:.3f}")
    print(
        f"      Significance: {'SIGNIFICANT (p < 0.05)' if pearson_p < 0.05 else 'Not Significant'}"
    )

    # Spearman correlation
    spearman_corr, spearman_p = spearmanr(
        merged_data["avg_vader_compound"], merged_data["daily_return_pct"]
    )
    print(f"   📈 Spearman Correlation:")
    print(f"      ρ = {spearman_corr:.3f}")
    print(f"      p-value = {spearman_p:.3f}")

    # 2. CORRELATION BY STOCK
    print(f"\n📊 CORRELATION BY STOCK - AGGREGATED DAILY:")
    stock_results = []

    for stock in merged_data["stock"].unique():
        stock_data = merged_data[merged_data["stock"] == stock]
        if len(stock_data) >= 2:  # Minimum for correlation
            # Pearson correlation with AGGREGATED sentiments
            p_corr, p_pval = pearsonr(
                stock_data["avg_vader_compound"], stock_data["daily_return_pct"]
            )

            stock_results.append(
                {
                    "stock": stock,
                    "n_days": len(stock_data),
                    "pearson_r": p_corr,
                    "pearson_p": p_pval,
                    "significant": p_pval < 0.05,
                }
            )

            sig_indicator = " **" if p_pval < 0.05 else ""
            print(
                f"   {stock}: r = {p_corr:.3f}{sig_indicator} (p = {p_pval:.3f}, n = {len(stock_data)} days)"
            )

    # 3. MULTI-ARTICLE DAYS ANALYSIS
    print(f"\n📅 MULTI-ARTICLE DAYS ANALYSIS:")
    multi_article_days = merged_data[merged_data["article_count"] > 1]
    single_article_days = merged_data[merged_data["article_count"] == 1]

    print(f"   Days with multiple articles: {len(multi_article_days)}")
    print(f"   Days with single articles: {len(single_article_days)}")

    if len(multi_article_days) > 0:
        multi_corr, multi_p = pearsonr(
            multi_article_days["avg_vader_compound"],
            multi_article_days["daily_return_pct"],
        )
        print(
            f"   Correlation on multi-article days: r = {multi_corr:.3f} (p = {multi_p:.3f})"
        )

    # 4. SENTIMENT VOLATILITY ANALYSIS
    print(f"\n📊 SENTIMENT VOLATILITY ANALYSIS:")
    high_volatility = merged_data[merged_data["std_vader_compound"] > 0.1]
    low_volatility = merged_data[merged_data["std_vader_compound"] <= 0.1]

    print(f"   High sentiment volatility days: {len(high_volatility)}")
    print(f"   Low sentiment volatility days: {len(low_volatility)}")

    return merged_data, stock_results


def save_aggregated_correlation_results(merged_data, stock_results):
    """Save the aggregated correlation results"""
    print("\n💾 SAVING AGGREGATED CORRELATION RESULTS...")

    # Save merged dataset
    output_path = "data/processed/aggregated_correlation_data.csv"
    merged_data.to_csv(output_path, index=False)
    print(f"✅ Saved aggregated correlation data: {output_path}")

    # Save correlation results
    if stock_results:
        corr_df = pd.DataFrame(stock_results)
        corr_path = "data/processed/aggregated_correlation_results.csv"
        corr_df.to_csv(corr_path, index=False)
        print(f"✅ Saved aggregated correlation results: {corr_path}")

    # Save final report
    report_path = "data/processed/task3_aggregated_final_report.txt"
    with open(report_path, "w") as f:
        f.write("TASK 3 - FINAL REPORT WITH AGGREGATED SENTIMENTS\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Analysis completed with AGGREGATED daily sentiments\n")
        f.write(f"Total daily records analyzed: {len(merged_data)}\n")
        f.write(f"Stocks included: {', '.join(merged_data['stock'].unique())}\n\n")

        # Overall correlation
        pearson_corr, pearson_p = pearsonr(
            merged_data["avg_vader_compound"], merged_data["daily_return_pct"]
        )
        f.write("AGGREGATED CORRELATION RESULTS:\n")
        f.write(f"  Pearson Correlation (r): {pearson_corr:.3f}\n")
        f.write(f"  p-value: {pearson_p:.3f}\n")
        f.write(
            f"  Significance: {'SIGNIFICANT' if pearson_p < 0.05 else 'Not Significant'}\n\n"
        )

        f.write("REQUIREMENTS COMPLETED:\n")
        f.write("✅ Date alignment with timestamp normalization\n")
        f.write("✅ Sentiment analysis with TextBlob and VADER\n")
        f.write("✅ Daily sentiment aggregation for multi-article days\n")
        f.write("✅ Daily stock returns calculation\n")
        f.write("✅ Pearson correlation with aggregated sentiments\n")

    print(f"✅ Saved final report: {report_path}")
    print(f"📊 Analysis completed with {len(merged_data)} aggregated daily records")


def main():
    """Main execution function"""
    try:
        print("🚀 Starting Aggregated Correlation Analysis...")

        # 1. Load aggregated data
        daily_sentiments, stock_data = load_aggregated_data()

        if daily_sentiments is not None:
            # 2. Merge aggregated sentiments with returns
            merged_data = merge_aggregated_sentiments_with_returns(
                daily_sentiments, stock_data
            )

            # 3. Calculate correlation with aggregated data
            if len(merged_data) > 0:
                final_data, correlations = calculate_aggregated_correlation(merged_data)

                # 4. Save results
                save_aggregated_correlation_results(final_data, correlations)

                print(f"\n🎉 AGGREGATED CORRELATION ANALYSIS COMPLETED!")
                print(f"   📊 Used {len(final_data)} AGGREGATED daily records")
                print(f"   🔗 Pearson correlation with AVERAGE daily sentiments")
                print(f"   ✅ ALL Task 3 requirements properly implemented")

    except Exception as e:
        print(f"\n❌ ERROR in aggregated correlation analysis: {e}")


if __name__ == "__main__":
    main()
