import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os

print("🎭 TASK 3 - SENTIMENT ANALYSIS ON NEWS HEADLINES")
print("=" * 55)

# Download required NLTK data
try:
    nltk.download("vader_lexicon", quiet=True)
    print("✅ VADER lexicon downloaded successfully")
except Exception as e:
    print(f"⚠️  NLTK download issue: {e}")


def load_aligned_data():
    """Load the properly aligned and normalized data"""
    print("📥 Loading aligned news-stock data...")

    # Try the properly normalized data first
    file_path = "data/processed/properly_normalized_alignment.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"✅ Loaded: {file_path}")
    else:
        # Fallback to basic aligned data
        file_path = "data/processed/aligned_news_stock_data.csv"
        df = pd.read_csv(file_path)
        print(f"✅ Loaded: {file_path}")

    print(f"   Records: {df.shape[0]}")
    print(f"   Stocks: {df['stock'].nunique()} ({', '.join(df['stock'].unique())})")
    print(f"   Date range: {df['trading_date'].min()} to {df['trading_date'].max()}")

    return df


def analyze_sentiment_textblob(headline):
    """Analyze sentiment using TextBlob"""
    analysis = TextBlob(str(headline))
    return {
        "tb_polarity": analysis.sentiment.polarity,
        "tb_subjectivity": analysis.sentiment.subjectivity,
    }


def analyze_sentiment_vader(headline):
    """Analyze sentiment using VADER (specifically for financial text)"""
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(str(headline))
    return scores


def classify_sentiment(compound_score, polarity_score):
    """Classify sentiment into categories"""
    # Primary classification based on VADER compound score
    if compound_score >= 0.05:
        sentiment = "Positive"
        strength = "Strong" if compound_score > 0.3 else "Moderate"
    elif compound_score <= -0.05:
        sentiment = "Negative"
        strength = "Strong" if compound_score < -0.3 else "Moderate"
    else:
        sentiment = "Neutral"
        strength = "Neutral"

    return sentiment, strength


def perform_comprehensive_sentiment_analysis(df):
    """Perform comprehensive sentiment analysis on all headlines"""
    print("\n🔍 Performing comprehensive sentiment analysis...")

    sentiment_results = []

    for idx, row in df.iterrows():
        headline = row["headline"]
        stock = row["stock"]
        date = row["trading_date"]

        # TextBlob analysis
        tb_results = analyze_sentiment_textblob(headline)

        # VADER analysis (better for financial text)
        vader_results = analyze_sentiment_vader(headline)

        # Classify sentiment
        sentiment, strength = classify_sentiment(
            vader_results["compound"], tb_results["tb_polarity"]
        )

        result = {
            "headline": headline,
            "stock": stock,
            "date": date,
            # TextBlob metrics
            "tb_polarity": tb_results["tb_polarity"],
            "tb_subjectivity": tb_results["tb_subjectivity"],
            # VADER metrics
            "vader_compound": vader_results["compound"],
            "vader_positive": vader_results["pos"],
            "vader_negative": vader_results["neg"],
            "vader_neutral": vader_results["neu"],
            # Classification
            "sentiment": sentiment,
            "strength": strength,
            # Combined confidence
            "confidence": abs(vader_results["compound"]),
        }

        sentiment_results.append(result)

        # Show progress for first few
        if idx < 3:
            print(
                f"   Sample analysis: '{headline[:50]}...' → {sentiment} ({strength})"
            )

    # Convert to DataFrame
    sentiment_df = pd.DataFrame(sentiment_results)

    print(f"✅ Completed sentiment analysis on {len(sentiment_df)} headlines")
    return sentiment_df


def aggregate_daily_sentiments(sentiment_df):
    """
    AGGREGATE SENTIMENTS: Compute average daily sentiment scores
    for days with multiple articles
    """
    print("\n📊 AGGREGATING DAILY SENTIMENTS...")

    # Group by stock and trading date, calculate average sentiments
    daily_aggregated = (
        sentiment_df.groupby(["stock", "date"])
        .agg(
            {
                "vader_compound": ["mean", "std", "count"],
                "tb_polarity": "mean",
                "headline": "count",
            }
        )
        .round(4)
    )

    # Flatten column names
    daily_aggregated.columns = [
        "avg_vader_compound",
        "std_vader_compound",
        "vader_count",
        "avg_tb_polarity",
        "article_count",
    ]
    daily_aggregated = daily_aggregated.reset_index()

    print(
        f"✅ Aggregated {len(sentiment_df)} articles into {len(daily_aggregated)} daily records"
    )

    # Show aggregation results
    multi_article_days = daily_aggregated[daily_aggregated["article_count"] > 1]
    print(f"   Days with multiple articles: {len(multi_article_days)}")

    if len(multi_article_days) > 0:
        print(f"   Sample aggregated days:")
        for _, row in multi_article_days.head(3).iterrows():
            print(
                f"      {row['stock']} on {row['date']}: {row['article_count']} articles, "
                f"Avg VADER: {row['avg_vader_compound']:.3f}"
            )

    return daily_aggregated


def analyze_sentiment_distribution(sentiment_df):
    """Analyze the distribution and patterns in sentiment scores"""
    print("\n📊 SENTIMENT DISTRIBUTION ANALYSIS")
    print("-" * 40)

    # Overall sentiment distribution
    sentiment_counts = sentiment_df["sentiment"].value_counts()
    print("🎯 OVERALL SENTIMENT DISTRIBUTION:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(sentiment_df)) * 100
        print(f"   {sentiment}: {count} articles ({percentage:.1f}%)")

    # Sentiment by stock
    print(f"\n📈 SENTIMENT BY STOCK:")
    stock_sentiment = (
        sentiment_df.groupby(["stock", "sentiment"]).size().unstack(fill_value=0)
    )
    print(stock_sentiment)

    # Score statistics
    print(f"\n📊 SENTIMENT SCORE STATISTICS:")
    score_stats = sentiment_df[["tb_polarity", "vader_compound"]].describe()
    print(score_stats.round(3))

    # Strong sentiment analysis
    strong_positive = sentiment_df[sentiment_df["vader_compound"] > 0.5]
    strong_negative = sentiment_df[sentiment_df["vader_compound"] < -0.5]

    print(f"\n🔥 STRONG SENTIMENT ANALYSIS:")
    print(f"   Strong Positive (>0.5): {len(strong_positive)} articles")
    print(f"   Strong Negative (<-0.5): {len(strong_negative)} articles")

    if len(strong_positive) > 0:
        print(
            f"   Sample Strong Positive: '{strong_positive.iloc[0]['headline'][:60]}...'"
        )
    if len(strong_negative) > 0:
        print(
            f"   Sample Strong Negative: '{strong_negative.iloc[0]['headline'][:60]}...'"
        )

    return sentiment_df


def save_sentiment_results(sentiment_df, original_df):
    """Save sentiment analysis results with aggregation"""
    print("\n💾 Saving results...")

    # Merge sentiment results with original data
    final_df = pd.merge(
        original_df,
        sentiment_df[
            [
                "headline",
                "tb_polarity",
                "tb_subjectivity",
                "vader_compound",
                "vader_positive",
                "vader_negative",
                "vader_neutral",
                "sentiment",
                "strength",
                "confidence",
            ]
        ],
        on="headline",
        how="left",
    )

    # AGGREGATE DAILY SENTIMENTS (NEW REQUIREMENT)
    daily_aggregated = aggregate_daily_sentiments(sentiment_df)

    # Save comprehensive results
    output_path = "data/processed/news_with_sentiment.csv"
    final_df.to_csv(output_path, index=False)

    # Save aggregated sentiments
    aggregated_path = "data/processed/daily_aggregated_sentiments.csv"
    daily_aggregated.to_csv(aggregated_path, index=False)

    print(f"✅ Saved comprehensive sentiment analysis to: {output_path}")
    print(f"✅ Saved daily aggregated sentiments to: {aggregated_path}")
    print(f"📊 Final dataset: {final_df.shape[0]} records, {final_df.shape[1]} columns")

    # Save sentiment summary
    summary_path = "data/processed/sentiment_analysis_summary.csv"
    sentiment_summary = (
        sentiment_df.groupby(["stock", "sentiment"])
        .agg({"vader_compound": ["count", "mean", "std"], "tb_polarity": "mean"})
        .round(3)
    )
    sentiment_summary.to_csv(summary_path)

    print(f"✅ Saved sentiment summary to: {summary_path}")

    return final_df, daily_aggregated


def main():
    """Main execution function"""
    try:
        # 1. Load aligned data
        aligned_data = load_aligned_data()

        # 2. Perform sentiment analysis
        sentiment_results = perform_comprehensive_sentiment_analysis(aligned_data)

        # 3. Analyze distribution
        analyzed_sentiment = analyze_sentiment_distribution(sentiment_results)

        # 4. Save results with aggregation
        final_data, daily_aggregated = save_sentiment_results(
            analyzed_sentiment, aligned_data
        )

        print(f"\n🎉 SENTIMENT ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"   📝 Analyzed {len(final_data)} news headlines")
        print(
            f"   📊 Created {len(daily_aggregated)} daily aggregated sentiment records"
        )
        print(f"   🎯 Ready for correlation analysis with aggregated sentiments")
        print(f"   🔍 Use 'daily_aggregated_sentiments.csv' for correlation analysis")

    except Exception as e:
        print(f"\n❌ ERROR in sentiment analysis: {e}")


if __name__ == "__main__":
    main()
