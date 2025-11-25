import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
import os

print("📊 TASK 3 - COMPREHENSIVE VISUALIZATION FOR FINAL REPORT")
print("=" * 60)


def load_task3_data():
    """Load all Task 3 data for visualization"""
    print("📥 Loading Task 3 data...")

    datasets = {}

    # 1. Load sentiment analysis results
    sentiment_path = "data/processed/news_with_sentiment.csv"
    if os.path.exists(sentiment_path):
        datasets["sentiment"] = pd.read_csv(sentiment_path)
        print(f"✅ Sentiment data: {len(datasets['sentiment'])} records")

    # 2. Load correlation results
    correlation_path = "data/processed/task3_final_analysis.csv"
    if os.path.exists(correlation_path):
        datasets["correlation"] = pd.read_csv(correlation_path)
        print(f"✅ Correlation data: {len(datasets['correlation'])} records")

    # 3. Load aggregated sentiments
    aggregated_path = "data/processed/daily_aggregated_sentiments.csv"
    if os.path.exists(aggregated_path):
        datasets["aggregated"] = pd.read_csv(aggregated_path)
        print(f"✅ Aggregated data: {len(datasets['aggregated'])} daily records")

    return datasets


def create_date_alignment_visualization(df):
    """Visualization for date alignment process"""
    print("\n📅 Creating date alignment visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Task 3 - Date Alignment & Normalization", fontsize=16, fontweight="bold"
    )

    # 1. News distribution by date
    ax1 = axes[0]
    date_counts = df["trading_date"].value_counts().sort_index()
    ax1.bar(range(len(date_counts)), date_counts.values, color="skyblue", alpha=0.7)
    ax1.set_title("News Articles Distribution by Trading Date")
    ax1.set_xlabel("Trading Days")
    ax1.set_ylabel("Number of Articles")
    ax1.set_xticks(range(len(date_counts)))
    ax1.set_xticklabels([d.split("-")[-1] for d in date_counts.index], rotation=45)
    ax1.grid(True, alpha=0.3)

    # 2. Articles by stock
    ax2 = axes[1]
    stock_counts = df["stock"].value_counts()
    colors = ["#FF9999", "#66B2FF", "#99FF99"]
    bars = ax2.bar(stock_counts.index, stock_counts.values, color=colors, alpha=0.7)
    ax2.set_title("News Articles Distribution by Stock")
    ax2.set_ylabel("Number of Articles")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("data/processed/task3_date_alignment.png", dpi=300, bbox_inches="tight")
    print("✅ Saved date alignment visualization")

    return fig


def create_sentiment_analysis_visualization(df):
    """Comprehensive sentiment analysis visualization"""
    print("\n🎭 Creating sentiment analysis visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Task 3 - Sentiment Analysis Results", fontsize=16, fontweight="bold")

    # 1. Sentiment distribution pie chart
    ax1 = axes[0, 0]
    sentiment_counts = df["sentiment"].value_counts()
    colors = ["#FF9999", "#FFD700", "#99FF99"]  # Red, Yellow, Green
    wedges, texts, autotexts = ax1.pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    ax1.set_title("Overall Sentiment Distribution", fontweight="bold")

    # 2. Sentiment by stock
    ax2 = axes[0, 1]
    sentiment_by_stock = df.groupby(["stock", "sentiment"]).size().unstack()
    sentiment_by_stock.plot(kind="bar", ax=ax2, color=colors)
    ax2.set_title("Sentiment Distribution by Stock")
    ax2.set_ylabel("Number of Articles")
    ax2.legend(title="Sentiment")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)

    # 3. Sentiment scores distribution
    ax3 = axes[1, 0]
    df["vader_compound"].hist(
        bins=15, ax=ax3, color="lightblue", alpha=0.7, edgecolor="black"
    )
    ax3.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Neutral Threshold")
    ax3.set_title("Distribution of VADER Sentiment Scores")
    ax3.set_xlabel("VADER Compound Score")
    ax3.set_ylabel("Frequency")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Strong sentiment examples
    ax4 = axes[1, 1]
    strong_positive = df[df["vader_compound"] > 0.5]
    strong_negative = df[df["vader_compound"] < -0.5]

    categories = ["Strong Positive", "Strong Negative"]
    counts = [len(strong_positive), len(strong_negative)]
    colors = ["#99FF99", "#FF9999"]

    bars = ax4.bar(categories, counts, color=colors, alpha=0.7, edgecolor="black")
    ax4.set_title("Strong Sentiment Articles Analysis")
    ax4.set_ylabel("Number of Articles")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        "data/processed/task3_sentiment_analysis.png", dpi=300, bbox_inches="tight"
    )
    print("✅ Saved sentiment analysis visualization")

    return fig


def create_returns_analysis_visualization(df):
    """Stock returns analysis visualization"""
    print("\n📈 Creating returns analysis visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Task 3 - Stock Returns Analysis", fontsize=16, fontweight="bold")

    # 1. Returns distribution
    ax1 = axes[0]
    df["daily_return_pct"].hist(
        bins=12, ax=ax1, color="lightgreen", alpha=0.7, edgecolor="black"
    )
    ax1.axvline(
        x=df["daily_return_pct"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {df['daily_return_pct'].mean():.2f}%",
    )
    ax1.set_title("Distribution of Daily Stock Returns")
    ax1.set_xlabel("Daily Return (%)")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Returns by stock
    ax2 = axes[1]
    returns_by_stock = df.groupby("stock")["daily_return_pct"].agg(["mean", "std"])
    colors = ["#FF9999", "#66B2FF", "#99FF99"]
    bars = ax2.bar(
        returns_by_stock.index,
        returns_by_stock["mean"],
        yerr=returns_by_stock["std"],
        capsize=5,
        color=colors,
        alpha=0.7,
    )
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax2.set_title("Average Daily Returns by Stock")
    ax2.set_ylabel("Average Return (%)")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}%",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        "data/processed/task3_returns_analysis.png", dpi=300, bbox_inches="tight"
    )
    print("✅ Saved returns analysis visualization")

    return fig


def create_correlation_visualization(df):
    """Main correlation analysis visualization"""
    print("\n🔗 Creating correlation analysis visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Task 3 - Sentiment vs Stock Returns Correlation",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Main correlation scatter plot
    ax1 = axes[0, 0]

    # Color by sentiment
    colors = {"Positive": "green", "Negative": "red", "Neutral": "gray"}
    for sentiment, color in colors.items():
        sentiment_data = df[df["sentiment"] == sentiment]
        ax1.scatter(
            sentiment_data["vader_compound"],
            sentiment_data["daily_return_pct"],
            c=color,
            label=sentiment,
            alpha=0.7,
            s=80,
            edgecolors="black",
            linewidth=0.5,
        )

    # Correlation line
    z = np.polyfit(df["vader_compound"], df["daily_return_pct"], 1)
    p = np.poly1d(z)
    ax1.plot(
        df["vader_compound"], p(df["vader_compound"]), "k-", alpha=0.8, linewidth=2
    )

    # Reference lines
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax1.axvline(x=0, color="black", linestyle="-", alpha=0.3)

    # Correlation stats
    corr, p_val = pearsonr(df["vader_compound"], df["daily_return_pct"])
    ax1.text(
        0.05,
        0.95,
        f"Pearson r = {corr:.3f}\np-value = {p_val:.3f}",
        transform=ax1.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    )

    ax1.set_xlabel("VADER Sentiment Score")
    ax1.set_ylabel("Daily Return (%)")
    ax1.set_title("Sentiment vs Stock Returns Correlation")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Returns by sentiment category
    ax2 = axes[0, 1]
    sentiment_performance = df.groupby("sentiment")["daily_return_pct"].agg(
        ["mean", "std", "count"]
    )
    colors = ["red", "gray", "green"]
    bars = ax2.bar(
        sentiment_performance.index,
        sentiment_performance["mean"],
        yerr=sentiment_performance["std"],
        capsize=5,
        color=colors,
        alpha=0.7,
    )

    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax2.set_title("Average Returns by Sentiment Category")
    ax2.set_ylabel("Average Daily Return (%)")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}%",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontweight="bold",
        )

    # 3. Correlation by stock
    ax3 = axes[1, 0]
    correlations = []
    for stock in df["stock"].unique():
        stock_data = df[df["stock"] == stock]
        if len(stock_data) > 2:
            corr, p_val = pearsonr(
                stock_data["vader_compound"], stock_data["daily_return_pct"]
            )
            correlations.append({"Stock": stock, "Correlation": corr, "P-Value": p_val})

    corr_df = pd.DataFrame(correlations)
    colors = ["green" if x > 0 else "red" for x in corr_df["Correlation"]]
    bars = ax3.bar(corr_df["Stock"], corr_df["Correlation"], color=colors, alpha=0.7)

    ax3.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax3.set_title("Correlation Strength by Stock")
    ax3.set_ylabel("Pearson Correlation Coefficient")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontweight="bold",
        )

    # 4. Strong sentiment impact
    ax4 = axes[1, 1]
    strong_positive = df[df["vader_compound"] > 0.5]
    strong_negative = df[df["vader_compound"] < -0.5]

    categories = ["Strong Positive", "Strong Negative"]
    avg_returns = [
        strong_positive["daily_return_pct"].mean(),
        strong_negative["daily_return_pct"].mean(),
    ]
    colors = ["green", "red"]

    bars = ax4.bar(categories, avg_returns, color=colors, alpha=0.7)
    ax4.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax4.set_title("Strong Sentiment Impact on Returns")
    ax4.set_ylabel("Average Return (%)")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}%",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        "data/processed/task3_correlation_analysis.png", dpi=300, bbox_inches="tight"
    )
    print("✅ Saved correlation analysis visualization")

    return fig


def main():
    """Main function to create all Task 3 visualizations"""
    try:
        print("🚀 Creating comprehensive Task 3 visualizations for final report...")

        # Load data
        datasets = load_task3_data()

        if "sentiment" in datasets:
            # Create all visualizations
            create_date_alignment_visualization(datasets["sentiment"])
            create_sentiment_analysis_visualization(datasets["sentiment"])

        if "correlation" in datasets:
            create_returns_analysis_visualization(datasets["correlation"])
            create_correlation_visualization(datasets["correlation"])

        print(f"\n🎉 TASK 3 VISUALIZATIONS COMPLETED!")
        print(f"   📊 4 comprehensive plots created for your final report")
        print(f"   🎨 Ready to include in your 10-page Medium-style blog post")
        print(f"   📈 Visualizations cover all Task 3 aspects:")
        print(f"      - Date alignment & normalization")
        print(f"      - Sentiment analysis distribution")
        print(f"      - Stock returns analysis")
        print(f"      - Correlation analysis")

    except Exception as e:
        print(f"\n❌ ERROR in visualization: {e}")


if __name__ == "__main__":
    main()
