# visualize_indicators.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

print("=== TECHNICAL INDICATORS VISUALIZATION ===")

# Load the data
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
data_file = os.path.join(project_root, "data", "processed", "technical_indicators.csv")

df = pd.read_csv(data_file)
df["Date"] = pd.to_datetime(df["Date"])

print(f"Data loaded: {len(df)} rows")

# Create visualizations
plt.style.use("seaborn-v0_8")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Technical Indicators Analysis", fontsize=16, fontweight="bold")

# 1. Price with Moving Averages
axes[0, 0].plot(
    df["Date"], df["Close"], label="Close Price", linewidth=2, color="black"
)
axes[0, 0].plot(df["Date"], df["MA_20"], label="20-day MA", linewidth=1.5, alpha=0.8)
axes[0, 0].plot(df["Date"], df["MA_50"], label="50-day MA", linewidth=1.5, alpha=0.8)
axes[0, 0].set_title("Price with Moving Averages")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis="x", rotation=45)

# 2. RSI
axes[0, 1].plot(df["Date"], df["RSI"], label="RSI", linewidth=2, color="purple")
axes[0, 1].axhline(
    y=70, color="red", linestyle="--", alpha=0.7, label="Overbought (70)"
)
axes[0, 1].axhline(
    y=30, color="green", linestyle="--", alpha=0.7, label="Oversold (30)"
)
axes[0, 1].fill_between(df["Date"], 70, 100, alpha=0.2, color="red")
axes[0, 1].fill_between(df["Date"], 0, 30, alpha=0.2, color="green")
axes[0, 1].set_title("Relative Strength Index (RSI)")
axes[0, 1].legend()
axes[0, 1].set_ylim(0, 100)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].tick_params(axis="x", rotation=45)

# 3. MACD
axes[1, 0].plot(df["Date"], df["MACD"], label="MACD", linewidth=2, color="blue")
axes[1, 0].plot(
    df["Date"], df["MACD_Signal"], label="Signal Line", linewidth=2, color="red"
)
macd_histogram = df["MACD"] - df["MACD_Signal"]
axes[1, 0].bar(
    df["Date"], macd_histogram, alpha=0.3, color="gray", label="MACD Histogram"
)
axes[1, 0].set_title("MACD Indicator")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].tick_params(axis="x", rotation=45)

# 4. Bollinger Bands
axes[1, 1].plot(
    df["Date"], df["Close"], label="Close Price", linewidth=2, color="black"
)
axes[1, 1].plot(
    df["Date"], df["BB_Upper"], label="Upper Band", linewidth=1, color="red"
)
axes[1, 1].plot(
    df["Date"], df["BB_Middle"], label="Middle Band", linewidth=1, color="orange"
)
axes[1, 1].plot(
    df["Date"], df["BB_Lower"], label="Lower Band", linewidth=1, color="green"
)
axes[1, 1].fill_between(
    df["Date"], df["BB_Upper"], df["BB_Lower"], alpha=0.2, color="gray"
)
axes[1, 1].set_title("Bollinger Bands")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(
    os.path.join(project_root, "data", "processed", "technical_analysis_dashboard.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()

print("✓ Visualization completed and saved!")
