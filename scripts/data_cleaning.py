import pandas as pd
import os

print("STARTING DATA CLEANING")

# Load raw data
df = pd.read_csv("data/raw/financial_data_raw.csv")
print("Raw data loaded:", df.shape)

# Basic cleaning
df_clean = df.dropna(subset=["headline"]).copy()
df_clean["publisher"] = df_clean["publisher"].fillna("Unknown")
df_clean["stock"] = df_clean["stock"].fillna("Unknown")

# Clean dates
df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
df_clean = df_clean.dropna(subset=["date"])

# Remove duplicates
df_clean = df_clean.drop_duplicates(["headline"])

# Clean text
df_clean["headline"] = df_clean["headline"].str.strip()

# Add analysis features
df_clean["headline_length"] = df_clean["headline"].str.len()
df_clean["headline_word_count"] = df_clean["headline"].str.split().str.len()
df_clean["date_only"] = df_clean["date"].dt.date
df_clean["year"] = df_clean["date"].dt.year
df_clean["month"] = df_clean["date"].dt.month
df_clean["day_of_week"] = df_clean["date"].dt.day_name()

print("Cleaned & enhanced data:", df_clean.shape)

# Save single analysis-ready file
os.makedirs("data/processed", exist_ok=True)
df_clean.to_csv("data/processed/financial_data_cleaned.csv", index=False)
print("Analysis-ready data saved: data/processed/financial_data_cleaned.csv")

print("DATA CLEANING COMPLETED")
print("Final dataset:", df_clean.shape)
print("Date range:", df_clean["date"].min(), "to", df_clean["date"].max())
