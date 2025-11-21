# scripts/validate_cleaned_data.py
import pandas as pd
import numpy as np


def validate_cleaned_data():
    """Comprehensive validation of cleaned data"""
    print("🔍 VALIDATING CLEANED DATA")

    # Load cleaned data
    df = pd.read_csv("data/processed/financial_data_cleaned.csv")
    print(f"Loaded cleaned data: {df.shape}\n")

    # 1. Basic Structure Validation
    print("=== BASIC STRUCTURE VALIDATION ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}\n")

    # 2. Missing Values Check
    print("=== MISSING VALUES CHECK ===")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        print("✅ No missing values found")
    else:
        print("❌ Missing values detected:")
        for col, count in missing_values[missing_values > 0].items():
            print(f"   {col}: {count} missing")
    print()

    # 3. Duplicates Check
    print("=== DUPLICATES CHECK ===")
    duplicate_rows = df.duplicated().sum()
    duplicate_headlines = df.duplicated(subset=["headline"]).sum()

    if duplicate_rows == 0:
        print("✅ No duplicate rows found")
    else:
        print(f"❌ {duplicate_rows} duplicate rows found")

    if duplicate_headlines == 0:
        print("✅ No duplicate headlines found")
    else:
        print(f"❌ {duplicate_headlines} duplicate headlines found")
    print()

    # 4. Data Quality Checks
    print("=== DATA QUALITY CHECKS ===")

    # Date validation
    date_cols = [col for col in df.columns if "date" in col.lower()]
    for col in date_cols:
        if col in df.columns:
            print(f"📅 {col}: {df[col].min()} to {df[col].max()}")

    # Text quality checks
    if "headline" in df.columns:
        empty_headlines = (df["headline"].str.strip() == "").sum()
        short_headlines = (df["headline"].str.len() < 5).sum()
        print(f"📰 Headlines - Empty: {empty_headlines}, Too short: {short_headlines}")

    # Numeric range checks for new features
    if "headline_length" in df.columns:
        invalid_lengths = (
            (df["headline_length"] <= 0) | (df["headline_length"] > 1000)
        ).sum()
        print(f"📏 Headline length - Invalid: {invalid_lengths}")

    if "headline_word_count" in df.columns:
        invalid_word_counts = (
            (df["headline_word_count"] <= 0) | (df["headline_word_count"] > 50)
        ).sum()
        print(f"📝 Word count - Invalid: {invalid_word_counts}")
    print()

    # 5. Statistical Summary
    print("=== STATISTICAL SUMMARY ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print("Numeric columns summary:")
        print(df[numeric_cols].describe())
    print()

    # 6. Categorical Data Check
    print("=== CATEGORICAL DATA DISTRIBUTION ===")
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if col != "headline":  # Skip headline for brevity
            print(f"\n{col} value counts (top 10):")
            print(df[col].value_counts().head(10))

    # 7. New Features Validation
    print("\n=== NEW FEATURES VALIDATION ===")
    new_features = [
        "headline_length",
        "headline_word_count",
        "year",
        "month",
        "day_of_week",
    ]
    for feature in new_features:
        if feature in df.columns:
            print(f"{feature}:")
            print(f"  Range: {df[feature].min()} - {df[feature].max()}")
            print(f"  Unique values: {df[feature].nunique()}")

    return df


def sample_data_preview(df, n_samples=5):
    """Show sample data for manual inspection"""
    print("\n=== SAMPLE DATA PREVIEW ===")
    print("First 5 rows:")
    print(df.head(n_samples).to_string())

    print("\nLast 5 rows:")
    print(df.tail(n_samples).to_string())


def check_analysis_readiness(df):
    """Final checklist for analysis readiness"""
    print("\n" + "=" * 60)
    print("📊 ANALYSIS READINESS CHECKLIST")
    print("=" * 60)

    checks = {
        "No missing values": df.isnull().sum().sum() == 0,
        "No duplicate headlines": df.duplicated(subset=["headline"]).sum() == 0,
        "Valid date range": len(df) > 0,
        "Headlines are clean": (df["headline"].str.strip() == "").sum() == 0,
        "New features present": all(
            feat in df.columns
            for feat in ["headline_length", "headline_word_count", "year"]
        ),
    }

    all_passed = True
    for check, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 DATA IS READY FOR ANALYSIS!")
    else:
        print("⚠️  Data needs additional cleaning")

    return all_passed


if __name__ == "__main__":
    df = validate_cleaned_data()
    sample_data_preview(df)
    check_analysis_readiness(df)
