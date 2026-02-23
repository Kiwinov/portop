import pandas as pd
from pathlib import Path

# ========= CONFIG =========
FEATURES_DIR = Path("features")
SENTIMENT_DIR = Path("company_news_sentiment/sentiment_scores")


def load_and_shift_sentiment(sentiment_path: Path) -> pd.DataFrame:
    """
    Load reduced sentiment and shift forward by one week.
    """

    sent = pd.read_csv(sentiment_path)

    sent["week_start"] = pd.to_datetime(sent["week_start"])

    # Shift sentiment forward: assign previous week sentiment
    sent["avg_sentiment"] = sent["avg_sentiment"].shift(1)

    return sent


def merge_sentiment(features_path, sentiment_df):

    feat = pd.read_csv(features_path)
    feat["Date"] = pd.to_datetime(feat["Date"])

    # IMPORTANT: prevent duplicate columns
    feat = feat.drop(
        columns=[c for c in feat.columns if "avg_sentiment" in c], errors="ignore"
    )

    # map to previous Friday
    feat["week_start"] = feat["Date"] - pd.to_timedelta(
        (feat["Date"].dt.weekday - 4) % 7, unit="D"
    )

    merged = feat.merge(sentiment_df, on="week_start", how="left")
    merged.drop(columns=["week_start"], inplace=True)

    return merged


def main():

    feature_files = list(FEATURES_DIR.glob("*.csv"))

    for feat_file in feature_files:

        sentiment_file = SENTIMENT_DIR / feat_file.name

        if not sentiment_file.exists():
            print(f"Skipping {feat_file.name} (no sentiment file)")
            continue

        try:
            sentiment_df = load_and_shift_sentiment(sentiment_file)
            merged_df = merge_sentiment(feat_file, sentiment_df)

            # overwrite features file
            merged_df.to_csv(feat_file, index=False)

            print(f"Merged sentiment into {feat_file.name}")

        except Exception as e:
            print(f"Failed {feat_file.name}: {e}")


if __name__ == "__main__":
    main()
