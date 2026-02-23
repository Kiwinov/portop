import pandas as pd
from pathlib import Path

# ========= CONFIG =========
INPUT_DIR = Path("company_news_sentiment")
OUTPUT_DIR = Path("company_news_sentiment/sentiment_scores")

# create output directory if missing
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def reduce_sentiment_file(csv_path: Path) -> pd.DataFrame:
    """
    Reduce raw news sentiment data into weekly average sentiment.
    Output columns:
        week_start, avg_sentiment
    """

    df = pd.read_csv(csv_path)

    # ensure datetime
    df["week_start"] = pd.to_datetime(df["week_start"])

    # group and average
    reduced = (
        df.groupby("week_start", as_index=False)["sentiment_score"]
        .mean()
        .rename(columns={"sentiment_score": "avg_sentiment"})
        .sort_values("week_start")
    )

    return reduced


def main():
    files = list(INPUT_DIR.glob("*.csv"))

    if not files:
        print("No sentiment files found.")
        return

    for file in files:
        try:
            reduced_df = reduce_sentiment_file(file)

            out_path = OUTPUT_DIR / file.name
            reduced_df.to_csv(out_path, index=False)

            print(f"Saved reduced sentiment -> {out_path}")

        except Exception as e:
            print(f"Failed processing {file.name}: {e}")


if __name__ == "__main__":
    main()
