import os
import pandas as pd
import numpy as np
import yfinance as yf
from pandas.tseries.offsets import MonthEnd

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_DIRS = {
    "quarterly": "quarterly",
    "balance": "balance_sheet",
    "cashflow": "cashflow",
}
OUTPUT_DIR = "features"

# Company to Ticker Mapping
COMPANY_MAPPING = {
    "reliance": "RELIANCE.NS",
    "hdfc": "HDFCBANK.NS",
    "infy": "INFY.NS",
    "mandm": "M&M.NS",
    "airtel": "BHARTIARTL.NS",
    "hul": "HINDUNILVR.NS",
}

# Date Range for Stock Price Download
START_DATE = "2019-01-01"
END_DATE = "2025-12-31"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# FUNCTIONS
# ==========================================


def get_stock_prices(ticker, start, end):
    """Downloads and formats stock data from Yahoo Finance."""
    print(f"Downloading data for {ticker}...")
    try:
        raw_data = yf.download(
            ticker,
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
        )

        if raw_data.empty:
            print(f"Warning: No price data found for {ticker}")
            return pd.DataFrame()

        # Handle yfinance MultiIndex columns if present
        if isinstance(raw_data.columns, pd.MultiIndex):
            df_market = (
                raw_data.stack(level=0)
                .reset_index()
                .rename(columns={"Level_1": "Ticker"})
            )
            df_prices = df_market[df_market["Ticker"] == ticker].set_index("Date")
            df_prices.drop(columns=["Ticker"], inplace=True)
        else:
            df_prices = raw_data

        return df_prices
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return pd.DataFrame()


def clean_currency_string(x):
    """
    Converts strings like '" 1,998.70 "' or '264,905.00' to float.
    Handles quotes, commas, and whitespace.
    """
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, str):
        clean_str = x.replace('"', "").replace("'", "").replace(",", "").strip()

        if clean_str in ["--", "", "-"]:
            return np.nan

        try:
            return float(clean_str)
        except ValueError:
            return np.nan
    return np.nan


def extract_and_transpose(filepath, header_row, date_format):
    """
    Reads CSV with specific header row, cleans numbers, duplicates, and transposes.
    """
    try:
        df = pd.read_csv(filepath, index_col=0, header=header_row)

        # Clean index
        df = df[df.index.notna()]
        df = df[df.index.astype(str).str.strip() != ""]
        df = df[~df.index.duplicated(keep="first")]

        # Transpose
        df = df.T

        # Normalize column names
        df.columns = (
            df.columns.astype(str)
            .str.replace("\xa0", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        # Clean numeric strings
        for col in df.columns:
            df[col] = df[col].apply(clean_currency_string)

        # Parse dates
        df.index = pd.to_datetime(df.index, format=date_format, errors="coerce")
        df.index = df.index + MonthEnd(0)
        df = df.sort_index()
        df = df[df.index.notnull()]

        return df
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return pd.DataFrame()


def compile_institutional_features(df_prices, df_quarterly, df_bs, df_cf, ticker):
    """Merges financial data with price data and calculates ratios."""

    if df_prices.empty or df_quarterly.empty or df_bs.empty or df_cf.empty:
        print(f"Missing data components for {ticker}. Skipping feature compilation.")
        return None

    # --- Pre-calculation on Financials ---
    q_calc = df_quarterly.copy()

    # Resolve profit column with fallback priority
    profit_series = None

    candidates = [
        "Net P/L After M.I & Associates",
        "Net Profit/(Loss) For the Period",
        "P/L After Tax from Ordinary Activities",
    ]

    for c in candidates:
        if c in q_calc.columns:
            if profit_series is None:
                profit_series = q_calc[c]
            else:
                profit_series = profit_series.fillna(q_calc[c])

    if profit_series is None:
        print(f"Error: No usable profit column found for {ticker}")
        return None

    q_calc["Profit_Base"] = profit_series

    q_calc["Annualized_Profit"] = q_calc["Profit_Base"].rolling(4, min_periods=4).sum()

    q_calc["Annualized_Sales"] = (
        q_calc["Total Income From Operations"].rolling(4, min_periods=4).sum()
    )

    q_calc["EPS_TTM"] = q_calc["Basic EPS"].rolling(4, min_periods=4).sum()

    # Resolve income column
    if "Total Income From Operations" not in q_calc.columns:
        if "Net Sales/Income from operations" in q_calc.columns:
            q_calc["Total Income From Operations"] = q_calc[
                "Net Sales/Income from operations"
            ]
        else:
            print(f"Error: Income column missing for {ticker}")
            return None

    # Check EPS
    if "Basic EPS" not in q_calc.columns:
        print(f"Error: Basic EPS missing for {ticker}")
        return None

    # Balance Sheet processing
    bs_calc = df_bs.copy()
    if "Total" not in bs_calc.columns:
        print(f"Error: Column 'Total' missing in Balance Sheet for {ticker}")
        return None

    bs_calc["Avg_Assets"] = (bs_calc["Total"] + bs_calc["Total"].shift(1)) / 2

    # --- Merge with Prices ---
    df = df_prices.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Quarterly integration
    q_cols = [
        "Total Income From Operations",
        "Profit_Base",
        "Basic EPS",
        "Annualized_Profit",
        "Annualized_Sales",
        "EPS_TTM",
    ]
    df = df.join(q_calc[q_cols], how="outer")
    df[q_cols] = df[q_cols].ffill()

    # Balance Sheet integration
    bs_cols = ["Equity Share Capital", "Reserves", "Borrowings", "Total", "Avg_Assets"]
    available_bs_cols = [c for c in bs_cols if c in bs_calc.columns]
    df = df.join(bs_calc[available_bs_cols], how="outer")
    df[available_bs_cols] = df[available_bs_cols].ffill()

    # Cash Flow integration
    cf_cols = ["Cash from Operating Activity"]
    if "Cash from Operating Activity" in df_cf.columns:
        df = df.join(df_cf[cf_cols], how="outer")
        df[cf_cols] = df[cf_cols].ffill()

    # Remove rows without price
    df = df[df["Close"].notna()]

    # --- Calculations ---
    try:
        df["EPS_TTM"] = df["EPS_TTM"].replace(0, np.nan)

        # Valuation
        df["Daily_PE"] = df["Close"] / df["EPS_TTM"]

        # Equity calculation
        if "Reserves" in df.columns:
            df["Total_Equity"] = df["Equity Share Capital"] + df["Reserves"].fillna(0)
        else:
            df["Total_Equity"] = df["Equity Share Capital"]

        # DuPont components
        df["Net_Margin"] = df["Profit_Base"] / df["Total Income From Operations"]
        df["Asset_Turnover"] = df["Annualized_Sales"] / df["Avg_Assets"]
        df["Equity_Multiplier"] = df["Total"] / df["Total_Equity"]

        df["DuPont_ROE"] = (
            df["Net_Margin"] * df["Asset_Turnover"] * df["Equity_Multiplier"]
        )

        # Risk & Earnings Quality
        if "Borrowings" in df.columns:
            df["Debt_to_Equity"] = df["Borrowings"] / df["Total_Equity"]
        else:
            df["Debt_to_Equity"] = 0

        if "Cash from Operating Activity" in df.columns:
            df["Earnings_Quality"] = (
                df["Cash from Operating Activity"] / df["Annualized_Profit"]
            )
        else:
            df["Earnings_Quality"] = np.nan

        # Yield
        df["Earnings_Yield"] = (df["EPS_TTM"] / df["Close"]) * 100

        # Clean
        df.dropna(subset=["Daily_PE", "DuPont_ROE"], inplace=True)
        print(f"Feature matrix compiled for {ticker}. Active Rows: {len(df)}")
        return df

    except Exception as e:
        print(f"Calculation Error for {ticker}: {e}")
        return None


import pandas as pd
import numpy as np
import os


# =========================================================
# TECHNICAL FEATURE ENGINEERING
# =========================================================
def feature_engineering_technicals(df):
    feat = df.copy()

    feat["Log_Return_1D"] = np.log(feat["Close"] / feat["Close"].shift(1))
    feat["Log_Return_5D"] = feat["Log_Return_1D"].rolling(5, min_periods=1).sum()
    feat["Log_Return_20D"] = feat["Log_Return_1D"].rolling(20, min_periods=1).sum()
    feat["Log_Return_60D"] = feat["Log_Return_1D"].rolling(60, min_periods=1).sum()

    rolling_max = feat["Close"].rolling(252, min_periods=1).max()
    feat["Drawdown"] = (feat["Close"] / rolling_max) - 1

    feat["SMA_20"] = feat["Close"].rolling(20, min_periods=1).mean()
    feat["EMA_20"] = feat["Close"].ewm(span=20, adjust=False).mean()
    feat["Dist_SMA_20"] = (feat["Close"] / feat["SMA_20"]) - 1
    feat["Dist_EMA_20"] = (feat["Close"] / feat["EMA_20"]) - 1

    delta = feat["Close"].diff()
    gain = delta.where(delta > 0, 0).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    rs = gain / loss
    feat["RSI_14"] = 100 - (100 / (1 + rs))

    ema_12 = feat["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = feat["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    feat["MACD_Histogram"] = (macd_line - macd_signal) / feat["Close"]

    rolling_std = feat["Close"].rolling(20, min_periods=1).std()
    upper_band = feat["SMA_20"] + (rolling_std * 2)
    lower_band = feat["SMA_20"] - (rolling_std * 2)
    feat["BB_Bandwidth"] = (upper_band - lower_band) / feat["SMA_20"]
    feat["BB_PctB"] = (feat["Close"] - lower_band) / (upper_band - lower_band)

    range_hl = (feat["High"] - feat["Low"]).replace(0, np.nan)
    feat["Buying_Pressure"] = (feat["Close"] - feat["Low"]) / range_hl
    feat["Selling_Pressure"] = (feat["High"] - feat["Close"]) / range_hl

    feat["Return_Lag1"] = feat["Log_Return_1D"].shift(1)
    feat["Return_Lag2"] = feat["Log_Return_1D"].shift(2)

    vol_sma_20 = feat["Volume"].rolling(20, min_periods=1).mean()
    feat["Volume_Attention_Today"] = feat["Volume"] / vol_sma_20
    feat["Volume_Attention_Lag1"] = feat["Volume_Attention_Today"].shift(1)

    drop_cols = ["Open", "High", "Low", "Close", "Volume", "SMA_20", "EMA_20"]
    feat.drop(columns=[c for c in drop_cols if c in feat.columns], inplace=True)

    feat.dropna(inplace=True)
    return feat


# =========================================================
# MACRO ENGINEERING
# =========================================================
def engineer_exhaustive_macro(df):
    m = df.copy()

    m["Yield Gap"] = m["Earnings_Yield"] - m["India_10Y_Yield"]

    for col in ["Crude_Oil", "Gold_Price", "USDINR", "SP500", "Nifty50"]:
        m[f"{col}_Ret_1D"] = np.log(m[col] / m[col].shift(1))
        m[f"{col}_Ret_5D"] = m[f"{col}_Ret_1D"].rolling(5, min_periods=1).sum()

    m["Gold_Oil_Ratio"] = m["Gold_Price"] / m["Crude_Oil"]
    m["Global_Equity_Diff"] = m["SP500_Ret_1D"] - m["Nifty50_Ret_1D"]
    m["Domestic_Yield_Slope"] = m["India_10Y_Yield"] - m["Repo_Rate"]
    m["Global_Yield_Spread"] = m["India_10Y_Yield"] - m["Global_Bond_Yield"]
    m["Real_Rate"] = m["India_10Y_Yield"] - m["CPI"]

    m["VIX_Level"] = m["India_VIX"]
    m["VIX_Change"] = m["India_VIX"].diff()
    m["Global_Vol_20D"] = m["SP500_Ret_1D"].rolling(20, min_periods=1).std()

    drop_cols = [
        "Crude_Oil",
        "Gold_Price",
        "USDINR",
        "SP500",
        "Global_Bond_Yield",
        "Nifty50",
        "India_VIX",
        "India_10Y_Yield",
        "Repo_Rate",
        "CPI",
    ]
    m.drop(columns=[c for c in drop_cols if c in m.columns], inplace=True)

    return m


# =========================================================
# FUNDAMENTAL ENGINEERING
# =========================================================
def feature_engineering_fundamentals(df):
    f = df.copy()
    DAYS_PER_YEAR = 252

    def safe_growth(series, window):
        return series / series.shift(window).fillna(series.iloc[0]) - 1

    f["Rev_Growth_YoY"] = safe_growth(f["Annualized_Sales"], DAYS_PER_YEAR)
    f["Profit_Growth_YoY"] = safe_growth(f["Annualized_Profit"], DAYS_PER_YEAR)
    f["EPS_Growth_YoY"] = safe_growth(f["EPS_TTM"], DAYS_PER_YEAR)

    f["Net_Profit_Margin"] = f["Net_Margin"]
    f["Asset_Turnover_Ratio"] = f["Asset_Turnover"]
    f["Equity_Multiplier_Ratio"] = f["Equity_Multiplier"]
    f["DuPont_ROE_Check"] = f["DuPont_ROE"]

    f["Earnings_Quality_Ratio"] = f["Earnings_Quality"]
    f["D_E_Ratio"] = f["Debt_to_Equity"]

    f["PE_to_Avg_1Y"] = (
        f["Daily_PE"]
        / f["Daily_PE"].rolling(window=DAYS_PER_YEAR, min_periods=1).mean()
    )

    raw_accounting = [
        "Total Income From Operations",
        "Profit_Base",
        "Basic EPS",
        "Annualized_Sales",
        "Annualized_Profit",
        "Equity Share Capital",
        "Reserves",
        "Borrowings",
        "Total",
        "Avg_Assets",
        "Cash from Operating Activity",
        "Total_Equity",
        "EPS_TTM",
        "Net_Margin",
        "Asset_Turnover",
        "Equity_Multiplier",
        "DuPont_ROE",
        "Debt_to_Equity",
        "Daily_PE",
        "Earnings_Quality",
        "Earnings_Yield",
    ]

    f.drop(columns=[c for c in raw_accounting if c in f.columns], inplace=True)

    return f


# ==========================================
# MAIN EXECUTION
# ==========================================


def main():
    print("\nLoading macro.csv ...")

    df_macro = pd.read_csv("macro.csv", parse_dates=["Date"])
    df_macro.columns = df_macro.columns.str.strip()
    df_macro.set_index("Date", inplace=True)
    df_macro.sort_index(inplace=True)

    for company_name, ticker in COMPANY_MAPPING.items():
        print(f"\nProcessing {company_name.upper()} ({ticker})")

        filename = f"{company_name}.csv"
        path_q = os.path.join(INPUT_DIRS["quarterly"], filename)
        path_b = os.path.join(INPUT_DIRS["balance"], filename)
        path_c = os.path.join(INPUT_DIRS["cashflow"], filename)

        if not (
            os.path.exists(path_q) and os.path.exists(path_b) and os.path.exists(path_c)
        ):
            print(f"Skipping {company_name}: Missing CSV.")
            continue

        df_prices = get_stock_prices(ticker, START_DATE, END_DATE)

        df_q = extract_and_transpose(path_q, header_row=0, date_format="%b '%y")
        df_b = extract_and_transpose(path_b, header_row=2, date_format="%b-%y")
        df_c = extract_and_transpose(path_c, header_row=2, date_format="%b-%y")

        df_final = compile_institutional_features(df_prices, df_q, df_b, df_c, ticker)

        if df_final is None or df_final.empty:
            print(f"Failed to generate output for {company_name}")

        print("Applying technical features...")
        df_final = feature_engineering_technicals(df_final)

        macro_aligned = df_macro.reindex(df_final.index)
        df_final = df_final.join(macro_aligned, how="left")

        print("Engineering macro factors...")
        df_final = engineer_exhaustive_macro(df_final)

        print("Engineering fundamental features...")
        df_final = feature_engineering_fundamentals(df_final)

        df_final.dropna(inplace=True)

        df_final = df_final.loc[df_final.index >= "2020-01-01"]

        output_path = os.path.join(OUTPUT_DIR, f"{company_name}.csv")
        df_final.to_csv(output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
