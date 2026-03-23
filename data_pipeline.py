"""
data_pipeline.py
================
Dataset download and preprocessing for the Sui et al. (2024) ensemble stock prediction paper.

Uses yfinance for data sourcing (mirrors the JPX dataset structure: OHLCV daily bars).
Handles:
  - Missing value imputation (forward fill, as per §3.6.1)
  - Outlier capping via IQR (§3.6.2)
  - Feature engineering: Return, MA, EMA, Volatility, RSI, Bollinger Bands (§3.6.3)
  - Train/val/test split
  - Sequence construction for LSTM/GRU inputs
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "JPM",  "GS",   "MS",    "BAC",  "C",
    "TSM",  "SONY", "TM",    "9984.T","7203.T",  # Add JP tickers
]

START_DATE   = "2017-01-04"
END_DATE     = "2021-12-03"   # matches paper dataset window
SEQUENCE_LEN = 20             # lookback window for LSTM/GRU
TEST_RATIO   = 0.15
VAL_RATIO    = 0.15
MA_WINDOWS   = [5, 10, 20]
EMA_WINDOWS  = [5, 10, 20]
VOL_WINDOWS  = [5, 10]
IQR_MULTIPLIER = 3.0          # cap outliers beyond 3×IQR from Q1/Q3


# ─── Data Download ────────────────────────────────────────────────────────────

def download_data(tickers=DEFAULT_TICKERS, start=START_DATE, end=END_DATE,
                  cache_dir="./data_cache") -> pd.DataFrame:
    """Download OHLCV data via yfinance and cache locally as parquet."""
    os.makedirs(cache_dir, exist_ok=True)
    frames = []
    for ticker in tickers:
        cache_path = os.path.join(cache_dir, f"{ticker.replace('.','-')}.parquet")
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            print(f"  [cache] {ticker}: {len(df)} rows")
        else:
            print(f"  [download] {ticker} ...", end="", flush=True)
            try:
                df = yf.download(ticker, start=start, end=end,
                                 auto_adjust=True, progress=False)
                if df.empty:
                    print(" NO DATA, skipping")
                    continue
                df.columns = [c.lower() for c in df.columns]
                df["ticker"] = ticker
                df.to_parquet(cache_path)
                print(f" {len(df)} rows")
            except Exception as e:
                print(f" ERROR: {e}")
                continue
        df["ticker"] = ticker
        frames.append(df)

    if not frames:
        raise RuntimeError("No data downloaded. Check tickers / network.")

    combined = pd.concat(frames)
    combined.index = pd.to_datetime(combined.index)
    combined = combined.sort_index()
    return combined


# ─── Preprocessing ────────────────────────────────────────────────────────────

def _cap_outliers_iqr(series: pd.Series, multiplier: float = IQR_MULTIPLIER) -> pd.Series:
    """Cap values outside [Q1 - k*IQR, Q3 + k*IQR] (§3.6.2)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    return series.clip(lower, upper)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators described in §3.6.3 for a single-ticker df
    with columns [open, high, low, close, volume].
    """
    df = df.copy()
    close = df["close"]

    # § Return
    df["return"] = close.pct_change()

    # § Moving Averages
    for w in MA_WINDOWS:
        df[f"ma_{w}"] = close.rolling(w).mean()
        df[f"ma_ret_{w}"] = df["return"].rolling(w).mean()

    # § Exponential Moving Average (α = 2/(N+1))
    for w in EMA_WINDOWS:
        df[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()

    # § Volatility (rolling std of returns as proxy for rolling price std / MA)
    for w in VOL_WINDOWS:
        df[f"vol_{w}"] = close.rolling(w).std()

    # § RSI (14-period, from ta library)
    rsi = RSIIndicator(close=close, window=14)
    df["rsi"] = rsi.rsi()

    # § Bollinger Bands
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"]  = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["bb_pct"]   = bb.bollinger_pband()

    # § Log volume
    df["log_volume"] = np.log1p(df["volume"])

    # § Price ratios
    df["high_low_ratio"] = df["high"] / df["low"]
    df["close_open_ratio"] = df["close"] / df["open"]

    # Target: next-day return (what we predict)
    df["target_return"] = df["return"].shift(-1)

    return df


def preprocess_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing values, cap outliers, engineer features."""
    df = df.ffill()                              # §3.6.1 forward fill
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if col in df.columns:
            df[col] = _cap_outliers_iqr(df[col])  # §3.6.2 IQR cap
    df = engineer_features(df)
    df = df.dropna()
    return df


def build_dataset(tickers=DEFAULT_TICKERS, start=START_DATE, end=END_DATE,
                  cache_dir="./data_cache") -> dict:
    """
    Full pipeline: download → preprocess → scale → return feature/label arrays.

    Returns
    -------
    dict with keys:
        X_train_seq, X_val_seq, X_test_seq   : (N, SEQ_LEN, F) for LSTM/GRU
        X_train_flat, X_val_flat, X_test_flat : (N, F)           for LightGBM / LR
        y_train, y_val, y_test               : (N,)
        feature_names                         : list[str]
        scaler                               : fitted StandardScaler
        raw_df                               : full processed DataFrame
    """
    print("=== Downloading data ===")
    raw = download_data(tickers, start, end, cache_dir)

    print("\n=== Preprocessing per-ticker ===")
    processed_frames = []
    for ticker, grp in raw.groupby("ticker"):
        try:
            processed = preprocess_ticker(grp)
            processed["ticker"] = ticker
            processed_frames.append(processed)
            print(f"  {ticker}: {len(processed)} rows after preprocessing")
        except Exception as e:
            print(f"  {ticker}: SKIPPED ({e})")

    full_df = pd.concat(processed_frames).sort_index()

    # Feature columns (drop raw price cols & metadata)
    exclude = {"open", "high", "low", "close", "volume", "ticker", "target_return"}
    feature_cols = [c for c in full_df.columns if c not in exclude]
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

    X = full_df[feature_cols].values.astype(np.float32)
    y = full_df["target_return"].values.astype(np.float32)

    # Chronological split (no shuffle — time series)
    n = len(X)
    n_test = int(n * TEST_RATIO)
    n_val  = int(n * VAL_RATIO)
    n_train = n - n_test - n_val

    X_train_flat = X[:n_train]
    X_val_flat   = X[n_train:n_train + n_val]
    X_test_flat  = X[n_train + n_val:]

    y_train = y[:n_train]
    y_val   = y[n_train:n_train + n_val]
    y_test  = y[n_train + n_val:]

    # Scale on train, apply to val/test
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat   = scaler.transform(X_val_flat)
    X_test_flat  = scaler.transform(X_test_flat)

    # Build sequences for LSTM/GRU
    def make_sequences(X_arr, y_arr, seq_len=SEQUENCE_LEN):
        Xs, ys = [], []
        for i in range(seq_len, len(X_arr)):
            Xs.append(X_arr[i - seq_len:i])
            ys.append(y_arr[i])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    X_tr_seq, y_tr_seq = make_sequences(X_train_flat, y_train)
    X_vl_seq, y_vl_seq = make_sequences(X_val_flat,   y_val)
    X_te_seq, y_te_seq = make_sequences(X_test_flat,  y_test)

    print(f"\n=== Dataset shapes ===")
    print(f"  Train seq : {X_tr_seq.shape}  y={y_tr_seq.shape}")
    print(f"  Val   seq : {X_vl_seq.shape}  y={y_vl_seq.shape}")
    print(f"  Test  seq : {X_te_seq.shape}  y={y_te_seq.shape}")
    print(f"  Train flat: {X_train_flat.shape}")

    return {
        "X_train_seq": X_tr_seq,   "y_train_seq": y_tr_seq,
        "X_val_seq":   X_vl_seq,   "y_val_seq":   y_vl_seq,
        "X_test_seq":  X_te_seq,   "y_test_seq":  y_te_seq,
        "X_train_flat": X_train_flat, "y_train": y_train,
        "X_val_flat":   X_val_flat,   "y_val":   y_val,
        "X_test_flat":  X_test_flat,  "y_test":  y_test,
        "feature_names": feature_cols,
        "scaler": scaler,
        "raw_df": full_df,
    }
