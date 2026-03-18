"""
Step 2: Build technical and regime features from normalized raw price data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT_DIR / "backend" / "data" / "raw" / "spx_raw.csv"
DEFAULT_OUTPUT = ROOT_DIR / "backend" / "data" / "features" / "spx_features.csv"


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_macd(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window).mean()


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    raw_df = raw_df.copy()
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    raw_df = raw_df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    feature_frames: list[pd.DataFrame] = []
    for symbol, group in raw_df.groupby("symbol", sort=False):
        g = group.copy()
        g["return_1d"] = g["close"].pct_change()
        g["log_return_1d"] = np.log(g["close"] / g["close"].shift(1))
        g["vol_20d"] = g["return_1d"].rolling(20).std() * np.sqrt(252)
        g["sma_20"] = g["close"].rolling(20).mean()
        g["sma_50"] = g["close"].rolling(50).mean()
        g["rsi_14"] = add_rsi(g, window=14)

        macd_line, signal_line, macd_hist = add_macd(g)
        g["macd_line"] = macd_line
        g["macd_signal"] = signal_line
        g["macd_hist"] = macd_hist
        g["atr_14"] = add_atr(g, window=14)

        g["rolling_max_252"] = g["close"].rolling(252, min_periods=30).max()
        g["drawdown"] = (g["close"] / g["rolling_max_252"]) - 1.0
        g["target_next_return_1d"] = g["return_1d"].shift(-1)
        g["target_next_direction"] = np.where(g["target_next_return_1d"] > 0, 1, 0)

        g["trend_regime"] = np.where(g["sma_20"] > g["sma_50"], "uptrend", "downtrend")
        g["momentum_regime"] = np.select(
            [g["rsi_14"] >= 70, g["rsi_14"] <= 30],
            ["overbought", "oversold"],
            default="neutral",
        )
        vol_p75 = g["vol_20d"].quantile(0.75)
        g["vol_regime"] = np.where(g["vol_20d"] >= vol_p75, "high_vol", "normal_vol")

        feature_frames.append(g)
        print(f"Built features for {symbol}: {len(g)} rows")

    feature_df = pd.concat(feature_frames, ignore_index=True)
    feature_df["timestamp"] = feature_df["timestamp"].dt.strftime("%Y-%m-%d")
    return feature_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SPX technical features")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input raw CSV path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output features CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Raw input file not found: {input_path}")

    raw_df = pd.read_csv(input_path)
    required_cols = {"timestamp", "open", "high", "low", "close", "volume", "symbol"}
    missing = required_cols - set(raw_df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in raw data: {sorted(missing)}")

    feature_df = build_features(raw_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    print(f"Wrote feature dataset to {output_path} ({len(feature_df)} rows)")


if __name__ == "__main__":
    main()
