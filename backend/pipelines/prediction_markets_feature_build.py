"""
Prediction Markets Pipeline - Step 2: Build features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT_DIR / "backend" / "data" / "raw" / "prediction_markets_raw.csv"
DEFAULT_OUTPUT = ROOT_DIR / "backend" / "data" / "features" / "prediction_markets_features.csv"


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "timestamp",
        "market_id",
        "market_slug",
        "question",
        "source",
        "implied_prob",
        "trade_count",
        "notional_volume",
    }
    missing = required - set(raw_df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    raw_df = raw_df.copy()
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    raw_df = raw_df.sort_values(["market_id", "timestamp"]).reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    for market_id, g in raw_df.groupby("market_id", sort=False):
        w = g.copy()
        w["implied_prob"] = pd.to_numeric(w["implied_prob"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
        w["trade_count"] = pd.to_numeric(w["trade_count"], errors="coerce").fillna(0.0)
        w["notional_volume"] = pd.to_numeric(w["notional_volume"], errors="coerce").fillna(0.0)

        w["prob_change_1d"] = w["implied_prob"].diff()
        w["prob_momentum_7d"] = w["implied_prob"] - w["implied_prob"].shift(7)
        w["prob_vol_7d"] = w["prob_change_1d"].rolling(7).std()
        w["prob_vol_30d"] = w["prob_change_1d"].rolling(30).std()
        w["prob_mean_30d"] = w["implied_prob"].rolling(30).mean()
        w["prob_zscore_30d"] = (
            (w["implied_prob"] - w["prob_mean_30d"])
            / (w["implied_prob"].rolling(30).std() + 1e-9)
        )
        w["liquidity_7d"] = w["notional_volume"].rolling(7).mean()
        w["activity_7d"] = w["trade_count"].rolling(7).mean()

        w["target_next_prob_change_1d"] = w["prob_change_1d"].shift(-1)
        w["target_next_direction"] = np.where(w["target_next_prob_change_1d"] > 0, 1, 0)

        w["regime_vol"] = np.where(
            w["prob_vol_30d"] >= w["prob_vol_30d"].quantile(0.75), "high_vol", "normal_vol"
        )
        w["regime_prob_level"] = np.select(
            [w["implied_prob"] <= 0.2, w["implied_prob"] >= 0.8],
            ["low_prob_tail", "high_prob_tail"],
            default="mid_range",
        )

        frames.append(w)
        print(f"Built features for market_id={market_id}: {len(w)} rows")

    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prediction-market features")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input raw CSV path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output feature CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Raw input file not found: {input_path}")

    raw_df = pd.read_csv(input_path)
    feat_df = build_features(raw_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(output_path, index=False)
    print(f"Wrote feature dataset to {output_path} ({len(feat_df)} rows)")


if __name__ == "__main__":
    main()
