"""
Step 4: Production-style walk-forward evaluation and trading policy for SPX/SPY.

What this script does:
1) Reads engineered features from pipelines/feature_build.py output.
2) Builds a raw next-day up-probability score.
3) Calibrates probabilities in a walk-forward manner (Platt scaling).
4) Applies an explicit trading policy (long / short / flat for SPY or ES).
5) Writes machine-readable metrics and daily decisions for monitoring/backtests.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT_DIR / "backend" / "data" / "features" / "spx_features.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "backend" / "data" / "eval"


EPS = 1e-12
TRADING_DAYS_PER_YEAR = 252


@dataclass
class PolicyConfig:
    long_threshold: float
    short_threshold: float
    confidence_for_es: float
    fee_bps_per_turnover: float


def _safe_series(s: pd.Series, fill: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(fill)


def build_raw_probability(df: pd.DataFrame) -> np.ndarray:
    """
    Build a deterministic base probability from feature mix.
    This is intentionally transparent and stable for production monitoring.
    """
    rsi = _safe_series(df["rsi_14"], 50.0)
    macd_hist = _safe_series(df["macd_hist"], 0.0)
    ret_1d = _safe_series(df["return_1d"], 0.0)
    vol_20d = _safe_series(df["vol_20d"], 0.0)
    drawdown = _safe_series(df["drawdown"], 0.0)
    sma_20 = _safe_series(df["sma_20"], 0.0)
    sma_50 = _safe_series(df["sma_50"], 0.0)

    trend_spread = np.where(np.abs(sma_50) > EPS, (sma_20 - sma_50) / np.abs(sma_50), 0.0)
    z_macd = (macd_hist - np.nanmean(macd_hist)) / (np.nanstd(macd_hist) + EPS)
    z_ret = (ret_1d - np.nanmean(ret_1d)) / (np.nanstd(ret_1d) + EPS)
    z_vol = (vol_20d - np.nanmean(vol_20d)) / (np.nanstd(vol_20d) + EPS)

    # Weighted blend: trend + momentum + mild mean reversion + volatility penalty.
    score = (
        0.70 * np.clip(trend_spread, -0.05, 0.05) / 0.05
        + 0.45 * np.clip((rsi - 50.0) / 20.0, -2.0, 2.0)
        + 0.55 * np.clip(z_macd, -3.0, 3.0)
        - 0.25 * np.clip(z_ret, -3.0, 3.0)
        - 0.20 * np.clip(z_vol, -3.0, 3.0)
        + 0.15 * np.clip(-drawdown / 0.10, -2.0, 2.0)
    )

    return 1.0 / (1.0 + np.exp(-score))


def fit_platt_scaler(p_raw: np.ndarray, y: np.ndarray, steps: int = 250, lr: float = 0.05) -> tuple[float, float]:
    """
    Fit p_calibrated = sigmoid(a * logit(p_raw) + b) via gradient descent.
    """
    p = np.clip(p_raw, 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    y = y.astype(float)
    a = 1.0
    b = 0.0
    n = max(len(x), 1)

    for _ in range(steps):
        z = a * x + b
        p_hat = 1.0 / (1.0 + np.exp(-z))
        err = p_hat - y
        grad_a = float(np.dot(err, x) / n)
        grad_b = float(np.sum(err) / n)
        a -= lr * grad_a
        b -= lr * grad_b

    return a, b


def apply_platt_scaler(p_raw: np.ndarray, a: float, b: float) -> np.ndarray:
    p = np.clip(p_raw, 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    z = a * x + b
    return 1.0 / (1.0 + np.exp(-z))


def log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-8, 1 - 1e-8)
    y = y_true.astype(float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    y = y_true.astype(float)
    return float(np.mean((p - y) ** 2))


def max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve / np.maximum(peak, EPS)) - 1.0
    return float(np.min(dd))


def annualized_sharpe(daily_returns: np.ndarray) -> float:
    std = float(np.std(daily_returns))
    if std < EPS:
        return 0.0
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * np.mean(daily_returns) / std)


def run_walk_forward(
    df: pd.DataFrame,
    min_train_days: int,
    train_window_days: int,
    policy: PolicyConfig,
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["raw_prob_up"] = build_raw_probability(out)
    out["prob_up"] = np.nan

    y_full = _safe_series(out["target_next_direction"], 0).astype(int).to_numpy()
    raw_full = out["raw_prob_up"].to_numpy()
    n = len(out)

    for i in range(min_train_days, n):
        lo = max(0, i - train_window_days)
        train_raw = raw_full[lo:i]
        train_y = y_full[lo:i]
        if len(train_raw) < 64:
            # Cold-start fallback
            out.at[i, "prob_up"] = float(raw_full[i])
            continue
        a, b = fit_platt_scaler(train_raw, train_y)
        out.at[i, "prob_up"] = float(apply_platt_scaler(np.array([raw_full[i]]), a, b)[0])

    out = out.dropna(subset=["prob_up"]).copy()
    out["prob_down"] = 1.0 - out["prob_up"]
    out["confidence"] = (out["prob_up"] - 0.5).abs() * 2.0

    # Trading policy.
    out["position"] = 0
    out.loc[out["prob_up"] >= policy.long_threshold, "position"] = 1
    out.loc[out["prob_up"] <= policy.short_threshold, "position"] = -1

    out["instrument"] = "SPY"
    out.loc[out["confidence"] >= policy.confidence_for_es, "instrument"] = "ES"

    out["next_return"] = _safe_series(out["target_next_return_1d"], 0.0)
    out["turnover"] = out["position"].diff().abs().fillna(out["position"].abs())
    out["fees"] = out["turnover"] * (policy.fee_bps_per_turnover / 10000.0)
    out["strategy_return"] = out["position"] * out["next_return"] - out["fees"]
    out["equity_curve"] = (1.0 + out["strategy_return"]).cumprod()

    return out


def summarize_metrics(eval_df: pd.DataFrame) -> dict:
    y = _safe_series(eval_df["target_next_direction"], 0).astype(int).to_numpy()
    p = _safe_series(eval_df["prob_up"], 0.5).to_numpy()
    pos = _safe_series(eval_df["position"], 0).astype(int).to_numpy()
    r = _safe_series(eval_df["strategy_return"], 0.0).to_numpy()

    pred_dir = (p >= 0.5).astype(int)
    mask_active = pos != 0
    active_count = int(np.sum(mask_active))

    directional_acc = float(np.mean(pred_dir == y))
    active_acc = float(np.mean(pred_dir[mask_active] == y[mask_active])) if active_count else 0.0
    pnl_total = float(np.prod(1.0 + r) - 1.0)
    trade_win_rate = float(np.mean(r[mask_active] > 0)) if active_count else 0.0

    return {
        "rows_evaluated": int(len(eval_df)),
        "coverage_active_positions": float(active_count / max(len(eval_df), 1)),
        "directional_accuracy_all": directional_acc,
        "directional_accuracy_when_active": active_acc,
        "brier_score": brier_score(y, p),
        "log_loss": log_loss(y, p),
        "strategy_total_return": pnl_total,
        "strategy_annualized_sharpe": annualized_sharpe(r),
        "strategy_max_drawdown": max_drawdown(eval_df["equity_curve"].to_numpy()),
        "trade_win_rate_when_active": trade_win_rate,
    }


def build_report_text(metrics: dict, symbol: str, cfg: dict) -> str:
    return "\n".join(
        [
            f"# Production Evaluation Report ({symbol})",
            "",
            "## Walk-forward setup",
            f"- min_train_days: {cfg['min_train_days']}",
            f"- train_window_days: {cfg['train_window_days']}",
            f"- long_threshold: {cfg['long_threshold']:.3f}",
            f"- short_threshold: {cfg['short_threshold']:.3f}",
            f"- confidence_for_es: {cfg['confidence_for_es']:.3f}",
            f"- fee_bps_per_turnover: {cfg['fee_bps_per_turnover']:.2f}",
            "",
            "## Probability quality",
            f"- directional_accuracy_all: {metrics['directional_accuracy_all']:.4f}",
            f"- directional_accuracy_when_active: {metrics['directional_accuracy_when_active']:.4f}",
            f"- brier_score: {metrics['brier_score']:.6f}",
            f"- log_loss: {metrics['log_loss']:.6f}",
            "",
            "## Trading quality",
            f"- rows_evaluated: {metrics['rows_evaluated']}",
            f"- coverage_active_positions: {metrics['coverage_active_positions']:.4f}",
            f"- strategy_total_return: {metrics['strategy_total_return']:.4f}",
            f"- strategy_annualized_sharpe: {metrics['strategy_annualized_sharpe']:.4f}",
            f"- strategy_max_drawdown: {metrics['strategy_max_drawdown']:.4f}",
            f"- trade_win_rate_when_active: {metrics['trade_win_rate_when_active']:.4f}",
            "",
            "## Notes",
            "- This is a strict walk-forward evaluation (no future leakage).",
            "- Use this report for calibration and threshold tuning before live deployment.",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production walk-forward evaluator for SPX/SPY")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Feature CSV path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--symbol", default="SPY", help="Symbol to evaluate")
    parser.add_argument("--min-train-days", type=int, default=756, help="Minimum history before first forecast")
    parser.add_argument("--train-window-days", type=int, default=2520, help="Rolling training window size")
    parser.add_argument("--long-threshold", type=float, default=0.56, help="Long signal threshold")
    parser.add_argument("--short-threshold", type=float, default=0.44, help="Short signal threshold")
    parser.add_argument("--confidence-for-es", type=float, default=0.22, help="Confidence cutoff to route signal to ES")
    parser.add_argument("--fee-bps-per-turnover", type=float, default=1.0, help="Transaction cost in bps per unit turnover")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Feature input file not found: {input_path}")
    if not (0.0 < args.short_threshold < args.long_threshold < 1.0):
        raise ValueError("Thresholds must satisfy 0 < short_threshold < long_threshold < 1")

    df = pd.read_csv(input_path)
    sym_df = df[df["symbol"] == args.symbol].sort_values("timestamp").reset_index(drop=True)
    if len(sym_df) < args.min_train_days + 64:
        raise RuntimeError(
            f"Not enough rows for walk-forward evaluation. Have {len(sym_df)}, "
            f"need at least {args.min_train_days + 64}."
        )

    policy = PolicyConfig(
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        confidence_for_es=args.confidence_for_es,
        fee_bps_per_turnover=args.fee_bps_per_turnover,
    )
    eval_df = run_walk_forward(
        sym_df,
        min_train_days=args.min_train_days,
        train_window_days=args.train_window_days,
        policy=policy,
    )
    metrics = summarize_metrics(eval_df)

    cfg = {
        "symbol": args.symbol,
        "min_train_days": args.min_train_days,
        "train_window_days": args.train_window_days,
        "long_threshold": args.long_threshold,
        "short_threshold": args.short_threshold,
        "confidence_for_es": args.confidence_for_es,
        "fee_bps_per_turnover": args.fee_bps_per_turnover,
    }

    decisions_path = output_dir / f"{args.symbol}_prod_decisions.csv"
    metrics_path = output_dir / f"{args.symbol}_prod_metrics.json"
    report_path = output_dir / f"{args.symbol}_prod_report.md"

    keep_cols = [
        "timestamp",
        "symbol",
        "prob_up",
        "prob_down",
        "confidence",
        "position",
        "instrument",
        "next_return",
        "fees",
        "strategy_return",
        "equity_curve",
        "target_next_direction",
    ]
    eval_df[keep_cols].to_csv(decisions_path, index=False)
    metrics_path.write_text(json.dumps({"config": cfg, "metrics": metrics}, indent=2), encoding="utf-8")
    report_path.write_text(build_report_text(metrics, args.symbol, cfg), encoding="utf-8")

    print(f"Wrote decisions -> {decisions_path}")
    print(f"Wrote metrics -> {metrics_path}")
    print(f"Wrote report -> {report_path}")
    print(f"Final total return: {metrics['strategy_total_return']:.4f}")
    print(f"Annualized Sharpe: {metrics['strategy_annualized_sharpe']:.4f}")


if __name__ == "__main__":
    main()
