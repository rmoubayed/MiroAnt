"""
Prediction Markets Pipeline - Step 3: Export seed files for MiroFish.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT_DIR / "backend" / "data" / "features" / "prediction_markets_features.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "backend" / "data" / "seeds" / "prediction_markets"
DEFAULT_PROMPT_PATH = ROOT_DIR / "backend" / "data" / "seeds" / "prediction_markets_prompt_template.txt"


def _safe(v: object, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def build_market_report(df: pd.DataFrame, market_slug: str, market_question: str) -> str:
    df = df.sort_values("timestamp").reset_index(drop=True)
    min_date = str(df["timestamp"].iloc[0])
    max_date = str(df["timestamp"].iloc[-1])
    n = len(df)

    lines = [
        f"# Prediction Market Report: {market_slug}",
        "",
        f"## Question",
        market_question,
        "",
        f"## Coverage",
        f"- Sessions: {n}",
        f"- Range: {min_date} to {max_date}",
        "",
        "## Daily Narrative",
    ]
    for _, row in df.iterrows():
        date = str(row["timestamp"])
        p = _safe(row.get("implied_prob"), 0.5)
        dp = _safe(row.get("prob_change_1d"), 0.0) * 100
        vol = _safe(row.get("notional_volume"), 0.0)
        trades = int(_safe(row.get("trade_count"), 0.0))
        regime_vol = str(row.get("regime_vol", "unknown"))
        regime_prob = str(row.get("regime_prob_level", "unknown"))
        direction = "rose" if dp > 0 else "fell" if dp < 0 else "was flat"
        lines.extend(
            [
                "",
                f"### {date}",
                (
                    f"Implied probability closed at {p:.3f} and {direction} {abs(dp):.2f} pts day-over-day. "
                    f"Trading activity recorded {trades} trades and {vol:,.2f} notional volume."
                ),
                f"Regimes: volatility={regime_vol}, probability_state={regime_prob}.",
                (
                    "Agent interpretation: policy analysts, news desks, and speculative traders update beliefs "
                    "as new information and liquidity shifts arrive."
                ),
            ]
        )
    return "\n".join(lines)


def build_event_briefs(df: pd.DataFrame, market_slug: str, top_n: int) -> list[tuple[str, str]]:
    w = df.copy()
    w["abs_move"] = w["prob_change_1d"].abs()
    events = w.nlargest(top_n, "abs_move")
    out: list[tuple[str, str]] = []
    for _, row in events.iterrows():
        date = str(row["timestamp"])
        p = _safe(row["implied_prob"], 0.5)
        dp = _safe(row["prob_change_1d"], 0.0) * 100
        direction = "up-move" if dp > 0 else "down-move"
        text = (
            f"# Event Briefing: {market_slug} {direction} on {date}\n\n"
            f"- Implied probability: {p:.3f}\n"
            f"- One-day move: {dp:+.2f} pts\n"
            f"- Trade count: {int(_safe(row.get('trade_count'), 0.0))}\n"
            f"- Notional volume: {_safe(row.get('notional_volume'), 0.0):,.2f}\n\n"
            "## Simulation prompts\n"
            "- Which agent clusters changed their beliefs first?\n"
            "- Did liquidity precede narrative shift, or vice versa?\n"
            "- Is this a temporary overreaction or a persistent repricing?\n"
        )
        out.append((f"event_{date}_{market_slug}.md", text))
    return out


def build_prompt_template(market_slug: str, market_question: str, min_date: str, max_date: str, n: int) -> str:
    return textwrap.dedent(
        f"""\
        Simulate multi-agent prediction-market dynamics for: {market_slug}

        Market question: {market_question}
        Data coverage: {n} sessions from {min_date} to {max_date}.

        Model these agent groups:
        1. Policy/news analysts (information producers)
        2. Flow traders and market-makers (liquidity providers)
        3. Speculative momentum traders
        4. Contrarian value traders
        5. Retail sentiment cluster

        Prediction output:
        - Base-case next-session implied probability
        - Bull/bear alternative paths with probabilities
        - Mispricing vs current market probability
        - Recommended action with confidence and invalidation criteria
        """
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export prediction-market seed docs")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input feature CSV path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--prompt-output", default=str(DEFAULT_PROMPT_PATH), help="Prompt output path")
    parser.add_argument("--market-slug", default="", help="Market slug to export (default: largest market)")
    parser.add_argument("--max-sessions", type=int, default=365, help="Recent sessions to include")
    parser.add_argument("--event-count", type=int, default=10, help="Top event files to export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    prompt_path = Path(args.prompt_output)
    if not input_path.exists():
        raise FileNotFoundError(f"Feature input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if "market_slug" not in df.columns:
        raise RuntimeError("Feature file missing market_slug")

    if args.market_slug:
        mdf = df[df["market_slug"] == args.market_slug].copy()
    else:
        # default: choose market with most rows
        slug = (
            df.groupby("market_slug", dropna=False)["timestamp"]
            .count()
            .sort_values(ascending=False)
            .index[0]
        )
        mdf = df[df["market_slug"] == slug].copy()

    if mdf.empty:
        raise RuntimeError("No rows found for selected market_slug")
    mdf = mdf.sort_values("timestamp").tail(args.max_sessions).reset_index(drop=True)

    market_slug = str(mdf["market_slug"].iloc[0])
    market_question = str(mdf["question"].iloc[0])
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_market_report(mdf, market_slug, market_question)
    report_path = output_dir / f"{market_slug}_market_report.md"
    report_path.write_text(report, encoding="utf-8")

    briefs = build_event_briefs(mdf, market_slug, top_n=args.event_count)
    for filename, content in briefs:
        (output_dir / filename).write_text(content, encoding="utf-8")

    min_date = str(mdf["timestamp"].iloc[0])
    max_date = str(mdf["timestamp"].iloc[-1])
    prompt = build_prompt_template(market_slug, market_question, min_date, max_date, len(mdf))
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")

    print(f"Wrote report -> {report_path}")
    print(f"Wrote {len(briefs)} event briefs -> {output_dir}")
    print(f"Wrote prompt template -> {prompt_path}")


if __name__ == "__main__":
    main()
