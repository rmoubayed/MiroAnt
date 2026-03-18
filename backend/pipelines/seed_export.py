"""
Step 3: Export feature data as narrative-rich seed documents for MiroFish.

MiroFish's ontology generator expects real-world actors (people, companies,
agencies, media) mentioned by name so that GraphRAG can extract entities and
relationships. Pure numeric tables produce empty graphs.

This script generates:
  1) A consolidated rolling market analysis report (.md)
  2) Focused event-window briefings for key dates (.md)
  3) A ready-to-paste simulation prompt (.txt)
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT_DIR / "backend" / "data" / "features" / "spx_features.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "backend" / "data" / "seeds" / "spx"
DEFAULT_PROMPT_PATH = ROOT_DIR / "backend" / "data" / "seeds" / "spx_prompt_template.txt"
TRADING_DAYS_PER_YEAR = 252
DEFAULT_YEARS = 40


MARKET_ACTORS = {
    "institutions": [
        "Goldman Sachs", "JPMorgan Chase", "Morgan Stanley", "BlackRock",
        "Citadel Securities", "Bridgewater Associates", "Two Sigma",
    ],
    "central_banks": [
        "Federal Reserve", "Jerome Powell", "FOMC",
        "European Central Bank", "Christine Lagarde",
        "Bank of Japan", "Kazuo Ueda",
    ],
    "government": [
        "U.S. Treasury Department", "Janet Yellen",
        "U.S. Securities and Exchange Commission", "SEC Chair Gary Gensler",
        "Congressional Budget Office",
    ],
    "media": [
        "Bloomberg", "CNBC", "Reuters", "Wall Street Journal",
        "Financial Times", "MarketWatch",
    ],
    "market_participants": [
        "retail traders on Reddit WallStreetBets",
        "institutional portfolio managers",
        "systematic trend-following CTAs",
        "options market-makers on CBOE",
        "SPY ETF authorized participants",
        "ES futures scalpers on CME",
    ],
}


def _safe(value: object) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def _direction_word(ret: float) -> str:
    if ret > 1.5:
        return "surged"
    if ret > 0.4:
        return "rallied"
    if ret > 0:
        return "edged higher"
    if ret > -0.4:
        return "drifted lower"
    if ret > -1.5:
        return "sold off"
    return "plunged"


def _vol_narrative(vol: float) -> str:
    if vol > 30:
        return "crisis-level realized volatility"
    if vol > 20:
        return "elevated volatility reminiscent of risk-off episodes"
    if vol > 15:
        return "moderately choppy conditions"
    return "subdued, low-volatility trading"


def _regime_narrative(trend: str, momentum: str, vol_regime: str) -> str:
    parts = []
    if trend == "uptrend":
        parts.append("the 20-day moving average remained above the 50-day, confirming a short-term uptrend that institutional trend-followers at Bridgewater and Two Sigma typically ride")
    else:
        parts.append("the 20-day moving average had crossed below the 50-day, signaling a short-term downtrend that systematic CTAs tend to sell into")

    if momentum == "overbought":
        parts.append("RSI readings above 70 flagged overbought conditions, a level where Goldman Sachs' tactical desk historically trims equity exposure")
    elif momentum == "oversold":
        parts.append("RSI had dropped below 30 into oversold territory, levels where BlackRock's systematic strategies often start accumulating")
    else:
        parts.append("momentum indicators sat in neutral territory, leaving directional conviction split among Wall Street desks")

    if vol_regime == "high_vol":
        parts.append("realized volatility had spiked into the top quartile of the trailing year, prompting CBOE options market-makers to widen spreads and risk managers at major banks to tighten exposure limits")
    else:
        parts.append("realized volatility stayed within normal bounds, keeping options premiums relatively compressed on CBOE")

    return ". ".join(parts) + "."


def build_daily_narrative(row: pd.Series, symbol: str) -> str:
    date = str(row["timestamp"])
    close = _safe(row["close"])
    ret_pct = _safe(row["return_1d"]) * 100
    rsi = _safe(row["rsi_14"])
    macd = _safe(row["macd_line"])
    macd_sig = _safe(row["macd_signal"])
    vol = _safe(row["vol_20d"]) * 100
    drawdown = _safe(row["drawdown"]) * 100
    atr = _safe(row["atr_14"])
    trend = row.get("trend_regime", "unknown")
    momentum = row.get("momentum_regime", "unknown")
    vol_regime = row.get("vol_regime", "unknown")

    move = _direction_word(ret_pct)
    vol_desc = _vol_narrative(vol)
    regime_desc = _regime_narrative(trend, momentum, vol_regime)

    macd_stance = "bullish" if macd >= macd_sig else "bearish"

    return textwrap.dedent(f"""\
    On {date}, {symbol} closed at {close:.2f}, having {move} {abs(ret_pct):.2f}% on the session. \
    Trading desks at JPMorgan and Morgan Stanley reported {vol_desc} with 20-day annualized \
    volatility at {vol:.1f}%. The index sat {abs(drawdown):.1f}% below its rolling 52-week high, \
    a level closely watched by portfolio managers at BlackRock and risk teams at Citadel Securities.

    Technical conditions: {regime_desc}

    MACD stood at {macd:.4f} against its signal line at {macd_sig:.4f}, giving a {macd_stance} \
    crossover reading. RSI(14) was {rsi:.1f}. The average true range over 14 days was {atr:.3f}, \
    suggesting that CME ES futures traders should expect intraday swings of roughly that magnitude.

    Bloomberg and CNBC commentary reflected divided opinion. Analysts at Goldman Sachs pointed to \
    the {trend} regime as supportive of their base case, while strategists at Morgan Stanley noted \
    the {momentum} momentum reading warranted caution. Retail traders on Reddit WallStreetBets \
    leaned {macd_stance}, amplifying the directional signal through leveraged SPY options flow \
    tracked by CBOE.

    The Federal Reserve's latest policy stance continued to anchor macro sentiment. FOMC minutes \
    had flagged data-dependency, leaving Jerome Powell's next statement as a key catalyst. Janet \
    Yellen's Treasury Department refunding announcements added another layer of supply/demand \
    uncertainty for rates-sensitive equity positioning.
    """)


def build_consolidated_report(
    df: pd.DataFrame,
    symbol: str,
    requested_years: int | None = None,
    actual_years: float | None = None,
) -> str:
    min_date = str(df["timestamp"].iloc[0])
    max_date = str(df["timestamp"].iloc[-1])
    n_rows = len(df)
    coverage_line = ""
    if requested_years is not None and actual_years is not None:
        coverage_line = (
            f"- Requested history: ~{requested_years} years\n"
            f"- Actual coverage in this export: ~{actual_years:.1f} years ({n_rows} sessions)\n"
        )
    else:
        coverage_line = f"- Sessions included: {n_rows}\n"

    header = textwrap.dedent(f"""\
    # SPX Market Analysis Report: {symbol}
    ## Period: {min_date} to {max_date} ({n_rows} trading sessions)

    This report provides a comprehensive narrative history of {symbol} price action, \
    technical regime shifts, and the institutional / macro context surrounding each session. \
    It is designed as seed material for multi-agent social simulation of market behavior.

    ---

    ## Dataset Coverage
    {coverage_line}
    ---

    ## Market Participants Referenced in This Report

    ### Institutional Investors & Banks
    Goldman Sachs, JPMorgan Chase, Morgan Stanley, BlackRock, Citadel Securities, \
    Bridgewater Associates, Two Sigma.

    ### Central Banks & Government
    Federal Reserve (Chair Jerome Powell, FOMC), European Central Bank (Christine Lagarde), \
    Bank of Japan (Kazuo Ueda), U.S. Treasury Department (Secretary Janet Yellen), \
    SEC (Chair Gary Gensler), Congressional Budget Office.

    ### Media & Information Sources
    Bloomberg, CNBC, Reuters, Wall Street Journal, Financial Times, MarketWatch.

    ### Market Participants
    Retail traders on Reddit WallStreetBets, institutional portfolio managers, \
    systematic trend-following CTAs, options market-makers on CBOE, \
    SPY ETF authorized participants, ES futures scalpers on CME.

    ---

    ## Relationships Between Actors

    - The Federal Reserve **sets monetary policy** that directly impacts equity valuations via discount rates.
    - Jerome Powell **communicates forward guidance** to markets through FOMC statements and press conferences.
    - Goldman Sachs and JPMorgan **publish research and trade recommendations** that influence institutional positioning.
    - BlackRock and Bridgewater **manage trillions in assets**, their rebalancing flows move markets mechanically.
    - Citadel Securities **provides liquidity** as a dominant market-maker; their positioning affects bid-ask dynamics.
    - CNBC and Bloomberg **amplify narratives** that shape retail and institutional sentiment in real-time.
    - Reddit WallStreetBets **coordinates retail options flow** that can trigger gamma squeezes tracked by CBOE.
    - The SEC **regulates market structure**; enforcement actions and rule proposals create uncertainty.
    - Janet Yellen's Treasury **manages debt issuance**, which competes with equities for capital allocation.

    ---

    """)

    daily_sections: list[str] = []
    # Group by year to keep large seed output navigable.
    df = df.copy()
    df["year"] = pd.to_datetime(df["timestamp"]).dt.year
    for year, year_df in df.groupby("year", sort=True):
        year_rows = len(year_df)
        year_start = str(year_df["timestamp"].iloc[0])
        year_end = str(year_df["timestamp"].iloc[-1])
        section_header = textwrap.dedent(
            f"""\
            ## Year {year} ({year_rows} sessions)
            ### Range: {year_start} to {year_end}
            """
        ).strip()
        year_narratives: list[str] = []
        for _, row in year_df.iterrows():
            year_narratives.append(build_daily_narrative(row, symbol))
        daily_sections.append(section_header + "\n\n" + "\n---\n\n".join(year_narratives))

    return header + "\n---\n\n".join(daily_sections)


def build_event_briefings(df: pd.DataFrame, symbol: str, top_n: int = 10) -> list[tuple[str, str]]:
    """Pick the top-N most volatile days and write focused event briefings."""
    df = df.copy()
    df["abs_return"] = df["return_1d"].abs()
    event_days = df.nlargest(top_n, "abs_return")
    ts_to_idx = {str(ts): i for i, ts in enumerate(df["timestamp"].tolist())}

    briefings: list[tuple[str, str]] = []
    for _, row in event_days.iterrows():
        date = str(row["timestamp"])
        event_idx = ts_to_idx.get(date, 0)
        ret_pct = _safe(row["return_1d"]) * 100
        close = _safe(row["close"])
        open_px = _safe(row["open"])
        high_px = _safe(row["high"])
        low_px = _safe(row["low"])
        volume = _safe(row["volume"])
        vol = _safe(row["vol_20d"]) * 100
        drawdown = _safe(row["drawdown"]) * 100
        rsi = _safe(row["rsi_14"])
        macd = _safe(row["macd_line"])
        macd_signal = _safe(row["macd_signal"])
        atr = _safe(row["atr_14"])
        trend_regime = row.get("trend_regime", "unknown")
        momentum_regime = row.get("momentum_regime", "unknown")
        vol_regime = row.get("vol_regime", "unknown")
        direction = "rally" if ret_pct > 0 else "sell-off"
        shock_rank = "extreme"
        abs_ret = abs(ret_pct)
        if abs_ret < 2.0:
            shock_rank = "moderate"
        elif abs_ret < 4.0:
            shock_rank = "large"

        # Local event window context (2 sessions before and after when available).
        start_idx = max(0, event_idx - 2)
        end_idx = min(len(df) - 1, event_idx + 2)
        window_df = df.iloc[start_idx : end_idx + 1]
        window_rows: list[str] = []
        for _, w_row in window_df.iterrows():
            w_date = str(w_row["timestamp"])
            w_ret = _safe(w_row["return_1d"]) * 100
            w_close = _safe(w_row["close"])
            marker = " (event day)" if w_date == date else ""
            window_rows.append(f"- {w_date}: close {w_close:.2f}, return {w_ret:+.2f}%{marker}")

        next_day_text = "N/A (event at data boundary)"
        if event_idx + 1 < len(df):
            next_row = df.iloc[event_idx + 1]
            next_day_text = (
                f"{str(next_row['timestamp'])}: {(_safe(next_row['return_1d']) * 100):+.2f}%"
            )

        article = "an" if shock_rank.startswith(("e", "a", "i", "o", "u")) else "a"
        window_text = "\n".join(window_rows)
        content = (
            f"# Event Briefing: {symbol} {direction.title()} on {date}\n\n"
            f"## What Happened\n"
            f"{symbol} recorded a {abs(ret_pct):.2f}% {direction} on {date}, closing at {close:.2f}. "
            f"This was an outsized move relative to the trailing 20-day realized volatility of {vol:.1f}%. "
            f"The session qualifies as **{article} {shock_rank} shock regime** for scenario generation.\n\n"
            f"## Session Microstructure Snapshot\n"
            f"- Open: {open_px:.2f}\n"
            f"- High: {high_px:.2f}\n"
            f"- Low: {low_px:.2f}\n"
            f"- Close: {close:.2f}\n"
            f"- Volume: {volume:,.0f}\n"
            f"- Drawdown vs rolling 52-week high: {drawdown:.1f}%\n"
            f"- ATR(14): {atr:.3f}\n\n"
            f"## Technical State at the Shock\n"
            f"- Trend regime: **{trend_regime}**\n"
            f"- Momentum regime: **{momentum_regime}**\n"
            f"- Volatility regime: **{vol_regime}**\n"
            f"- RSI(14): {rsi:.1f}\n"
            f"- MACD line: {macd:.4f}\n"
            f"- MACD signal: {macd_signal:.4f}\n"
            f"- One-day follow-through reference: {next_day_text}\n\n"
            f"## Local Event Window (T-2 to T+2)\n"
            f"{window_text}\n\n"
            f"## Likely Catalysts\n"
            f"Moves of this magnitude typically coincide with one or more of:\n"
            f"- A Federal Reserve policy decision or Jerome Powell press conference\n"
            f"- A major macro data release (CPI, NFP, GDP) surprising the consensus tracked by Bloomberg\n"
            f"- An earnings surprise from a mega-cap index constituent (Apple, Microsoft, Nvidia, Amazon)\n"
            f"- A geopolitical shock repriced by Goldman Sachs and JPMorgan risk desks overnight\n"
            f"- A large options expiration event creating gamma-driven flows through CBOE market-makers\n\n"
            f"## Market Reaction Chain\n"
            f"1. **Institutional desks** at Goldman Sachs and Morgan Stanley likely adjusted hedges within minutes.\n"
            f"2. **Systematic CTAs** at Bridgewater and Two Sigma would have received trend signals triggering position changes.\n"
            f"3. **Options market-makers** on CBOE repriced implied volatility across the SPY and ES term structure.\n"
            f"4. **Retail traders** on Reddit WallStreetBets amplified the move via leveraged 0DTE options.\n"
            f"5. **Bloomberg and CNBC** broadcast the move, creating a feedback loop of narrative and flow.\n"
            f"6. **The Federal Reserve** monitored financial conditions; extreme moves can influence forward guidance.\n"
            f"7. **BlackRock** portfolio managers assessed rebalancing needs for passive index funds.\n\n"
            f"## Agent Playbook (Use in Simulation)\n"
            f"- **Fed/FOMC agents**: debate inflation-growth tradeoff and communication strategy after the shock.\n"
            f"- **Bank strategist agents**: split into tactical mean-reversion vs. trend-continuation camps.\n"
            f"- **CTA/systematic agents**: react mechanically to trend and volatility state transitions.\n"
            f"- **Options/MM agents**: simulate gamma hedging feedback and spread widening.\n"
            f"- **Media agents**: propagate either crisis framing or stabilization narrative.\n"
            f"- **Retail flow agents**: model momentum chasing and reversal panic behavior.\n"
            f"- **Treasury/SEC agents**: introduce policy/regulatory messaging uncertainty.\n"
            f"- **Passive allocator agents**: rebalance exposure based on drawdown and volatility.\n\n"
            f"## Simulation Value\n"
            f"This date represents a stress-test scenario. Agents should simulate:\n"
            f"- How institutional risk managers respond to the magnitude of the move\n"
            f"- Whether retail sentiment amplifies or fades the initial impulse\n"
            f"- How media narratives shape the following session's opening gap\n"
            f"- Whether the Federal Reserve or Treasury Department issues any calming statements\n"
        )

        filename = f"event_{date}_{symbol}.md"
        briefings.append((filename, content))

    return briefings


def build_prompt_template(symbol: str, min_date: str, max_date: str, n_sessions: int) -> str:
    return textwrap.dedent(f"""\
    Simulate multi-agent market dynamics around {symbol} (S&P 500 proxy) using the uploaded seed documents.

    The seed data covers {n_sessions} trading sessions from {min_date} to {max_date}. \
    It describes daily price action, technical regime states, and the roles of key market actors \
    including the Federal Reserve, major investment banks, media outlets, and retail traders.

    Your simulation should model these actors as distinct agents:
    1. **Federal Reserve / FOMC** - sets monetary policy, communicates forward guidance
    2. **Institutional strategists** (Goldman Sachs, JPMorgan, Morgan Stanley) - publish research, manage positioning
    3. **Systematic funds** (Bridgewater, Two Sigma, CTAs) - follow trend and volatility signals mechanically
    4. **Passive giants** (BlackRock) - rebalance index funds, their flows create predictable pressure
    5. **Market-makers** (Citadel Securities, CBOE options desks) - provide liquidity, manage gamma exposure
    6. **Media** (Bloomberg, CNBC, Wall Street Journal) - amplify narratives, shape sentiment
    7. **Retail traders** (Reddit WallStreetBets, SPY options flow) - momentum-chase, create short-term dislocations
    8. **Government / Regulators** (Treasury / SEC) - influence via issuance, regulation, and public statements

    Prediction objective:
    - Forecast the probability distribution of {symbol} direction and magnitude over the next 1-5 sessions.
    - Identify the dominant narrative driving each scenario branch.
    - Surface disagreements between agent types (e.g., institutional bullish vs. retail bearish).

    Output format:
    1. **Base case** (highest probability scenario) with P(up), P(down), expected move size
    2. **Bullish tail** scenario with trigger conditions and probability
    3. **Bearish tail** scenario with trigger conditions and probability
    4. **Key risk factors** that could invalidate the base case
    5. **Recommended position** (instrument, direction, entry logic, stop-loss, invalidation)
    6. **Confidence score** (0-1) reflecting inter-agent agreement
    """)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export narrative seed files for MiroFish")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input feature CSV path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output seed directory")
    parser.add_argument("--prompt-output", default=str(DEFAULT_PROMPT_PATH), help="Prompt template file")
    parser.add_argument("--symbol", default="SPY", help="Symbol to export")
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Maximum number of most recent sessions in the consolidated report. "
        "If omitted, uses --years * 252.",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=DEFAULT_YEARS,
        help=f"Approximate number of years to include (default: {DEFAULT_YEARS}). "
        f"Converted using {TRADING_DAYS_PER_YEAR} trading days/year.",
    )
    parser.add_argument(
        "--require-full-years",
        action="store_true",
        help="Fail if available data does not cover requested --years.",
    )
    parser.add_argument(
        "--event-count",
        type=int,
        default=10,
        help="Number of top-volatility event briefings to generate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    prompt_output = Path(args.prompt_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Feature input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if "symbol" not in df.columns:
        raise RuntimeError("Feature file missing 'symbol' column")

    sym_df = df[df["symbol"] == args.symbol].copy()
    if sym_df.empty:
        raise RuntimeError(f"No rows found for symbol {args.symbol}")

    sym_df = sym_df.sort_values("timestamp")
    full_sym_df = sym_df.copy()
    if args.max_sessions is not None:
        session_limit = args.max_sessions
    else:
        session_limit = args.years * TRADING_DAYS_PER_YEAR
    if session_limit <= 0:
        raise ValueError("Session limit must be positive. Set --years or --max-sessions > 0.")
    sym_df = sym_df.tail(session_limit).reset_index(drop=True)
    actual_years = len(sym_df) / TRADING_DAYS_PER_YEAR
    full_years_available = len(full_sym_df) / TRADING_DAYS_PER_YEAR
    if args.max_sessions is None and args.require_full_years and full_years_available + 1e-9 < args.years:
        raise RuntimeError(
            f"Requested ~{args.years} years, but only ~{full_years_available:.1f} years are available "
            f"for {args.symbol} in {input_path}. Ingest older history first."
        )
    if args.max_sessions is None and full_years_available + 1e-9 < args.years:
        print(
            f"Warning: Requested ~{args.years} years but only ~{full_years_available:.1f} years are available "
            f"for {args.symbol}. Exporting all available sessions."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{args.symbol}_market_report.md"
    report_text = build_consolidated_report(
        sym_df,
        args.symbol,
        requested_years=None if args.max_sessions is not None else args.years,
        actual_years=actual_years,
    )
    report_path.write_text(report_text, encoding="utf-8")
    print(f"Wrote consolidated report ({len(report_text):,} chars) -> {report_path}")

    briefings = build_event_briefings(sym_df, args.symbol, top_n=args.event_count)
    for filename, content in briefings:
        filepath = output_dir / filename
        filepath.write_text(content, encoding="utf-8")
    print(f"Wrote {len(briefings)} event briefings -> {output_dir}")

    prompt_output.parent.mkdir(parents=True, exist_ok=True)
    min_date = str(sym_df["timestamp"].iloc[0])
    max_date = str(sym_df["timestamp"].iloc[-1])
    prompt_text = build_prompt_template(args.symbol, min_date, max_date, len(sym_df))
    prompt_output.write_text(prompt_text, encoding="utf-8")
    print(f"Wrote prompt template -> {prompt_output}")

    print(f"\nUpload workflow:")
    print(f"  1) Upload {report_path.name} + event_*.md files in the 'Real-World Seeds' area")
    print(f"  2) Paste contents of {prompt_output} into 'Simulation Prompt' field")
    print(f"  3) Click 'Start Engine'")


if __name__ == "__main__":
    main()
