"""
Prediction Markets Engine (LLM-seed prep pipeline)

End-to-end flow:
1) Pull market data (Manifold or normalized CSV)
2) Build quantitative features
3) Pull web research headlines (Google News RSS)
4) Export structured seed artifacts for MiroFish
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import textwrap
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import urlopen

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

import prediction_markets_feature_build as pm_feat
import prediction_markets_ingest as pm_ingest


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RAW_OUTPUT = ROOT_DIR / "backend" / "data" / "raw" / "prediction_markets_raw.csv"
DEFAULT_FEATURE_OUTPUT = ROOT_DIR / "backend" / "data" / "features" / "prediction_markets_features.csv"
DEFAULT_SEED_DIR = ROOT_DIR / "backend" / "data" / "seeds" / "prediction_markets"
DEFAULT_PROMPT_PATH = ROOT_DIR / "backend" / "data" / "seeds" / "prediction_markets_prompt_template.txt"
DEFAULT_RESEARCH_PATH = ROOT_DIR / "backend" / "data" / "seeds" / "prediction_markets_research.json"
DEFAULT_SELECTION_PATH = ROOT_DIR / "backend" / "data" / "seeds" / "prediction_markets_selection.json"

_ENV_PATH = ROOT_DIR / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)

_SSL_CTX = ssl.create_default_context()
try:
    import certifi

    _SSL_CTX.load_verify_locations(certifi.where())
except ImportError:
    _SSL_CTX.check_hostname = False
    _SSL_CTX.verify_mode = ssl.CERT_NONE


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _iso_date(v: object) -> str:
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return str(v)[:10]


def _keyword_candidates(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    stop = {
        "the",
        "and",
        "for",
        "will",
        "with",
        "this",
        "that",
        "from",
        "what",
        "when",
        "where",
        "who",
        "how",
        "into",
        "above",
        "below",
        "market",
        "markets",
        "prediction",
        "price",
        "yes",
        "no",
    }
    keep = [w for w in words if len(w) >= 3 and w not in stop]
    # preserve order while de-duplicating
    out: list[str] = []
    seen: set[str] = set()
    for w in keep:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out[:8]


def fetch_google_news_rss(query: str, max_items: int = 10) -> list[dict[str, str]]:
    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(query)
        + "&hl=en-US&gl=US&ceid=US:en"
    )
    with urlopen(url, context=_SSL_CTX) as response:
        content = response.read().decode("utf-8", errors="ignore")

    root = ET.fromstring(content)
    items = root.findall("./channel/item")
    rows: list[dict[str, str]] = []
    for item in items[:max_items]:
        rows.append(
            {
                "title": (item.findtext("title") or "").strip(),
                "link": (item.findtext("link") or "").strip(),
                "pub_date": (item.findtext("pubDate") or "").strip(),
                "source": (item.findtext("source") or "").strip(),
                "query": query,
            }
        )
    return rows


def collect_research(question: str, market_slug: str, custom_queries: list[str], max_headlines: int) -> list[dict[str, str]]:
    queries: list[str] = []
    queries.append(f'"{market_slug.replace("-", " ")}"')
    queries.append(f'"{question}"')
    keywords = _keyword_candidates(question)
    if keywords:
        queries.append(" ".join(keywords[:4]))
    queries.extend(custom_queries)

    all_rows: list[dict[str, str]] = []
    for q in queries:
        try:
            all_rows.extend(fetch_google_news_rss(q, max_items=max_headlines))
        except Exception:
            continue

    # Deduplicate by title/link pair.
    dedup: dict[tuple[str, str], dict[str, str]] = {}
    for row in all_rows:
        key = (row.get("title", ""), row.get("link", ""))
        if key not in dedup:
            dedup[key] = row
    return list(dedup.values())


def _parse_rss_date_to_iso(value: str) -> str:
    if not value:
        return ""
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value[:10]


def compute_signal_math(mdf: pd.DataFrame) -> dict[str, float]:
    mdf = mdf.sort_values("timestamp").reset_index(drop=True)
    last = mdf.iloc[-1]
    p_now = _safe_float(last.get("implied_prob"), 0.5)
    d1 = _safe_float(last.get("prob_change_1d"), 0.0)
    m7 = _safe_float(last.get("prob_momentum_7d"), 0.0)
    z30 = _safe_float(last.get("prob_zscore_30d"), 0.0)
    v30 = _safe_float(last.get("prob_vol_30d"), 0.0)
    liq7 = _safe_float(last.get("liquidity_7d"), 0.0)

    # Simple transparent "fair probability" heuristic.
    p_fair = p_now + 0.55 * m7 - 0.20 * d1 - 0.10 * z30
    p_fair = min(0.99, max(0.01, p_fair))
    edge = p_fair - p_now
    confidence = min(1.0, max(0.0, abs(edge) / (v30 + 0.01)))

    return {
        "prob_now": p_now,
        "prob_fair": p_fair,
        "edge": edge,
        "confidence": confidence,
        "prob_change_1d": d1,
        "prob_momentum_7d": m7,
        "prob_zscore_30d": z30,
        "prob_vol_30d": v30,
        "liquidity_7d": liq7,
    }


def build_report(
    mdf: pd.DataFrame,
    market_slug: str,
    question: str,
    math: dict[str, float],
    headlines: list[dict[str, str]],
) -> str:
    min_date = _iso_date(mdf["timestamp"].iloc[0])
    max_date = _iso_date(mdf["timestamp"].iloc[-1])
    n = len(mdf)

    recent_rows = mdf.tail(min(10, len(mdf)))
    tape_lines = []
    for _, row in recent_rows.iterrows():
        tape_lines.append(
            f"- {_iso_date(row['timestamp'])}: p={_safe_float(row.get('implied_prob'), 0.5):.3f}, "
            f"dp1={_safe_float(row.get('prob_change_1d'), 0.0) * 100:+.2f} pts, "
            f"trades={int(_safe_float(row.get('trade_count'), 0.0))}, "
            f"notional={_safe_float(row.get('notional_volume'), 0.0):,.2f}"
        )

    top_headlines = headlines[:20]
    headline_lines = []
    for h in top_headlines:
        d = _parse_rss_date_to_iso(h.get("pub_date", ""))
        title = h.get("title", "").strip()
        source = h.get("source", "").strip() or "news"
        link = h.get("link", "").strip()
        headline_lines.append(f"- {d} | {source}: {title} ({link})")

    decision_bias = "bullish-YES" if math["edge"] > 0 else "bearish-NO" if math["edge"] < 0 else "neutral"
    headlines_block = headline_lines if headline_lines else ["- No headlines fetched for the selected queries."]

    return "\n".join(
        [
            f"# Prediction Market Engine Report: {market_slug}",
            "",
            "## Market Question",
            question,
            "",
            "## Coverage",
            f"- Sessions: {n}",
            f"- Range: {min_date} to {max_date}",
            "",
            "## Quantitative Signal Snapshot",
            f"- Current implied probability: {math['prob_now']:.3f}",
            f"- Heuristic fair probability: {math['prob_fair']:.3f}",
            f"- Edge (fair - implied): {math['edge']:+.4f}",
            f"- Confidence score: {math['confidence']:.3f}",
            f"- 1-day probability change: {math['prob_change_1d'] * 100:+.2f} pts",
            f"- 7-day probability momentum: {math['prob_momentum_7d'] * 100:+.2f} pts",
            f"- 30-day z-score: {math['prob_zscore_30d']:+.3f}",
            f"- 30-day volatility: {math['prob_vol_30d']:.4f}",
            f"- 7-day liquidity (notional): {math['liquidity_7d']:,.2f}",
            "",
            "## Latest Market Tape (last 10 sessions)",
            *tape_lines,
            "",
            "## Web Research (headline evidence)",
            *headlines_block,
            "",
            "## Agent Framing for Simulation",
            "- Information producers: journalists, policy analysts, niche experts",
            "- Flow/liquidity agents: market makers, high-frequency traders, large accounts",
            "- Narrative agents: social media sentiment clusters with herding/contrarian behavior",
            "- Decision objective: estimate next-session repricing and identify mispricing persistence",
            "",
            "## Provisional Decision Stub",
            f"- Bias: {decision_bias}",
            "- Convert this into action only if simulation consensus aligns with quant edge and confidence.",
        ]
    )


def build_event_briefs(
    mdf: pd.DataFrame,
    market_slug: str,
    headlines: list[dict[str, str]],
    top_n: int,
) -> list[tuple[str, str]]:
    mdf = mdf.copy()
    mdf["abs_move"] = mdf["prob_change_1d"].abs()
    events = mdf.nlargest(top_n, "abs_move")

    # Build headline date index.
    by_date: dict[str, list[dict[str, str]]] = {}
    for h in headlines:
        d = _parse_rss_date_to_iso(h.get("pub_date", ""))
        if d:
            by_date.setdefault(d, []).append(h)

    out: list[tuple[str, str]] = []
    for _, row in events.iterrows():
        d = pd.to_datetime(row["timestamp"])
        date = d.strftime("%Y-%m-%d")
        dp = _safe_float(row.get("prob_change_1d"), 0.0) * 100
        p = _safe_float(row.get("implied_prob"), 0.5)
        direction = "up-move" if dp > 0 else "down-move"

        # Attach nearby headlines in +/- 1 day window.
        nearby: list[str] = []
        for offset in (-1, 0, 1):
            key = (d + timedelta(days=offset)).strftime("%Y-%m-%d")
            for h in by_date.get(key, [])[:3]:
                title = h.get("title", "").strip()
                link = h.get("link", "").strip()
                nearby.append(f"- {key}: {title} ({link})")

        text = (
            f"# Event Briefing: {market_slug} {direction} on {date}\n\n"
            f"## Price Action\n"
            f"- Implied probability close: {p:.3f}\n"
            f"- One-day move: {dp:+.2f} points\n"
            f"- Trades: {int(_safe_float(row.get('trade_count'), 0.0))}\n"
            f"- Notional volume: {_safe_float(row.get('notional_volume'), 0.0):,.2f}\n\n"
            "## Nearby Headline Context\n"
            + ("\n".join(nearby) if nearby else "- No nearby headlines captured in the current RSS sample.")
            + "\n\n"
            "## Agent Questions\n"
            "- Which agent type moved first: information producers or liquidity takers?\n"
            "- Did this move continue next session or mean-revert?\n"
            "- Which narrative should be considered low-confidence noise?\n"
        )
        out.append((f"event_{date}_{market_slug}.md", text))
    return out


def build_prompt_template(market_slug: str, question: str, math: dict[str, float], min_date: str, max_date: str, n: int) -> str:
    return textwrap.dedent(
        f"""\
        Simulate a prediction-market intelligence engine for market: {market_slug}
        Question: {question}

        Data coverage: {n} sessions from {min_date} to {max_date}.
        Quant anchor:
        - implied probability now: {math['prob_now']:.3f}
        - heuristic fair probability: {math['prob_fair']:.3f}
        - edge (fair - implied): {math['edge']:+.4f}
        - confidence: {math['confidence']:.3f}

        Required behavior:
        1) Use web headline evidence to map narrative catalysts.
        2) Use quantitative tape features (momentum, volatility, liquidity) to test whether the move is durable.
        3) Force disagreement between agent clusters and resolve via weighted evidence.
        4) Output calibrated next-session repricing probability and confidence interval.

        Output format:
        - Base case (probability and rationale)
        - Bull case / Bear case branches
        - Mispricing estimate vs current market implied probability
        - Trade recommendation (YES/NO/NO-TRADE) with invalidation
        """
    )


def build_multi_market_report(feat_df: pd.DataFrame) -> str:
    rows = []
    grouped = (
        feat_df.groupby(["market_slug", "question"], dropna=False)
        .agg(
            sessions=("timestamp", "count"),
            min_date=("timestamp", "min"),
            max_date=("timestamp", "max"),
            avg_prob=("implied_prob", "mean"),
            avg_volume=("notional_volume", "mean"),
            avg_trades=("trade_count", "mean"),
        )
        .reset_index()
        .sort_values("sessions", ascending=False)
    )
    for _, r in grouped.iterrows():
        rows.append(
            "- {slug} | sessions={sessions} | range={min_d}->{max_d} | "
            "avg_prob={avg_p:.3f} | avg_vol={avg_v:,.2f} | avg_trades={avg_t:.1f}\n"
            "  question: {q}".format(
                slug=str(r["market_slug"]),
                sessions=int(r["sessions"]),
                min_d=str(r["min_date"])[:10],
                max_d=str(r["max_date"])[:10],
                avg_p=float(r["avg_prob"]),
                avg_v=float(r["avg_volume"]),
                avg_t=float(r["avg_trades"]),
                q=str(r["question"]),
            )
        )

    rows_block = rows if rows else ["- No markets available"]
    return "\n".join(
        [
            "# Prediction Markets Universe Summary",
            "",
            "This file summarizes all markets loaded in the current engine run.",
            "Use it as a cross-market seed so the LLM can reason over multiple contracts.",
            "",
            "## Loaded Markets",
            *rows_block,
        ]
    )


def _market_candidate_table(feat_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        feat_df.groupby(["market_slug", "question"], dropna=False)
        .agg(
            sessions=("timestamp", "count"),
            min_date=("timestamp", "min"),
            max_date=("timestamp", "max"),
            avg_prob=("implied_prob", "mean"),
            avg_volume=("notional_volume", "mean"),
            avg_trades=("trade_count", "mean"),
            prob_vol_30d=("prob_vol_30d", "mean"),
            prob_momentum_7d=("prob_momentum_7d", "mean"),
        )
        .reset_index()
    )
    grouped["coverage_days"] = (
        pd.to_datetime(grouped["max_date"], errors="coerce")
        - pd.to_datetime(grouped["min_date"], errors="coerce")
    ).dt.days.fillna(0).astype(int)
    return grouped


def _heuristic_candidate_score(row: pd.Series) -> float:
    sessions = _safe_float(row.get("sessions"), 0.0)
    coverage = _safe_float(row.get("coverage_days"), 0.0)
    volume = _safe_float(row.get("avg_volume"), 0.0)
    trades = _safe_float(row.get("avg_trades"), 0.0)
    prob_vol = _safe_float(row.get("prob_vol_30d"), 0.0)
    # Favor sufficient history + liquidity + movement.
    return (
        0.40 * min(1.0, sessions / 90.0)
        + 0.25 * min(1.0, coverage / 120.0)
        + 0.20 * min(1.0, volume / 5000.0)
        + 0.10 * min(1.0, trades / 50.0)
        + 0.05 * min(1.0, prob_vol / 0.05)
    )


def _llm_select_market(candidates: pd.DataFrame, model: str) -> dict[str, Any]:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip() or None
    if not api_key:
        raise RuntimeError("LLM_API_KEY not configured for auto market selection")

    payload_rows = []
    for _, r in candidates.iterrows():
        payload_rows.append(
            {
                "market_slug": str(r["market_slug"]),
                "question": str(r["question"]),
                "sessions": int(_safe_float(r["sessions"], 0.0)),
                "coverage_days": int(_safe_float(r["coverage_days"], 0.0)),
                "avg_volume": round(_safe_float(r["avg_volume"], 0.0), 3),
                "avg_trades": round(_safe_float(r["avg_trades"], 0.0), 3),
                "prob_vol_30d": round(_safe_float(r["prob_vol_30d"], 0.0), 6),
                "prob_momentum_7d": round(_safe_float(r["prob_momentum_7d"], 0.0), 6),
                "heuristic_score": round(_safe_float(r["heuristic_score"], 0.0), 4),
            }
        )

    client = OpenAI(api_key=api_key, base_url=base_url)
    sys_prompt = (
        "You are selecting the single best prediction market for a multi-agent simulation engine. "
        "Prioritize: tradability/liquidity, enough historical depth, signal variability, and clear event semantics."
    )
    user_prompt = (
        "Select one best market from the candidates below for MiroFish.\n"
        "Return strict JSON with keys: market_slug, confidence, reason, backup_market_slug.\n"
        f"Candidates:\n{json.dumps(payload_rows, ensure_ascii=True)}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=700,
        response_format={"type": "json_object"},
    )
    content = (resp.choices[0].message.content or "").strip()
    data = json.loads(content)
    return {
        "market_slug": str(data.get("market_slug", "")).strip(),
        "backup_market_slug": str(data.get("backup_market_slug", "")).strip(),
        "reason": str(data.get("reason", "")).strip(),
        "confidence": float(data.get("confidence", 0.0) or 0.0),
        "raw": data,
    }


def _llm_generate_market_seed_bundle(
    model: str,
    market_slug: str,
    question: str,
    math: dict[str, float],
    tape_rows: list[dict[str, Any]],
    headlines: list[dict[str, str]],
    event_count: int,
) -> dict[str, Any]:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip() or None
    if not api_key:
        raise RuntimeError("LLM_API_KEY not configured for LLM seed generation")

    client = OpenAI(api_key=api_key, base_url=base_url)
    sys_prompt = (
        "You are a quantitative prediction-market research analyst creating seed documents for a "
        "multi-agent simulation engine. Use only provided data. Be specific and structured."
    )
    user_prompt = (
        "Generate JSON with keys: research_summary_markdown, quant_summary_markdown, "
        "market_report_markdown, event_briefs, prompt_template_text.\n"
        "- event_briefs must be an array of objects: {date, title, markdown} with up to the requested count.\n"
        "- market_report_markdown should be rich and reference both quantitative and headline evidence.\n"
        "- prompt_template_text should be concise and actionable for simulation.\n\n"
        f"requested_event_count={event_count}\n"
        f"market_slug={market_slug}\n"
        f"question={question}\n"
        f"math={json.dumps(math, ensure_ascii=True)}\n"
        f"tape_rows={json.dumps(tape_rows, ensure_ascii=True)}\n"
        f"headlines={json.dumps(headlines[:30], ensure_ascii=True)}\n"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=3000,
        response_format={"type": "json_object"},
    )
    content = (resp.choices[0].message.content or "").strip()
    data = json.loads(content)
    return {
        "research_summary_markdown": str(data.get("research_summary_markdown", "")).strip(),
        "quant_summary_markdown": str(data.get("quant_summary_markdown", "")).strip(),
        "market_report_markdown": str(data.get("market_report_markdown", "")).strip(),
        "prompt_template_text": str(data.get("prompt_template_text", "")).strip(),
        "event_briefs": data.get("event_briefs", []),
        "raw": data,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end prediction market seed engine")
    parser.add_argument("--provider", choices=["manifold", "csv"], default="manifold")
    parser.add_argument("--market-id", default="", help="Market id (manifold)")
    parser.add_argument("--market-slug", default="", help="Market slug (manifold)")
    parser.add_argument("--top-markets", type=int, default=8, help="Discovery mode: number of markets to ingest")
    parser.add_argument("--discovery-pages", type=int, default=5, help="Number of manifold discovery pages to scan")
    parser.add_argument("--min-days-per-market", type=int, default=3, help="Require at least this many daily points per market")
    parser.add_argument("--max-bets", type=int, default=12000, help="Max manifold bets")
    parser.add_argument("--input-csv", default="", help="Input CSV if provider=csv")
    parser.add_argument("--raw-output", default=str(DEFAULT_RAW_OUTPUT), help="Raw output path")
    parser.add_argument("--feature-output", default=str(DEFAULT_FEATURE_OUTPUT), help="Feature output path")
    parser.add_argument("--seed-output-dir", default=str(DEFAULT_SEED_DIR), help="Seed output directory")
    parser.add_argument("--prompt-output", default=str(DEFAULT_PROMPT_PATH), help="Prompt output path")
    parser.add_argument("--research-output", default=str(DEFAULT_RESEARCH_PATH), help="Research JSON output path")
    parser.add_argument("--selection-output", default=str(DEFAULT_SELECTION_PATH), help="Selection decision JSON path")
    parser.add_argument("--target-market-slug", default="", help="Which market slug to export (default: largest)")
    parser.add_argument("--auto-select-best-market", action="store_true", help="Use LLM to select best market among candidates")
    parser.add_argument("--selection-model", default="", help="LLM model for market selection (default: LLM_MODEL_NAME)")
    parser.add_argument(
        "--seed-model",
        default="",
        help="LLM model for research/math/seed generation (default: LLM_MODEL_NAME)",
    )
    parser.add_argument("--max-sessions", type=int, default=365, help="Max sessions to include in seeds")
    parser.add_argument("--event-count", type=int, default=12, help="Event brief count")
    parser.add_argument("--headline-count", type=int, default=12, help="Max headlines per query")
    parser.add_argument("--query", action="append", default=[], help="Extra research query (repeatable)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_output = Path(args.raw_output)
    feature_output = Path(args.feature_output)
    seed_dir = Path(args.seed_output_dir)
    prompt_output = Path(args.prompt_output)
    research_output = Path(args.research_output)
    selection_output = Path(args.selection_output)

    # 1) Ingest
    if args.provider == "manifold":
        if args.market_id:
            bars = pm_ingest.fetch_manifold_market_by_id(args.market_id, max_bets=args.max_bets)
        elif args.market_slug:
            bars = pm_ingest.fetch_manifold_market_daily(args.market_slug, max_bets=args.max_bets)
        else:
            bars = pm_ingest.fetch_manifold_markets_batch(
                max_markets=args.top_markets,
                max_bets_per_market=args.max_bets,
                discovery_pages=args.discovery_pages,
                min_days_per_market=args.min_days_per_market,
            )
    else:
        if not args.input_csv:
            raise RuntimeError("--input-csv is required for provider=csv")
        bars = pm_ingest.read_normalized_csv(Path(args.input_csv))
    pm_ingest.write_bars_csv(bars, raw_output)

    # 2) Features
    raw_df = pd.read_csv(raw_output)
    feat_df = pm_feat.build_features(raw_df)
    feature_output.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(feature_output, index=False)

    # Pick target market.
    selected_by = "largest_sessions"
    selection_reason = ""
    selection_confidence = 0.0
    backup_slug = ""
    candidate_df = _market_candidate_table(feat_df)
    candidate_df["heuristic_score"] = candidate_df.apply(_heuristic_candidate_score, axis=1)
    candidate_df = candidate_df.sort_values("heuristic_score", ascending=False).reset_index(drop=True)
    if args.target_market_slug:
        mdf = feat_df[feat_df["market_slug"] == args.target_market_slug].copy()
        selected_by = "user_target_market_slug"
    elif args.auto_select_best_market:
        selection_model = args.selection_model.strip() or os.getenv("LLM_MODEL_NAME", "").strip() or "gpt-4o-mini"
        if not selection_model:
            raise RuntimeError("selection model not provided and LLM_MODEL_NAME is empty")
        top_candidates = candidate_df.head(min(len(candidate_df), max(5, args.top_markets)))
        llm_pick = _llm_select_market(top_candidates, model=selection_model)
        pick_slug = llm_pick["market_slug"]
        if pick_slug:
            mdf = feat_df[feat_df["market_slug"] == pick_slug].copy()
            selected_by = "llm_selector"
            selection_reason = llm_pick["reason"]
            selection_confidence = float(llm_pick["confidence"])
            backup_slug = llm_pick["backup_market_slug"]
        else:
            top_slug = str(candidate_df.iloc[0]["market_slug"])
            mdf = feat_df[feat_df["market_slug"] == top_slug].copy()
            selected_by = "heuristic_fallback_no_llm_slug"
    else:
        top_slug = str(candidate_df.iloc[0]["market_slug"])
        mdf = feat_df[feat_df["market_slug"] == top_slug].copy()
    if mdf.empty:
        raise RuntimeError("No rows found for selected market")
    mdf = mdf.sort_values("timestamp").tail(args.max_sessions).reset_index(drop=True)

    market_slug = str(mdf["market_slug"].iloc[0])
    question = str(mdf["question"].iloc[0])

    # Cross-market summary seed (LLM can read across loaded contracts).
    universe_report_path = seed_dir / "prediction_markets_universe_report.md"
    seed_dir.mkdir(parents=True, exist_ok=True)
    universe_report_path.write_text(build_multi_market_report(feat_df), encoding="utf-8")

    # 3) Web research + math
    headlines = collect_research(
        question=question,
        market_slug=market_slug,
        custom_queries=args.query,
        max_headlines=args.headline_count,
    )
    math = compute_signal_math(mdf)
    seed_model = args.seed_model.strip() or os.getenv("LLM_MODEL_NAME", "").strip() or "gpt-4o-mini"

    # LLM-driven research + math interpretation + seed writing.
    tape_rows = (
        mdf.tail(min(20, len(mdf)))[
            ["timestamp", "implied_prob", "prob_change_1d", "trade_count", "notional_volume"]
        ]
        .to_dict(orient="records")
    )
    llm_bundle = _llm_generate_market_seed_bundle(
        model=seed_model,
        market_slug=market_slug,
        question=question,
        math=math,
        tape_rows=tape_rows,
        headlines=headlines,
        event_count=args.event_count,
    )

    # 4) Seed artifacts
    # Keep deterministic core report as fallback; prepend LLM sections.
    deterministic_report = build_report(mdf, market_slug, question, math, headlines)
    llm_report = llm_bundle.get("market_report_markdown", "")
    llm_research_summary = llm_bundle.get("research_summary_markdown", "")
    llm_quant_summary = llm_bundle.get("quant_summary_markdown", "")
    report = "\n\n".join(
        part for part in [llm_report, llm_research_summary, llm_quant_summary, deterministic_report] if part
    )
    report_path = seed_dir / f"{market_slug}_market_report.md"
    report_path.write_text(report, encoding="utf-8")

    briefs = build_event_briefs(mdf, market_slug, headlines, top_n=args.event_count)
    llm_event_briefs = llm_bundle.get("event_briefs", [])
    # Write LLM event briefs first.
    llm_written = 0
    for idx, item in enumerate(llm_event_briefs):
        if not isinstance(item, dict):
            continue
        date = str(item.get("date", "")).strip() or f"llm_{idx+1:02d}"
        title = str(item.get("title", "")).strip() or f"{market_slug} event"
        markdown = str(item.get("markdown", "")).strip()
        if not markdown:
            continue
        safe_date = re.sub(r"[^0-9A-Za-z_-]", "-", date)
        safe_title = re.sub(r"[^0-9A-Za-z_-]", "-", title.lower())[:48].strip("-") or "event"
        filename = f"event_{safe_date}_{safe_title}.md"
        (seed_dir / filename).write_text(markdown, encoding="utf-8")
        llm_written += 1
    # Fallback deterministic events if LLM produced none.
    if llm_written == 0:
        for filename, content in briefs:
            (seed_dir / filename).write_text(content, encoding="utf-8")

    min_date = _iso_date(mdf["timestamp"].iloc[0])
    max_date = _iso_date(mdf["timestamp"].iloc[-1])
    prompt_text = llm_bundle.get("prompt_template_text", "").strip()
    if not prompt_text:
        prompt_text = build_prompt_template(market_slug, question, math, min_date, max_date, len(mdf))
    prompt_output.parent.mkdir(parents=True, exist_ok=True)
    prompt_output.write_text(prompt_text, encoding="utf-8")

    research_output.parent.mkdir(parents=True, exist_ok=True)
    research_output.write_text(
        json.dumps(
            {
                "market_slug": market_slug,
                "question": question,
                "math": math,
                "headline_count": len(headlines),
                "headlines": headlines,
                "llm_research_summary_markdown": llm_research_summary,
                "llm_quant_summary_markdown": llm_quant_summary,
                "seed_model": seed_model,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    selection_output.parent.mkdir(parents=True, exist_ok=True)
    selection_payload = {
        "selected_by": selected_by,
        "selected_market_slug": market_slug,
        "selection_reason": selection_reason,
        "selection_confidence": selection_confidence,
        "backup_market_slug": backup_slug,
        "candidate_count": int(len(candidate_df)),
        "top_candidates": candidate_df.head(10).to_dict(orient="records"),
    }
    selection_output.write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")

    print(f"Wrote raw data -> {raw_output}")
    print(f"Wrote features -> {feature_output}")
    print(f"Wrote report -> {report_path}")
    print(f"Wrote universe summary -> {universe_report_path}")
    print(f"Wrote event briefs -> {seed_dir}")
    print(f"Wrote prompt -> {prompt_output}")
    print(f"Wrote research JSON -> {research_output}")
    print(f"Wrote selection JSON -> {selection_output}")


if __name__ == "__main__":
    main()
