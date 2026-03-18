"""
Prediction Markets Pipeline - Step 1: Ingest market trade/probability data.

Supported providers:
- manifold (public API, no key required)
- csv (local file with normalized columns)
"""

from __future__ import annotations

import argparse
import csv
import json
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import urlopen


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT_DIR / "backend" / "data" / "raw" / "prediction_markets_raw.csv"

_SSL_CTX = ssl.create_default_context()
try:
    import certifi

    _SSL_CTX.load_verify_locations(certifi.where())
except ImportError:
    _SSL_CTX.check_hostname = False
    _SSL_CTX.verify_mode = ssl.CERT_NONE


@dataclass
class PMBar:
    timestamp: str
    market_id: str
    market_slug: str
    question: str
    source: str
    implied_prob: float
    trade_count: int
    notional_volume: float


def _http_get_json(url: str, params: dict | None = None) -> dict | list:
    if params:
        url = f"{url}?{urlencode(params)}"
    with urlopen(url, context=_SSL_CTX) as response:
        return json.loads(response.read().decode("utf-8"))


def _as_iso_date(ms_or_iso: object) -> str:
    if ms_or_iso is None:
        return ""
    if isinstance(ms_or_iso, (int, float)):
        return datetime.fromtimestamp(float(ms_or_iso) / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
    return str(ms_or_iso)[:10]


def _to_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def fetch_manifold_market_daily(slug: str, max_bets: int = 10000) -> list[PMBar]:
    market = _http_get_json("https://api.manifold.markets/v0/slug/" + slug)
    if not isinstance(market, dict):
        raise RuntimeError(f"Unexpected Manifold market payload for slug={slug}")

    market_id = str(market.get("id", slug))
    question = str(market.get("question", slug))
    market_slug = str(market.get("slug", slug))

    request_limit = max(1, min(int(max_bets), 1000))
    bets: list[dict] = []
    before_id: str | None = None
    while len(bets) < max_bets:
        params: dict[str, object] = {"contractId": market_id, "limit": request_limit}
        if before_id:
            params["before"] = before_id
        page = _http_get_json("https://api.manifold.markets/v0/bets", params)
        if not isinstance(page, list):
            raise RuntimeError(f"Unexpected Manifold bets payload for market={market_id}")
        if not page:
            break
        remaining = max_bets - len(bets)
        bets.extend(page[:remaining])
        before_id = str(page[-1].get("id", "")) or None
        if len(page) < request_limit or before_id is None:
            break
    if not bets:
        raise RuntimeError(f"No bets returned for market slug={slug}")

    daily: dict[str, dict[str, float | int]] = {}
    # Oldest->newest so "last probability of day" is deterministic.
    bets_sorted = sorted(
        bets,
        key=lambda b: _to_float(b.get("createdTime"), 0.0),
    )
    for bet in bets_sorted:
        day = _as_iso_date(bet.get("createdTime"))
        if not day:
            continue
        bucket = daily.setdefault(day, {"trade_count": 0, "notional_volume": 0.0, "last_prob": 0.0})
        bucket["trade_count"] = int(bucket["trade_count"]) + 1
        bucket["notional_volume"] = float(bucket["notional_volume"]) + abs(_to_float(bet.get("amount"), 0.0))
        # Use probAfter if present; fallback to probBefore.
        p = bet.get("probAfter")
        if p is None:
            p = bet.get("probBefore")
        bucket["last_prob"] = _to_float(p, float(bucket["last_prob"]))

    bars: list[PMBar] = []
    for day in sorted(daily.keys()):
        b = daily[day]
        bars.append(
            PMBar(
                timestamp=day,
                market_id=market_id,
                market_slug=market_slug,
                question=question,
                source="manifold",
                implied_prob=float(b["last_prob"]),
                trade_count=int(b["trade_count"]),
                notional_volume=float(b["notional_volume"]),
            )
        )
    return bars


def fetch_manifold_market_by_id(market_id: str, max_bets: int = 10000) -> list[PMBar]:
    market = _http_get_json("https://api.manifold.markets/v0/market/" + market_id)
    if not isinstance(market, dict):
        raise RuntimeError(f"Unexpected Manifold market payload for market_id={market_id}")
    slug = str(market.get("slug", market_id))
    return fetch_manifold_market_daily(slug=slug, max_bets=max_bets)


def fetch_default_manifold_market(max_bets: int = 10000) -> list[PMBar]:
    markets = _http_get_json("https://api.manifold.markets/v0/markets", {"limit": 1})
    if not isinstance(markets, list) or not markets:
        raise RuntimeError("No markets returned from Manifold /v0/markets")
    market = markets[0]
    slug = str(market.get("slug", ""))
    if not slug:
        raise RuntimeError("Default market has no slug in /v0/markets response")
    return fetch_manifold_market_daily(slug=slug, max_bets=max_bets)


def fetch_manifold_markets_batch(
    max_markets: int = 10,
    max_bets_per_market: int = 1000,
    discovery_pages: int = 5,
    min_days_per_market: int = 3,
) -> list[PMBar]:
    """
    Fetch multiple markets and merge their daily bars.
    Uses /v0/markets as the discovery source, then fetches per-market bets.
    """
    page_limit = 100
    pages = max(1, int(discovery_pages))
    discovered: list[dict] = []
    before_id: str | None = None
    for _ in range(pages):
        params: dict[str, object] = {"limit": page_limit}
        if before_id:
            params["before"] = before_id
        markets = _http_get_json("https://api.manifold.markets/v0/markets", params)
        if not isinstance(markets, list) or not markets:
            break
        discovered.extend(markets)
        before_id = str(markets[-1].get("id", "")) or None
        if len(markets) < page_limit or before_id is None:
            break
    if not discovered:
        raise RuntimeError("No markets returned from Manifold /v0/markets discovery")

    all_bars: list[PMBar] = []
    selected = 0
    for m in discovered:
        slug = str(m.get("slug", "")).strip()
        if not slug:
            continue
        try:
            bars = fetch_manifold_market_daily(slug=slug, max_bets=max_bets_per_market)
        except Exception:
            continue
        if not bars or len(bars) < min_days_per_market:
            continue
        all_bars.extend(bars)
        selected += 1
        if selected >= max_markets:
            break

    if not all_bars:
        raise RuntimeError(
            "Failed to fetch bars for any market in manifold batch mode. "
            "Try lowering --min-days-per-market."
        )
    return all_bars


def read_normalized_csv(input_path: Path) -> list[PMBar]:
    rows = list(csv.DictReader(input_path.read_text(encoding="utf-8").splitlines()))
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
    if not rows:
        raise RuntimeError(f"No rows found in CSV: {input_path}")
    missing = required - set(rows[0].keys())
    if missing:
        raise RuntimeError(f"CSV missing required columns: {sorted(missing)}")

    bars: list[PMBar] = []
    for row in rows:
        bars.append(
            PMBar(
                timestamp=str(row["timestamp"])[:10],
                market_id=str(row["market_id"]),
                market_slug=str(row["market_slug"]),
                question=str(row["question"]),
                source=str(row["source"]),
                implied_prob=_to_float(row.get("implied_prob"), 0.0),
                trade_count=int(_to_float(row.get("trade_count"), 0.0)),
                notional_volume=_to_float(row.get("notional_volume"), 0.0),
            )
        )
    return bars


def write_bars_csv(bars: Iterable[PMBar], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "market_id",
                "market_slug",
                "question",
                "source",
                "implied_prob",
                "trade_count",
                "notional_volume",
            ],
        )
        writer.writeheader()
        for bar in bars:
            writer.writerow(
                {
                    "timestamp": bar.timestamp,
                    "market_id": bar.market_id,
                    "market_slug": bar.market_slug,
                    "question": bar.question,
                    "source": bar.source,
                    "implied_prob": f"{bar.implied_prob:.6f}",
                    "trade_count": str(bar.trade_count),
                    "notional_volume": f"{bar.notional_volume:.6f}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest prediction market data")
    parser.add_argument("--provider", choices=["manifold", "csv"], default="manifold")
    parser.add_argument(
        "--market-slug",
        default="",
        help="Market slug for manifold provider (optional)",
    )
    parser.add_argument("--market-id", default="", help="Market id for manifold provider (optional)")
    parser.add_argument(
        "--top-markets",
        type=int,
        default=1,
        help="When no market-id/slug is provided, fetch this many markets from manifold discovery feed",
    )
    parser.add_argument(
        "--discovery-pages",
        type=int,
        default=5,
        help="How many /v0/markets pages to scan when selecting multiple markets",
    )
    parser.add_argument(
        "--min-days-per-market",
        type=int,
        default=3,
        help="Skip markets with fewer than this many daily points",
    )
    parser.add_argument("--max-bets", type=int, default=10000, help="Max bets to fetch for manifold")
    parser.add_argument("--input-csv", default="", help="Input CSV path for csv provider")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if args.provider == "manifold":
        if args.market_id:
            bars = fetch_manifold_market_by_id(market_id=args.market_id, max_bets=args.max_bets)
            print(f"Fetched {len(bars)} daily bars from manifold market_id={args.market_id}")
        elif args.market_slug:
            bars = fetch_manifold_market_daily(slug=args.market_slug, max_bets=args.max_bets)
            print(f"Fetched {len(bars)} daily bars from manifold slug={args.market_slug}")
        else:
            bars = fetch_manifold_markets_batch(
                max_markets=args.top_markets,
                max_bets_per_market=args.max_bets,
                discovery_pages=args.discovery_pages,
                min_days_per_market=args.min_days_per_market,
            )
            print(f"Fetched {len(bars)} daily bars across top {args.top_markets} manifold markets")
    else:
        if not args.input_csv:
            raise RuntimeError("--input-csv is required when --provider csv")
        bars = read_normalized_csv(Path(args.input_csv))
        print(f"Loaded {len(bars)} rows from CSV provider")

    write_bars_csv(bars, output_path)
    print(f"Wrote normalized market data to {output_path}")


if __name__ == "__main__":
    main()
