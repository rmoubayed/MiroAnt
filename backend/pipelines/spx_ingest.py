"""
Step 1: Ingest SPX proxy market data into a normalized raw dataset.

Supported providers:
- alpha_vantage (recommended for daily SPY pulls)
- nasdaq_data_link (formerly Quandl)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import ssl
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import urlopen, Request


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "backend" / "data" / "raw"

_SSL_CTX = ssl.create_default_context()
try:
    import certifi
    _SSL_CTX.load_verify_locations(certifi.where())
except ImportError:
    _SSL_CTX.check_hostname = False
    _SSL_CTX.verify_mode = ssl.CERT_NONE


@dataclass
class PriceBar:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    source: str


def _http_get_json(base_url: str, params: dict) -> dict:
    query = urlencode(params)
    with urlopen(f"{base_url}?{query}", context=_SSL_CTX) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_get_csv(base_url: str, params: dict) -> list[dict]:
    query = urlencode(params)
    with urlopen(f"{base_url}?{query}", context=_SSL_CTX) as response:
        decoded = response.read().decode("utf-8")
    return list(csv.DictReader(decoded.splitlines()))


def fetch_alpha_vantage_daily(symbol: str, api_key: str) -> list[PriceBar]:
    outputsize = os.getenv("AV_OUTPUTSIZE", "full")
    data = _http_get_json(
        "https://www.alphavantage.co/query",
        {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": api_key,
        },
    )
    if "Error Message" in data:
        raise RuntimeError(f"Alpha Vantage error for {symbol}: {data['Error Message']}")

    ts_key = "Time Series (Daily)"
    if ts_key not in data:
        if "Information" in data:
            raise RuntimeError(f"Alpha Vantage: {data['Information']}")
        raise RuntimeError(f"Unexpected Alpha Vantage response for {symbol}: {list(data.keys())}")

    bars: list[PriceBar] = []
    for timestamp, values in data[ts_key].items():
        bars.append(
            PriceBar(
                timestamp=timestamp,
                open=float(values["1. open"]),
                high=float(values["2. high"]),
                low=float(values["3. low"]),
                close=float(values["4. close"]),
                volume=float(values["5. volume"]),
                symbol=symbol,
                source="alpha_vantage",
            )
        )
    return sorted(bars, key=lambda b: b.timestamp)


def fetch_nasdaq_data_link(dataset_code: str, api_key: str | None = None) -> list[PriceBar]:
    params = {"order": "asc"}
    if api_key:
        params["api_key"] = api_key

    rows = _http_get_csv(
        f"https://data.nasdaq.com/api/v3/datasets/{dataset_code}.csv",
        params,
    )
    if not rows:
        raise RuntimeError(f"No rows returned for dataset {dataset_code}")

    column_map = {c.lower(): c for c in rows[0].keys()}
    required = ["date", "open", "high", "low", "close"]
    missing = [c for c in required if c not in column_map]
    if missing:
        raise RuntimeError(
            f"Dataset {dataset_code} missing required columns {missing}. "
            "Provide a dataset with OHLC columns."
        )

    volume_col = column_map.get("volume")
    symbol = dataset_code.replace("/", "_")
    bars: list[PriceBar] = []
    for row in rows:
        bars.append(
            PriceBar(
                timestamp=row[column_map["date"]],
                open=float(row[column_map["open"]]),
                high=float(row[column_map["high"]]),
                low=float(row[column_map["low"]]),
                close=float(row[column_map["close"]]),
                volume=float(row[volume_col]) if volume_col and row.get(volume_col) else 0.0,
                symbol=symbol,
                source="nasdaq_data_link",
            )
        )
    return bars


def write_bars_csv(bars: Iterable[PriceBar], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "open", "high", "low", "close", "volume", "symbol", "source"],
        )
        writer.writeheader()
        for bar in bars:
            writer.writerow(
                {
                    "timestamp": bar.timestamp,
                    "open": f"{bar.open:.8f}",
                    "high": f"{bar.high:.8f}",
                    "low": f"{bar.low:.8f}",
                    "close": f"{bar.close:.8f}",
                    "volume": f"{bar.volume:.2f}",
                    "symbol": bar.symbol,
                    "source": bar.source,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest SPX proxy price data")
    parser.add_argument(
        "--provider",
        choices=["alpha_vantage", "nasdaq_data_link"],
        default="alpha_vantage",
        help="Data provider to use",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY"],
        help="Symbols for alpha_vantage provider",
    )
    parser.add_argument(
        "--dataset-code",
        default="CHRIS/CME_ES1",
        help="Dataset code for nasdaq_data_link provider",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "spx_raw.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    if args.provider == "alpha_vantage":
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing ALPHAVANTAGE_API_KEY in environment")

        all_bars: list[PriceBar] = []
        for symbol in args.symbols:
            bars = fetch_alpha_vantage_daily(symbol=symbol, api_key=api_key)
            all_bars.extend(bars)
            print(f"Fetched {len(bars)} bars for {symbol}")
    else:
        api_key = os.getenv("NASDAQ_DATA_LINK_API_KEY")
        all_bars = fetch_nasdaq_data_link(dataset_code=args.dataset_code, api_key=api_key)
        print(f"Fetched {len(all_bars)} bars for {args.dataset_code}")

    write_bars_csv(all_bars, output_path)
    print(f"Wrote normalized raw data to {output_path}")
    print(f"Generated at {datetime.utcnow().isoformat()}Z")


if __name__ == "__main__":
    main()
