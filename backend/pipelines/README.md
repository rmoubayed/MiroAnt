# SPX Pipeline (Steps 1-3)

This folder implements the first three build steps to create MiroFish-ready
seed material for SPX/market prediction simulations.

1. **Market data ingestion** - pull OHLCV from Alpha Vantage or Nasdaq Data Link
2. **Feature engineering** - compute RSI, MACD, ATR, volatility, regime labels
3. **Seed export** - generate narrative-rich `.md` documents naming real market
   actors (Fed, banks, media, retail) so the GraphRAG can extract entities and
   relationships

## Environment variables

Set one of the following in your `.env` or shell:

- `ALPHAVANTAGE_API_KEY` (for Alpha Vantage - recommended)
- `NASDAQ_DATA_LINK_API_KEY` (optional, for Nasdaq Data Link / Quandl)

## Install dependencies

From project root:

```bash
npm run setup:backend
```

## Step 1: Ingest raw market data

Alpha Vantage (default):

```bash
cd backend
uv run python pipelines/spx_ingest.py --provider alpha_vantage --symbols SPY
```

Nasdaq Data Link:

```bash
cd backend
uv run python pipelines/spx_ingest.py --provider nasdaq_data_link --dataset-code CHRIS/CME_ES1
```

Output:

- `backend/data/raw/spx_raw.csv`

## Step 2: Build technical features

```bash
cd backend
uv run python pipelines/feature_build.py
```

Output:

- `backend/data/features/spx_features.csv`

## Step 3: Export MiroFish seed files

```bash
cd backend
uv run python pipelines/seed_export.py --symbol SPY --years 40 --event-count 10
```

Strict mode (fail if less than full 40-year coverage is available):

```bash
cd backend
uv run python pipelines/seed_export.py --symbol SPY --years 40 --event-count 10 --require-full-years
```

Outputs:

- `backend/data/seeds/spx/SPY_market_report.md` - consolidated narrative report
- `backend/data/seeds/spx/event_*.md` - focused briefings for top-volatility days
- `backend/data/seeds/spx_prompt_template.txt` - paste into simulation prompt field

## Using the output in MiroFish UI

1. Upload `SPY_market_report.md` + all `event_*.md` files in the **Real-World Seeds** area
2. Paste contents of `spx_prompt_template.txt` into the **Simulation Prompt** field
3. Click **Start Engine**

The ontology generator will extract entities like Federal Reserve, Goldman Sachs,
Jerome Powell, CNBC, Reddit WallStreetBets, etc. and build a knowledge graph of
their relationships. Agents will then simulate market dynamics across those actors.

## Prediction Markets Pipeline (parallel to SPX)

You can run a similar 3-step seed pipeline for prediction markets:

1) Ingest (Manifold example):

```bash
cd backend
uv run python pipelines/prediction_markets_ingest.py --provider manifold
```

1) Build features:

```bash
cd backend
uv run python pipelines/prediction_markets_feature_build.py
```

1) Export MiroFish seeds:

```bash
cd backend
uv run python pipelines/prediction_markets_seed_export.py --event-count 12
```

Outputs:

- `backend/data/seeds/prediction_markets/*_market_report.md`
- `backend/data/seeds/prediction_markets/event_*.md`
- `backend/data/seeds/prediction_markets_prompt_template.txt`

### One-command Engine (data + web research + quant + seeds)

If you want an end-to-end run in one command:

```bash
cd backend
uv run python pipelines/prediction_markets_engine.py --provider manifold --market-id PLUUON2uQt --event-count 12 --headline-count 12
```

To load a **universe of markets** (recommended for richer LLM seeds):

```bash
cd backend
uv run python pipelines/prediction_markets_engine.py --provider manifold --top-markets 12 --discovery-pages 8 --min-days-per-market 3 --max-bets 1200 --event-count 20 --headline-count 8
```

To let the engine automatically choose the best market (LLM + quant heuristic):

```bash
cd backend
uv run python pipelines/prediction_markets_engine.py --provider manifold --top-markets 50 --discovery-pages 12 --min-days-per-market 2 --max-bets 1200 --auto-select-best-market --event-count 20 --headline-count 8
```

Additional outputs from the one-command engine:

- `backend/data/raw/prediction_markets_raw.csv`
- `backend/data/features/prediction_markets_features.csv`
- `backend/data/seeds/prediction_markets_research.json`
- `backend/data/seeds/prediction_markets/prediction_markets_universe_report.md`
- `backend/data/seeds/prediction_markets_selection.json`

## Step 4: Production evaluation (walk-forward + policy)

Run an explicit walk-forward evaluation with calibrated probabilities and a trading
policy for SPY/ES routing:

```bash
cd backend
uv run python pipelines/prod_eval.py --symbol SPY
```

Outputs:

- `backend/data/eval/SPY_prod_decisions.csv` - daily probability + position decisions
- `backend/data/eval/SPY_prod_metrics.json` - machine-readable KPI metrics
- `backend/data/eval/SPY_prod_report.md` - human-readable evaluation summary
