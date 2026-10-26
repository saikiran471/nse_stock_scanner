# NSE Stock Scanner (Flask + ML)

A learning project that scans a user-selected list of NSE ticker symbols, predicts a 20-trading-day adjusted-close return, estimates a simple risk score, and reports a small walk-forward backtest with an assumed transaction cost.

## Requirements
- Python 3.10+
- Internet access for Yahoo Finance data

## Run locally

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install and run:
```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

## Usage
- The text box starts with a sample universe of large and well-known NSE tickers.
- Replace or edit the comma-separated list. Use Yahoo Finance NSE symbols ending in `.NS`, such as `RELIANCE.NS` or `TCS.NS`.
- Keep each scan to at most 40 tickers. The app uses a small thread pool to limit simultaneous requests.
- The scanner downloads up to five years of daily adjusted prices, constructs trailing indicators, trains a histogram gradient-boosting regressor, estimates the next 20-trading-day return, computes a heuristic risk score, and evaluates historical forecasts.

## Methodology and caveats
- The target is `close[t+20] / close[t] - 1`.
- The feature rows only use data through date `t`.
- Training rows are purged by target-end date so a training label must have matured before each test origin.
- Walk-forward test origins are spaced by 20 observations to avoid testing overlapping 20-day holding windows. The sample is intentionally small for an interactive app; increase folds and test coverage for serious research.
- The assumed round-trip trading cost defaults to 0.30% in `app.py`. Replace it with an estimate appropriate for your broker, order size, taxes, slippage, and trade type.
- The risk score is a transparent heuristic from recent annualized volatility and trailing drawdown, not a standard financial risk metric.
- `BUY CANDIDATE`, `WATCH`, and `AVOID` are screening labels derived from fixed thresholds, not personalized investment advice.
- The strategy return shown is based on a tiny sample of historical walk-forward forecast origins. Do not treat it as a robust performance estimate. Expand the backtest, compare against baselines, test multiple market regimes, include realistic costs, and paper trade before use.
- The model uses price/volume history only. It does not ingest news, fundamentals, corporate events, or broader economic data.
- yfinance is an independent open-source tool, not affiliated with or endorsed by Yahoo. Check the data provider's terms and suitability before any production/commercial use.

## Production hardening
Use a production WSGI server, add caching and rate limits, persist fetched data for reproducibility, log model/data versions, add robust retry handling, and review licensing/data-provider permissions. Do not enable Flask debug mode on a public server.
