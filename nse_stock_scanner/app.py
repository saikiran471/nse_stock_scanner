from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Edit this starter universe in the UI or here. These are examples, not recommendations.
DEFAULT_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "HINDUNILVR.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "SUNPHARMA.NS", "MARUTI.NS", "TATAMOTORS.NS",
    "NTPC.NS", "POWERGRID.NS", "TITAN.NS", "ASIANPAINT.NS", "WIPRO.NS",
]
HORIZON = 20
HISTORY_PERIOD = "5y"
ROUND_TRIP_COST = 0.003  # Assumed 0.30% round-trip cost; customize to your actual costs.
MIN_TRAIN_SAMPLES = 180
FEATURES = [
    "ret_5", "ret_10", "ret_20", "ret_60", "ma_gap_20", "ma_gap_50",
    "vol_20", "vol_60", "rsi_14", "volume_change_20", "drawdown_60",
]

def _as_series(frame: pd.DataFrame, column: str) -> pd.Series:
    value = frame[column]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return value

def download_history(ticker: str) -> pd.DataFrame:
    data = yf.download(
        ticker, period=HISTORY_PERIOD, interval="1d",
        auto_adjust=True, progress=False, threads=False, timeout=20,
    )
    if data is None or data.empty:
        raise ValueError("No historical data returned")
    close = pd.to_numeric(_as_series(data, "Close"), errors="coerce")
    volume = pd.to_numeric(_as_series(data, "Volume"), errors="coerce")
    df = pd.DataFrame({"close": close, "volume": volume}).dropna(subset=["close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["close"] > 0]
    if len(df) < 300:
        raise ValueError("Fewer than 300 daily observations")
    return df

def make_dataset(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Features at t predict adjusted-close return from t to t+HORIZON."""
    close = history["close"].astype(float)
    volume = history["volume"].astype(float).replace(0, np.nan)
    returns = close.pct_change()
    f = pd.DataFrame(index=history.index)
    for n in (5, 10, 20, 60):
        f[f"ret_{n}"] = close.pct_change(n)
    f["ma_gap_20"] = close / close.rolling(20).mean() - 1
    f["ma_gap_50"] = close / close.rolling(50).mean() - 1
    f["vol_20"] = returns.rolling(20).std() * np.sqrt(252)
    f["vol_60"] = returns.rolling(60).std() * np.sqrt(252)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    f["rsi_14"] = 100 - (100 / (1 + rs))
    f["volume_change_20"] = volume / volume.rolling(20).mean() - 1
    f["drawdown_60"] = close / close.rolling(60).max() - 1
    target = close.shift(-HORIZON) / close - 1
    # Target end date is used to purge training examples whose outcome would not
    # have been known at the validation date.
    target_end = pd.Series(history.index, index=history.index).shift(-HORIZON)
    valid = f[FEATURES].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    valid &= target.notna()
    X = f.loc[valid, FEATURES]
    y = target.loc[valid]
    end_dates = target_end.loc[valid]
    return X, y, end_dates

def new_model():
    # Small robust baseline; do not treat its outputs as calibrated probabilities.
    return HistGradientBoostingRegressor(
        max_iter=120, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=2.0, random_state=42,
    )

def risk_score(history: pd.DataFrame) -> tuple[int, float, float]:
    close = history["close"].astype(float)
    daily = close.pct_change().dropna()
    vol = float(daily.tail(60).std() * np.sqrt(252)) if len(daily) >= 20 else 1.0
    peak = close.cummax()
    drawdown = float((close / peak - 1).tail(252).min()) if len(close) else -1.0
    # Transparent heuristic score, not a formal risk rating.
    score = int(np.clip((vol / 0.65) * 55 + (abs(min(drawdown, 0)) / 0.45) * 45, 0, 100))
    return score, vol, drawdown

def walk_forward_backtest(X: pd.DataFrame, y: pd.Series, target_end: pd.Series) -> dict:
    """Expanding-window tests; each test origin is at least HORIZON rows apart.
    Training targets must have fully matured by the test origin (purging leakage).
    """
    n = len(X)
    if n < MIN_TRAIN_SAMPLES + HORIZON * 3:
        return {"status": "Insufficient history for walk-forward test", "trades": 0}

    # Use up to four chronological test origins, separated by the forecast horizon.
    first_test = max(MIN_TRAIN_SAMPLES, int(n * 0.68))
    origins = list(range(first_test, n - 1, HORIZON))
    origins = origins[-4:]
    predicted, actual, baseline, trade_returns = [], [], [], []
    for pos in origins:
        origin_date = X.index[pos]
        train_mask = target_end < origin_date
        train_idx = np.flatnonzero(train_mask.to_numpy())
        if len(train_idx) < MIN_TRAIN_SAMPLES:
            continue
        model = new_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = float(model.predict(X.iloc[[pos]])[0])
        truth = float(y.iloc[pos])
        predicted.append(pred)
        actual.append(truth)
        baseline.append(0.0)  # Buy-and-hold return forecast baseline = no price change.
        # Long-only strategy takes a position only when expected return clears costs + 1%.
        trade_returns.append(truth - ROUND_TRIP_COST if pred > ROUND_TRIP_COST + 0.01 else 0.0)

    if not actual:
        return {"status": "Insufficient matured training rows", "trades": 0}
    actual_a = np.asarray(actual)
    pred_a = np.asarray(predicted)
    strat = np.asarray(trade_returns)
    trade_count = int(np.count_nonzero(strat))
    total_return = float(np.prod(1 + strat) - 1)
    avg_trade = float(np.mean(strat[strat != 0])) if trade_count else 0.0
    return {
        "status": "OK",
        "trades": len(actual),
        "mae_pct": float(mean_absolute_error(actual_a, pred_a) * 100),
        "directional_accuracy_pct": float(np.mean(np.sign(actual_a) == np.sign(pred_a)) * 100),
        "baseline_mae_pct": float(mean_absolute_error(actual_a, np.zeros_like(actual_a)) * 100),
        "strategy_return_pct": total_return * 100,
        "active_positions": trade_count,
        "avg_active_return_pct": avg_trade * 100,
    }

def scan_one(ticker: str) -> dict:
    history = download_history(ticker)
    X, y, target_end = make_dataset(history)
    if len(X) < MIN_TRAIN_SAMPLES + HORIZON:
        raise ValueError("Not enough usable history after features")

    # Final model only sees labels whose 20-day outcome has already matured.
    latest_date = X.index[-1]
    mature = target_end < latest_date
    X_train, y_train = X.loc[mature], y.loc[mature]
    if len(X_train) < MIN_TRAIN_SAMPLES:
        raise ValueError("Not enough matured training examples")
    model = new_model()
    model.fit(X_train, y_train)
    latest_features = X.iloc[[-1]]
    predicted_return = float(model.predict(latest_features)[0])
    latest_close = float(history["close"].iloc[-1])
    predicted_price = latest_close * (1 + predicted_return)
    score, vol, drawdown = risk_score(history)
    bt = walk_forward_backtest(X, y, target_end)

    # Conservative heuristic labels. These are screening labels, not personal advice.
    if predicted_return > ROUND_TRIP_COST + 0.015 and score <= 40:
        label = "BUY CANDIDATE"
    elif predicted_return > ROUND_TRIP_COST and score <= 65:
        label = "WATCH"
    else:
        label = "AVOID"
    # Avoid promoting a model that loses to the zero-return baseline on tested MAE.
    if bt.get("status") == "OK" and bt["mae_pct"] >= bt["baseline_mae_pct"]:
        if label == "BUY CANDIDATE":
            label = "WATCH"

    return {
        "ticker": ticker,
        "date": history.index[-1].strftime("%Y-%m-%d"),
        "close": latest_close,
        "predicted_return_pct": predicted_return * 100,
        "predicted_price": predicted_price,
        "risk_score": score,
        "annual_vol_pct": vol * 100,
        "max_drawdown_pct": drawdown * 100,
        "label": label,
        "backtest": bt,
        "observations": len(history),
    }

def scan_universe(tickers: list[str]) -> tuple[list[dict], list[str]]:
    results, errors = [], []
    # Conservative concurrency to reduce provider throttling.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(scan_one, ticker): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                app.logger.warning("Scan failed for %s: %s", ticker, exc)
                errors.append(f"{ticker}: {str(exc)}")
    label_order = {"BUY CANDIDATE": 0, "WATCH": 1, "AVOID": 2}
    results.sort(key=lambda r: (label_order.get(r["label"], 3), -r["predicted_return_pct"], r["risk_score"]))
    return results, errors

@app.route("/", methods=["GET", "POST"])
def index():
    raw = ", ".join(DEFAULT_UNIVERSE)
    results, errors, scanned_at = [], [], None
    error = None
    if request.method == "POST":
        raw = request.form.get("tickers", raw)
        tickers = list(dict.fromkeys(
            item.strip().upper() for item in raw.replace("\n", ",").split(",")
            if item.strip()
        ))
        tickers = [t for t in tickers if len(t) <= 24 and all(c.isalnum() or c in ".-_^=" for c in t)]
        if not tickers:
            error = "Enter at least one valid ticker, separated by commas."
        elif len(tickers) > 40:
            error = "Please scan at most 40 tickers per request to avoid data-provider throttling."
        else:
            results, errors = scan_universe(tickers)
            scanned_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            if not results:
                error = "No tickers could be scanned. Check ticker symbols, internet access, or data-provider availability."
    return render_template(
        "index.html", tickers=raw, results=results, errors=errors,
        error=error, scanned_at=scanned_at, horizon=HORIZON,
        cost_pct=ROUND_TRIP_COST * 100,
    )

if __name__ == "__main__":
    # Debug is off by default; enable locally with FLASK_DEBUG=1 if needed.
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
