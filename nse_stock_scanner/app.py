
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Cache the NSE ticker list during application startup
STOCKS = {}


def load_nse_stocks():
    """Load NSE symbols dynamically from the official NSE equity list."""
    global STOCKS

    urls = [
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    ]

    for url in urls:
        try:
            df = pd.read_csv(url)

            df.columns = df.columns.str.strip().str.upper()

            symbol_col = "SYMBOL"
            name_col = "NAME OF COMPANY"

            if symbol_col not in df.columns:
                continue

            for _, row in df.iterrows():
                symbol = str(row[symbol_col]).strip().upper()

                if not symbol or symbol == "NAN":
                    continue

                name = str(row.get(name_col, symbol)).strip()

                STOCKS[symbol] = {
                    "symbol": symbol,
                    "name": name,
                    "ticker": symbol + ".NS"
                }

            if STOCKS:
                logging.info("Loaded %s NSE stocks", len(STOCKS))
                return

        except Exception as error:
            logging.warning("NSE list error: %s", error)

    logging.warning("Could not download NSE list")


def get_stock_data(ticker):
    try:
        data = yf.download(
            ticker,
            period="2y",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[["Open", "High", "Low", "Close", "Volume"]]
        data = data.dropna()

        if len(data) < 100:
            return None

        return data

    except Exception as error:
        logging.error("Download failed for %s: %s", ticker, error)
        return None


def scan_stock(ticker):
    data = get_stock_data(ticker)

    if data is None:
        return {"error": "Stock data unavailable. Try another stock."}

    try:
        df = data.copy()

        df["MA_5"] = df["Close"].rolling(5).mean()
        df["MA_10"] = df["Close"].rolling(10).mean()
        df["MA_20"] = df["Close"].rolling(20).mean()
        df["MA_50"] = df["Close"].rolling(50).mean()

        df["Return"] = df["Close"].pct_change()
        df["Target"] = df["Close"].shift(-1)

        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        features = [
            "Open", "High", "Low", "Close", "Volume",
            "MA_5", "MA_10", "MA_20", "MA_50", "Return"
        ]

        if len(df) < 100:
            return {"error": "Not enough historical data."}

        X = df[features]
        y = df["Target"]

        split = int(len(df) * 0.8)

        X_train = X.iloc[:split]
        X_test = X.iloc[split:]
        y_train = y.iloc[:split]
        y_test = y.iloc[split:]

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)

        # Predict next trading day's price using latest available features
        latest_features = df[features].iloc[[-1]]
        predicted_price = float(model.predict(latest_features)[0])

        current_price = float(data["Close"].iloc[-1])

        change = ((predicted_price - current_price) / current_price) * 100

        if change > 1:
            signal = "BUY"
        elif change < -1:
            signal = "SELL"
        else:
            signal = "HOLD"

        chart_data = data.tail(60)

        dates = [
            date.strftime("%Y-%m-%d")
            for date in chart_data.index
        ]

        prices = [
            round(float(price), 2)
            for price in chart_data["Close"]
        ]

        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "change": round(change, 2),
            "signal": signal,
            "mae": round(float(mae), 2),
            "dates": dates,
            "prices": prices,
            "last_updated": datetime.now().strftime("%d-%m-%Y %H:%M")
        }

    except Exception as error:
        logging.exception("Prediction error")
        return {"error": "Prediction failed: " + str(error)}


@app.route("/", methods=["GET", "POST"])
def index():

    if not STOCKS:
        load_nse_stocks()

    search = request.values.get("search", "").strip().upper()

    if search:
        filtered_stocks = [
            stock for stock in STOCKS.values()
            if search in stock["symbol"]
            or search in stock["name"].upper()
        ]
    else:
        filtered_stocks = list(STOCKS.values())

    filtered_stocks = sorted(
        filtered_stocks,
        key=lambda stock: stock["symbol"]
    )

    result = None
    selected_stock = ""

    if request.method == "POST":
        selected_stock = request.form.get("ticker", "").strip().upper()

        if selected_stock in STOCKS:
            ticker = STOCKS[selected_stock]["ticker"]
            result = scan_stock(ticker)
        else:
            result = {"error": "Please select a valid NSE stock."}

    return render_template(
        "index.html",
        stocks=filtered_stocks,
        result=result,
        search=search,
        selected_stock=selected_stock,
        total_stocks=len(STOCKS)
    )


@app.route("/refresh-stocks")
def refresh_stocks():
    global STOCKS

    STOCKS = {}
    load_nse_stocks()

    return {
        "message": "Stock list refreshed",
        "total_stocks": len(STOCKS)
    }


if __name__ == "__main__":
    load_nse_stocks()
    app.run(debug=True)
