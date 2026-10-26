# NSE Stock Price Predictor Using Flask and Machine Learning

## 1. Introduction
NSE Stock Price Predictor is a web-based application developed using Python, Flask, and Machine Learning. It analyzes historical Indian stock market data to estimate potential stock returns over the next 20 trading days. The application scans selected NSE-listed stocks, ranks them, evaluates risk, and provides screening signals to support investment research.

## 2. Objectives
- Analyze historical stock market data.
- Predict potential future stock returns using machine learning.
- Scan and rank multiple NSE-listed stocks.
- Evaluate stock risk using volatility and drawdown.
- Test strategy performance through historical backtesting.
- Display results through a user-friendly web interface.

## 3. Technologies Used
- **Python:** Core programming language.
- **Flask:** Web application framework.
- **Pandas and NumPy:** Data processing and numerical calculations.
- **yfinance:** Historical stock market data retrieval.
- **Scikit-learn:** Machine learning model development.
- **HTML and CSS:** Web page structure and styling.
- **JavaScript and Chart.js:** Interactive charts and result visualization.

## 4. How the Project Was Developed

### Step 1: Data Collection
Historical stock prices are retrieved from Yahoo Finance for selected NSE stocks. NSE ticker symbols generally end with `.NS`, such as `RELIANCE.NS` and `TCS.NS`.

### Step 2: Data Preprocessing
The collected data is organized and checked for missing or invalid values. Historical price and volume information is prepared for analysis.

### Step 3: Feature Engineering
The application calculates indicators such as:
- Historical returns
- Moving-average differences
- Price volatility
- Relative Strength Index (RSI)
- Volume changes
- Maximum drawdown

These features help the model identify patterns in historical market data.

### Step 4: Machine Learning
A HistGradientBoostingRegressor model from Scikit-learn is used to estimate stock returns over a 20-trading-day horizon. The model learns relationships between historical market features and subsequent returns.

### Step 5: Stock Scanning and Ranking
The application processes the selected stock universe, estimates potential returns, calculates risk scores, and ranks stocks according to its screening rules.

### Step 6: Backtesting
Walk-forward backtesting evaluates the strategy on historical periods. The application accounts for an assumed round-trip transaction cost of 0.30% when calculating relevant performance metrics.

### Step 7: Web Application
Flask connects the Python analysis to a browser-based interface. The results are presented in a table or dashboard so users can compare stocks.

## 5. Features
- Automated scanning of a predefined NSE stock list.
- Machine-learning-based return estimates.
- Stock ranking based on screening criteria.
- Risk assessment using volatility and drawdown.
- Buy Candidate, Watch, and Avoid classifications.
- Walk-forward backtesting.
- Transaction-cost-aware strategy evaluation.
- Web-based results display.

## 6. How to Install and Run

### Requirements
Install Python 3.10 or a compatible version supported by the project's dependencies.

### Installation
1. Download or extract the project folder.
2. Open a terminal in the project directory.
3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the application:

   ```bash
   python app.py
   ```

5. Open the following address in your browser:

   `http://127.0.0.1:5000`

## 7. How to Use the Application

1. Start the Flask application.
2. Open the web interface in your browser.
3. Run the stock scan using the available interface.
4. Wait for historical stock data to be downloaded and analyzed.
5. Review the ranked list of stocks.
6. Compare predicted returns and risk scores.
7. Use the screening labels to identify stocks for further research.
8. Review the backtesting results to understand historical strategy performance.

The scanner uses a predefined stock universe. Adding more stocks requires updating the ticker list in the application.

## 8. Understanding the Results

- **Predicted Return:** The model's estimated percentage return over the next 20 trading days.
- **Risk Score:** A score based on risk-related measures, including recent volatility and drawdown. A higher score indicates greater estimated risk.
- **BUY CANDIDATE:** The stock meets the application's configured prediction and risk criteria.
- **WATCH:** The stock does not fully meet the candidate criteria and may require further monitoring.
- **AVOID:** The stock fails the configured screening criteria.
- **Backtesting Metrics:** Historical results used to evaluate the strategy, including the effect of estimated transaction costs.

These labels are screening outputs, not guaranteed trading recommendations.

## 9. Project Structure

```text
nse_stock_scanner/
├── app.py
├── requirements.txt
├── README.md
└── templates/
    └── index.html
```

- `app.py`: Contains Flask routes, data processing, machine learning, stock scanning, and backtesting logic.
- `requirements.txt`: Lists the Python packages required to run the project.
- `templates/index.html`: Contains the web interface for displaying stock analysis and results.
- `README.md`: Provides project documentation and setup instructions.

## 10. Limitations
- Predictions depend on the quality and availability of historical data.
- The model cannot reliably predict unexpected market events.
- Yahoo Finance data availability and update timing may vary.
- Results depend on the predefined stock universe and model settings.
- Historical backtesting does not guarantee future profitability.
- Transaction costs and simulated execution may differ from actual trading conditions.

## 11. Future Enhancements
- Add a search option for individual NSE stocks.
- Expand the stock universe and provide configurable filters.
- Add more technical indicators and benchmark comparisons.
- Include portfolio tracking and watchlists.
- Add charts for historical prices and predicted returns.
- Improve validation and compare multiple machine learning models.
- Integrate additional data sources where permitted.

## 12. Conclusion
The NSE Stock Price Predictor demonstrates how machine learning and web development can be combined to analyze stocks, estimate potential returns, assess risk, and rank investment candidates. It provides an educational platform for exploring data-driven stock screening and historical strategy evaluation.

**Disclaimer:** This project is intended for educational and research purposes only. Predictions are estimates, not guarantees of future returns or financial advice.
