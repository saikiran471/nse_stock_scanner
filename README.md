📈 NSE Stock Price Predictor & Scanner

A machine learning-powered web application built with Python, Flask, and Scikit-learn to analyze NSE-listed stocks, predict potential returns, evaluate investment risk, and rank stocks based on data-driven screening criteria.

🌐 Live Demo: https://nse-stock-scanner-2-jvfv.onrender.com

🚀 Project Overview

The NSE Stock Price Predictor is a web-based stock analysis and screening application that uses historical Indian stock market data to estimate potential stock returns over the next 20 trading days.

The application scans a predefined list of NSE-listed companies, applies machine learning techniques, evaluates risk factors, and generates stock classifications to help users identify potential investment opportunities.

It also includes historical walk-forward backtesting to evaluate the performance of its screening strategy.

✨ Key Features
📊 Automated Stock Scanning: Analyzes multiple NSE-listed stocks.
🤖 Machine Learning Predictions: Estimates potential returns over the next 20 trading days.
📈 Stock Ranking: Ranks stocks based on predicted returns and configured screening criteria.
⚠️ Risk Assessment: Evaluates volatility and maximum drawdown.
🎯 Investment Signals: Classifies stocks into Buy Candidate, Watch, and Avoid categories.
🔍 Technical Indicators: Uses RSI, moving averages, historical returns, and volume changes.
🧪 Historical Backtesting: Evaluates strategy performance using walk-forward testing.
💰 Transaction Cost Analysis: Includes an assumed 0.30% round-trip transaction cost in relevant backtesting calculations.
🌐 Interactive Web Interface: Displays stock analysis and results through a Flask-powered dashboard.
🛠️ Technologies Used
Technology	Purpose
Python	Core programming language
Flask	Backend web framework
Pandas	Data manipulation and analysis
NumPy	Numerical computations
yfinance	Historical stock market data
Scikit-learn	Machine learning model
HTML & CSS	Frontend development
JavaScript	Interactive functionality
Chart.js	Data visualization
🧠 Machine Learning Approach

The application uses the HistGradientBoostingRegressor algorithm from Scikit-learn to estimate potential stock returns over a 20-trading-day prediction horizon.

Feature Engineering

The model uses several historical market features, including:

Historical stock returns
Moving-average differences
Price volatility
Relative Strength Index (RSI)
Trading volume changes
Maximum drawdown

These features are used to identify patterns in historical stock market data and estimate potential future returns.

Prediction Workflow
Retrieve historical stock market data.
Clean and preprocess the collected data.
Calculate technical indicators and relevant features.
Apply the trained machine learning model.
Estimate potential 20-day stock returns.
Calculate risk-related metrics.
Rank stocks and generate screening classifications.
📉 Backtesting

The application includes a walk-forward backtesting mechanism to evaluate the historical performance of its stock screening strategy.

Backtesting considers:

Historical stock returns.
Strategy performance across historical periods.
Risk-related measurements.
An assumed 0.30% round-trip transaction cost.

This helps assess how the screening approach would have performed under historical market conditions.

📋 Understanding Stock Signals
Signal	Description
🟢 BUY CANDIDATE	Meets the configured prediction and risk criteria.
🟡 WATCH	Requires further monitoring or does not fully meet candidate criteria.
🔴 AVOID	Does not satisfy the configured screening criteria.

Important Metrics:

Predicted Return: Estimated percentage return over the next 20 trading days.
Risk Score: Measures estimated risk using factors such as volatility and drawdown. Higher scores indicate greater estimated risk.
Backtesting Results: Historical strategy performance, including estimated transaction costs.
⚙️ Installation & Setup

Follow these steps to run the project locally.

1. Clone the Repository
bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git

2. Navigate to the Project Directory
bash
cd nse_stock_scanner

3. Install Dependencies
bash
pip install -r requirements.txt

4. Run the Application
bash
python app.py

5. Open in Your Browser

Visit:

text
http://127.0.0.1:5000


The application should now be running locally.

📂 Project Structure
text
nse_stock_scanner/
│
├── app.py
├── requirements.txt
├── README.md
│
└── templates/
    └── index.html

File	Description
app.py	Flask backend, stock scanning, machine learning, data processing, and backtesting logic.
requirements.txt	Required Python dependencies.
templates/index.html	Frontend dashboard and stock analysis interface.
README.md	Project documentation.
📖 How to Use
Open the live application or run it locally.
Access the stock scanner dashboard.
Start the stock scanning process.
Allow the application to retrieve and analyze historical market data.
Review the ranked NSE stock list.
Compare predicted returns and risk scores.
Explore the generated Buy Candidate, Watch, and Avoid classifications.
Review historical backtesting results.

The application currently uses a predefined universe of NSE-listed stocks. The ticker list can be modified to include additional companies.

⚠️ Limitations
Predictions depend on the quality and availability of historical market data.
Machine learning models cannot reliably predict unexpected market events.
Yahoo Finance data availability and update frequency may vary.
Results depend on the predefined stock universe and model configuration.
Historical backtesting does not guarantee future profitability.
Actual brokerage fees, taxes, slippage, and execution prices may differ from simulated assumptions.
🔮 Future Enhancements
Individual NSE stock search functionality.
Expandable stock universe with configurable filters.
Additional technical indicators.
Interactive historical stock price charts.
Portfolio tracking and watchlists.
Comparison of multiple machine learning algorithms.
Benchmark comparisons against market indices.
Integration of additional financial data sources.
🎯 Project Objective

The primary objective of this project is to demonstrate the practical application of Machine Learning, Data Analytics, and Web Development in financial market analysis.

It combines historical market data, feature engineering, predictive modeling, risk evaluation, and web technologies to create an accessible stock screening platform.

👨‍💻 Disclaimer

This project is developed strictly for educational and research purposes.

Predicted returns and generated stock classifications are estimates based on historical data and configured model criteria. They are not guaranteed investment outcomes or financial advice. Always conduct independent research before making investment decisions.

⭐ If you find this project interesting, consider giving this repository a star!

🌐 Try the Live Application: https://nse-stock-scanner-2-jvfv.onrender.com
