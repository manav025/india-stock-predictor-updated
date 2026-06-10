# 📈 Institutional AI Quant Terminal

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://india-stock-predictor-updated-nthaugwpfoxuawtkyd3jcy.streamlit.app/)

An advanced, end-to-end algorithmic trading dashboard powered by Deep Learning (PyTorch) and Natural Language Processing. This terminal predicts daily directional uptrends for 40 major Nifty50 stocks using a custom LSTM neural network trained on price action, macroeconomics, and news sentiment.

## 🚀 Live Demo
**[Launch the AI Quant Terminal](https://india-stock-predictor-updated-nthaugwpfoxuawtkyd3jcy.streamlit.app/)**

## ✨ Key Features
* **Deep Learning Prediction Engine:** Utilizes a 2-layer Long Short-Term Memory (LSTM) network to process 5-day sequential windows across 15 distinct features.
* **NLP Sentiment Analysis:** Integrates `FinBERT` to score market sentiment by scraping historical and live news headlines from Yahoo Finance and Google News RSS.
* **Macroeconomic Context:** Automatically factors in global macro indicators including India VIX, Crude Oil (WTI), and USD/INR exchange rates.
* **Institutional Backtesting:** Blazing-fast, vectorized historical backtesting engine comparing the AI Strategy's performance against a traditional Buy & Hold approach, calculating Alpha dynamically.
* **Live Analyst Targets:** Fetches real-time institutional target prices via IndianAPI.
* **Interactive Data Visualization:** Fully responsive, dark-mode financial charts built with Plotly, including Candlesticks, Bollinger Bands, RSI, and Return Correlation Matrices.

## 🛠️ Technical Architecture

### 1. Data Pipeline & Feature Engineering
* **Price Data:** Direct raw HTTP requests to Yahoo Finance APIs, engineered with fail-safes to bypass cloud server IP blacklists.
* **Technicals:** SMA (5, 20), RSI (14), and Bollinger Bands (2 SD).
* **Sentiment:** Pre-computed vectorized NLP arrays using the HuggingFace `ProsusAI/finbert` model to maintain a lightweight footprint (0 RAM crash) on cloud deployment.

### 2. The AI Model (PyTorch)
* **Architecture:** 15 Input Dimensions -> LSTM (64 hidden units, 2 layers, 0.3 dropout) -> Dense Network -> Sigmoid Output.
* **Training:** Custom PyTorch DataLoaders with sequence length 5, optimized using Adam and `ReduceLROnPlateau` for early stopping. Handled class imbalance using BCEWithLogitsLoss.

### 3. Frontend & Deployment
* Built entirely in **Python** using **Streamlit**.
* Highly optimized caching (`@st.cache_data`, `@st.cache_resource`) to handle complex matrix calculations in milliseconds.

## 💻 Local Installation

To run this terminal on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/india-stock-predictor.git](https://github.com/your-username/india-stock-predictor.git)
   cd india-stock-predictor'''
2.**Install dependencies:**
   pip install -r requirements.txt

3.**Set up API Keys:***
  Create a .streamlit/secrets.toml file in the root directory and add your IndianAPI key:
  INDIAN_API_KEY = "your_api_key_here"
4.**Launch the terminal:**
  python -m streamlit run app.py


     


  
   

