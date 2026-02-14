import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os

from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="Coca-Cola (KO) Stock Analysis",
    layout="wide"
)

st.title("📊 Coca-Cola (KO) Stock Analysis & Prediction")

@st.cache_data
def load_stock_data():
    df = yf.download("KO", period="max", auto_adjust=False)
    df.reset_index(inplace=True)
    return df

stock_history = load_stock_data()

if isinstance(stock_history.columns, pd.MultiIndex):
    stock_history.columns = stock_history.columns.get_level_values(0)

stock_history['MA_20'] = stock_history['Close'].rolling(20).mean()
stock_history['MA_50'] = stock_history['Close'].rolling(50).mean()
stock_history['Daily_Return'] = stock_history['Close'].pct_change()
stock_history['Volatility'] = stock_history['Daily_Return'].rolling(20).std()

@st.cache_resource
def load_model():
    if os.path.exists("rf_model.pkl"):
        return joblib.load("rf_model.pkl")
    return None

rf_model = load_model()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 EDA", "📉 Indicators", "🤖 Model Evaluation", "🔮 Live Prediction"]
)


with tab1:
    st.header("Exploratory Data Analysis")

    st.subheader("Closing Price Trend")
    st.line_chart(stock_history.set_index("Date")["Close"])

    st.subheader("Trading Volume")
    st.bar_chart(stock_history.set_index("Date")["Volume"])

    st.write("Summary Statistics")
    st.dataframe(stock_history[['Open','High','Low','Close','Volume']].describe())


with tab2:
    st.header("Technical Indicators")

    st.subheader("Moving Averages")
    st.line_chart(
        stock_history.set_index("Date")[["MA_20", "MA_50"]]
    )

    st.subheader("Daily Returns Distribution")
    st.line_chart(
        stock_history.set_index("Date")["Daily_Return"]
    )

    st.subheader("Volatility (20-Day Rolling)")
    st.line_chart(
        stock_history.set_index("Date")["Volatility"]
    )


with tab3:
    st.header("Model Evaluation")

    st.markdown("""
    Models trained in notebook:
    - Linear Regression (Baseline)
    - Random Forest Regressor
    - LSTM (Advanced)
    """)

    st.subheader("Performance Metrics (From Notebook)")
    col1, col2 = st.columns(2)

    col1.metric("Linear Regression MAE", "≈ 2.84")
    col2.metric("Random Forest MAE", "≈ 1.92")

    st.info(
        "Random Forest performed better than Linear Regression "
        "by capturing non-linear relationships in stock data."
    )

with tab4:
    st.header("Live Stock Price Prediction")

    if rf_model is None:
        st.error("❌ Trained model (rf_model.pkl) not found.")
        st.stop()

    st.write("Fetching latest market data...")

    live_data = yf.download("KO", period="90d", auto_adjust=False)

    live_data['MA_20'] = live_data['Close'].rolling(20).mean()
    live_data['MA_50'] = live_data['Close'].rolling(50).mean()
    live_data['Daily_Return'] = live_data['Close'].pct_change()
    live_data['Volatility'] = live_data['Daily_Return'].rolling(20).std()

    live_data = live_data.dropna()

    if live_data.empty:
        st.error("Not enough recent data to predict.")
        st.stop()

    latest = live_data.iloc[-1]

    X_live = np.array([
        latest['Open'],
        latest['High'],
        latest['Low'],
        latest['Volume'],
        latest['MA_20'],
        latest['MA_50'],
        latest['Daily_Return'],
        latest['Volatility']
    ]).reshape(1, -1)

    predicted_price = rf_model.predict(X_live)

    st.success(
        f"📌 Predicted Next Closing Price: **${predicted_price[0]:.2f}**"
    )

st.markdown("---")
st.caption("End-to-End Stock Analysis & Prediction | Built with Python, ML & Streamlit")
