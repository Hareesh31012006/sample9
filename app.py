# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import math

# ---------------------------------------------
# APP CONFIGURATION
# ---------------------------------------------
st.set_page_config(
    page_title="📈 LSTM + FinBERT Stock Market Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Stock Market Prediction and Sentiment Analyzer")
st.markdown("""
Welcome to the **Stock Market Intelligence Dashboard** powered by  
**LSTM (Long Short-Term Memory)** for price forecasting and  
**FinBERT** for financial sentiment analysis.
""")

# ---------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------
st.sidebar.header("🛠️ User Controls")

ticker = st.sidebar.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, INFY.NS):", "AAPL")
start_date = st.sidebar.date_input("Start Date", dt.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", dt.date.today())

look_back = st.sidebar.slider("Select Look-back Days for LSTM", 30, 120, 60)
epochs = st.sidebar.slider("Training Epochs", 10, 200, 50)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 64)

# ---------------------------------------------
# DOWNLOAD DATA
# ---------------------------------------------
st.subheader("📥 Step 1: Fetch Historical Stock Data")

data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    st.error("⚠️ No data found! Try a valid stock ticker.")
    st.stop()

st.dataframe(data.tail(10))

# ---------------------------------------------
# VISUALIZATION - CANDLESTICK & MOVING AVERAGES
# ---------------------------------------------
st.subheader("📈 Stock Trend Visualization")

ma1, ma2 = 20, 50
data['MA20'] = data['Close'].rolling(window=ma1).mean()
data['MA50'] = data['Close'].rolling(window=ma2).mean()

fig = make_subplots(rows=1, cols=1)
candlestick = go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="Candlestick"
)
ma20_line = go.Scatter(x=data.index, y=data['MA20'], mode='lines', name=f"MA{ma1}")
ma50_line = go.Scatter(x=data.index, y=data['MA50'], mode='lines', name=f"MA{ma2}")

fig.add_trace(candlestick)
fig.add_trace(ma20_line)
fig.add_trace(ma50_line)
fig.update_layout(xaxis_rangeslider_visible=False, height=500)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------
# DATA PREPROCESSING FOR LSTM
# ---------------------------------------------
st.subheader("🧮 Step 2: Preprocess Data for LSTM")

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

x_data, y_data = [], []
for i in range(look_back, len(scaled_data)):
    x_data.append(scaled_data[i - look_back:i, 0])
    y_data.append(scaled_data[i, 0])

x_data, y_data = np.array(x_data), np.array(y_data)
x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))

split = int(0.8 * len(x_data))
x_train, y_train = x_data[:split], y_data[:split]
x_test, y_test = x_data[split:], y_data[split:]

st.write(f"Training samples: {x_train.shape[0]} | Testing samples: {x_test.shape[0]}")

# ---------------------------------------------
# LSTM MODEL CREATION
# ---------------------------------------------
st.subheader("🧠 Step 3: Build and Train LSTM Model")

model_filename = f"{ticker}_lstm_model.h5"

if not os.path.exists(model_filename):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    with st.spinner("Training LSTM model... This may take a few minutes ⏳"):
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

    model.save(model_filename)
    st.success(f"Model trained and saved as `{model_filename}` ✅")
else:
    model = load_model(model_filename)
    st.info(f"Loaded pre-trained model `{model_filename}` from local storage.")

# ---------------------------------------------
# MODEL PREDICTION AND EVALUATION
# ---------------------------------------------
st.subheader("📊 Step 4: Model Evaluation and Predictions")

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
rmse = math.sqrt(np.mean(((predicted_prices - actual_prices) ** 2)))
st.metric(label="Model RMSE", value=f"{rmse:.4f}")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=actual_prices.flatten(), mode='lines', name='Actual'))
fig2.add_trace(go.Scatter(y=predicted_prices.flatten(), mode='lines', name='Predicted'))
fig2.update_layout(title="Actual vs Predicted Prices", xaxis_title="Time", yaxis_title="Stock Price")
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------
# FUTURE PREDICTION
# ---------------------------------------------
st.subheader("🔮 Step 5: Future Stock Prediction")

future_days = st.slider("Days to Predict into the Future", 1, 30, 7)

last_60 = scaled_data[-look_back:]
predictions_future = []

for _ in range(future_days):
    X_test = np.array([last_60])
    X_test = np.reshape(X_test, (1, look_back, 1))
    pred_price = model.predict(X_test)
    predictions_future.append(pred_price[0][0])
    last_60 = np.append(last_60[1:], pred_price[0][0])
    last_60 = np.reshape(last_60, (look_back,))

predicted_future_prices = scaler.inverse_transform(np.array(predictions_future).reshape(-1, 1))
future_dates = pd.date_range(end_date, periods=future_days + 1, freq='B')[1:]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical'))
fig3.add_trace(go.Scatter(x=future_dates, y=predicted_future_prices.flatten(), mode='lines+markers', name='Future Prediction'))
fig3.update_layout(title="Future Price Prediction", xaxis_title="Date", yaxis_title="Predicted Price")
st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------
# FINBERT SENTIMENT ANALYSIS
# ---------------------------------------------
st.subheader("📰 Step 6: FinBERT Sentiment Analysis on Financial News")

text_input = st.text_area("Paste financial news headlines or analysis text here:")

if st.button("Analyze Sentiment"):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
    outputs = finbert_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    sentiment_index = torch.argmax(probs).item()
    confidence = probs[0][sentiment_index].item()

    st.success(f"Sentiment: **{sentiment_labels[sentiment_index]}** ({confidence*100:.2f}% confidence)")
    st.bar_chart(pd.DataFrame(probs.detach().numpy()[0], index=sentiment_labels, columns=["Score"]))

# ---------------------------------------------
# SUMMARY DASHBOARD
# ---------------------------------------------
st.subheader("📋 Step 7: Combined Sentiment + Price Summary")

st.markdown("""
This table combines the **predicted stock trend** (from LSTM) and **sentiment polarity**  
(from FinBERT) to give an overall **investment insight**.
""")

if "predicted_future_prices" in locals():
    trend = "📈 Uptrend" if predicted_future_prices[-1] > predicted_future_prices[0] else "📉 Downtrend"
    st.write(f"**Predicted Market Trend:** {trend}")

    if st.button("Generate Summary Insight"):
        st.info(f"Based on recent trends, {ticker} is showing **{trend}** with FinBERT sentiment leaning towards **{sentiment_labels[sentiment_index]}**.")
