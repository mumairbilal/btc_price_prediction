"""
Bitcoin Price Prediction Dashboard
Hourly predictions with Binance API and Beautiful UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime, timedelta
import requests
import time
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="BTC Price Predictor - Hourly",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Metrics styling */
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Headers */
    h1 {
        color: #00d4ff !important;
        font-weight: 700 !important;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }
    
    h2, h3 {
        color: #ffffff !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(15, 12, 41, 0.95);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border-left: 4px solid #00d4ff;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        #base_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "Random Forest Regressor", "rf_btc_prediction_model.pkl")
        
        model = joblib.load(model_path)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)   # consistent naam
        with open('fine tuned model\config.pkl', 'rb') as f:
            config = pickle.load(f)
        return model, scaler, config
    except Exception as e:
        st.error(f"Model files not found! Error: {e}")
        return None, None

# Get real-time price from Binance
@st.cache_data(ttl=10)  # Update every 10 seconds
def get_binance_price():
    try:
        
        response = requests.get(url, params=params)
        response.raise_for_status()  # yeh add karein
        url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        current_price = float(data['lastPrice'])
        price_change = float(data['priceChange'])
        price_change_percent = float(data['priceChangePercent'])
        high_24h = float(data['highPrice'])
        low_24h = float(data['lowPrice'])
        volume_24h = float(data['volume'])
        
        return {
            'price': current_price,
            'change': price_change,
            'change_percent': price_change_percent,
            'high_24h': high_24h,
            'low_24h': low_24h,
            'volume_24h': volume_24h,
            'time': datetime.now()
        }
    except Exception as e:
        st.error(f"Error fetching price: {e}")
        return None

# Get historical hourly data from Binance
@st.cache_data(ttl=60)  # Cache for 1 minute (more frequent updates)
def get_binance_historical(hours=48):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': 'BTCUSDT',
            'interval': '1h',
            'limit': hours
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        return df
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()

# Get live last 24 hours for predictions
@st.cache_data(ttl=60)  # Update every minute
def get_live_24_hours(_scaler):
    """
    Get current last 24 hours data and normalize it for prediction
    """
    try:
        # Get last 24 hours data
        hist = get_binance_historical(24)
        if hist.empty:
            return None
        
        # Extract close prices
        close_prices = hist['close'].values.reshape(-1, 1)
        
        # Normalize using the same scaler
        scaled_prices = _scaler.transform(close_prices)
        
        return scaled_prices.flatten()
    except Exception as e:
        st.error(f"Error preparing live data: {e}")
        return None

# Predict next hour price
def predict_next_hour(model, scaler, last_24_hours):
    try:
        X_pred = last_24_hours.reshape(1, 24, 1)
        predicted_scaled = model.predict(X_pred, verbose=0)
        predicted_price = scaler.inverse_transform(predicted_scaled)
        pred = predicted_price[0][0]
        
        # Validation: prediction should be within reasonable range
        # Get last known price
        last_prices = scaler.inverse_transform(last_24_hours.reshape(-1, 1))
        last_price = last_prices[-1][0]
        
        # Check if prediction is within +/- 5% (reasonable for 1 hour)
        change_pct = abs((pred - last_price) / last_price * 100)
        if change_pct > 5:  # More than 5% change in 1 hour is suspicious
            # Return more conservative estimate
            st.warning(f"⚠️ Model prediction was {change_pct:.2f}% change - using conservative estimate")
            # Use simple moving average trend instead
            trend = np.mean(np.diff(last_prices[-5:].flatten()))
            pred = last_price + trend
        
        return pred
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Predict multiple hours
def predict_multiple_hours(model, scaler, last_24_hours, hours=12):
    predictions = []
    current_sequence = last_24_hours.copy()
    
    # Get last actual price for validation
    last_prices = scaler.inverse_transform(last_24_hours.reshape(-1, 1))
    prev_price = last_prices[-1][0]
    
    for i in range(hours):
        X_pred = current_sequence.reshape(1, 24, 1)
        next_pred_scaled = model.predict(X_pred, verbose=0)
        next_pred = scaler.inverse_transform(next_pred_scaled)[0][0]
        
        # Validation: each hour should not change more than 3% typically
        change_pct = abs((next_pred - prev_price) / prev_price * 100)
        if change_pct > 3:
            # Use more conservative prediction based on recent trend
            recent_trend = np.mean(np.diff(last_prices[-5:].flatten()))
            next_pred = prev_price + recent_trend * 0.8  # 80% of trend
        
        predictions.append(next_pred)
        
        # Update sequence with normalized prediction
        next_pred_normalized = scaler.transform([[next_pred]])[0][0]
        current_sequence = np.append(current_sequence[1:], next_pred_normalized)
        prev_price = next_pred
    
    return predictions

# Main dashboard
def main():
    # Load model
    model, scaler = load_model_and_scaler()
    
    if model is None:
        st.stop()
    
    # Prepare live last 24 hours normalized sequence
    live_last_24 = get_live_24_hours(scaler)
    if live_last_24 is None or len(live_last_24) < 24:
        st.error("Not enough live data to make predictions. Try again in a minute.")
        st.stop()
    
    # Header with live indicator
    col_title, col_live = st.columns([4, 1])
    with col_title:
        st.title("₿ Bitcoin Price Predictor")
        st.markdown("**Hourly Predictions with AI**")
    with col_live:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("🟢 **LIVE**", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=100)
        st.header("⚙️ Settings")
        
        chart_hours = st.selectbox("📊 Chart Period", [24, 48, 72, 168], index=1, format_func=lambda x: f"{x} hours ({x//24} days)")
        
        prediction_hours = st.select_slider(
    "🔮 Predict Next",
    options=list(range(1, 25)),
    value=6,
    format_func=lambda x: f"{x} hours"
)
        st.markdown("---")
        
        auto_refresh = st.checkbox("🔄 Auto Refresh (10s)", value=True)
        
        if st.button("🔃 Manual Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.info("**🤖 Model Info**\n\n"
                "- Type: LSTM Neural Network\n"
                "- Input: Last 24 hours (live)\n"
                "- Output: Next 1 hour\n"
                "- Data: Binance API\n"
                "- Validation: Max 5% change/hr")
        
        st.markdown("---")
        st.caption("Made for BIA Lahore 🎓")
    
    # Get current price
    price_data = get_binance_price()
    
    if price_data is None:
        st.error("Unable to fetch live price. Check your internet connection.")
        st.stop()
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Current BTC Price",
            value=f"${price_data['price']:,.2f}",
            delta=f"{price_data['change_percent']:.2f}%"
        )
    
    with col2:
        next_hour_pred = predict_next_hour(model, scaler, live_last_24)
        if next_hour_pred:
            pred_change = ((next_hour_pred - price_data['price']) / price_data['price'] * 100)
            st.metric(
                label="🔮 Next Hour Prediction",
                value=f"${next_hour_pred:,.2f}",
                delta=f"{pred_change:+.2f}%"
            )
    
    with col3:
        st.metric(
            label="📈 24h High",
            value=f"${price_data['high_24h']:,.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="📉 24h Low",
            value=f"${price_data['low_24h']:,.2f}",
            delta=None
        )
    
    st.markdown("---")
    
    # Main content area
    col_chart, col_predictions = st.columns([2.5, 1])
    
    with col_chart:
        st.subheader("📊 Price Chart (Hourly)")
        
        # Get historical data
        hist_data = get_binance_historical(chart_hours)
        
        if not hist_data.empty:
            # Create candlestick chart
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=('BTC-USDT Price', 'Volume'),
                vertical_spacing=0.05
            )
            
            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=hist_data['timestamp'],
                    open=hist_data['open'],
                    high=hist_data['high'],
                    low=hist_data['low'],
                    close=hist_data['close'],
                    name='BTCUSDT',
                    increasing_line_color='#00ff88',
                    decreasing_line_color='#ff4444'
                ),
                row=1, col=1
            )
            
            # Volume bars
            colors = ['#00ff88' if hist_data['close'].iloc[i] >= hist_data['open'].iloc[i] else '#ff4444' 
                     for i in range(len(hist_data))]
            
            fig.add_trace(
                go.Bar(
                    x=hist_data['timestamp'],
                    y=hist_data['volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.5
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                template='plotly_dark',
                height=600,
                showlegend=False,
                xaxis_rangeslider_visible=False,
                plot_bgcolor='rgba(0,0,0,0.3)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Unable to load historical data")
    
    with col_predictions:
        st.subheader("🔮 AI Predictions")
        
        # Get predictions
        predictions = predict_multiple_hours(model, scaler, live_last_24, prediction_hours)
        
        if predictions:
            current_time = datetime.now()
            
            st.markdown(f"**Starting from:** {current_time.strftime('%H:%M')}")
            st.markdown("---")
            
            for i, pred in enumerate(predictions, 1):
                future_time = current_time + timedelta(hours=i)
                diff = pred - price_data['price']
                diff_pct = (diff / price_data['price']) * 100
                
                color = "🟢" if diff > 0 else "🔴" if diff < 0 else "⚪"
                
                with st.container():
                    st.markdown(f"""
                    <div style='background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; margin-bottom: 8px;'>
                        <div style='display: flex; justify-content: space-between;'>
                            <span>{color} <b>{future_time.strftime('%H:%M')}</b></span>
                            <span><b>${pred:,.2f}</b></span>
                        </div>
                        <div style='font-size: 0.8em; color: {"#00ff88" if diff > 0 else "#ff4444"};'>
                            {diff:+,.2f} ({diff_pct:+.2f}%)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Prediction trend chart
    st.subheader("📈 Prediction Trend")
    
    if predictions and not hist_data.empty:
        # Get last N hours for context
        context_hours = min(24, len(hist_data))
        historical_prices = hist_data['close'].tail(context_hours).values
        historical_times = hist_data['timestamp'].tail(context_hours).values
        
        # Future times
        future_times = [datetime.now() + timedelta(hours=i) for i in range(1, len(predictions)+1)]
        
        fig_trend = go.Figure()
        
        # Historical line
        fig_trend.add_trace(go.Scatter(
            x=historical_times,
            y=historical_prices,
            mode='lines',
            name='Historical',
            line=dict(color='#00d4ff', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.1)'
        ))
        
        # Current price marker
        fig_trend.add_trace(go.Scatter(
            x=[datetime.now()],
            y=[price_data['price']],
            mode='markers',
            name='Current',
            marker=dict(size=12, color='#ffff00', symbol='star')
        ))
        
        # Prediction line
        fig_trend.add_trace(go.Scatter(
            x=future_times,
            y=predictions,
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#ff6b6b', width=3, dash='dash'),
            marker=dict(size=8, color='#ff6b6b')
        ))
        
        fig_trend.update_layout(
            template='plotly_dark',
            height=400,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0.3)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Footer
    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.caption(f"⏰ Last updated: {price_data['time'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    with col_f2:
        st.caption(f"📊 24h Volume: {price_data['volume_24h']:,.2f} BTC")
    
    with col_f3:
        st.caption("⚠️ For educational purposes only")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(10)
        st.rerun()

if __name__ == "__main__":
    main()
