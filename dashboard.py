"""
Bitcoin Price Prediction Dashboard
Hourly predictions with Binance API and Streamlit UI
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
import pickle
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
    color: #f7931a !important; /* Bitcoin orange */
    font-weight: 700 !important;
    text-shadow: 0 0 5px rgba(247, 147, 26, 0.5); /* subtle glow */
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
        color: white;
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
        scaler_path = os.path.join(base_dir, 'scaler.pkl')
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        
        config_path = os.path.join(base_dir, 'config.pkl')
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        
        return model, scaler, config
    except Exception as e:
        st.error(f"Model files not found! Error: {e}")
        return None, None
    # --- LSTM MODEL LOADING ---
    # try:
        
    #     #rf_model = joblib.load(r"fine tuned model\btc_prediction_model.pkl")
    #     model = load_model(r'fine tuned model\btc_prediction_model.h5')
    #     with open('fine tuned model\scaler.pkl', 'rb') as f:
    #         scaler = pickle.load(f)
    #     with open('fine tuned model\config.pkl', 'rb') as f:
    #         config = pickle.load(f)
    #     return model, scaler, config
    # except Exception as e:
    #     st.error(f" Model files not found! Please run 'python train_model.py' first.\n\nError: {e}")
    #     return None, None, None

# Get real-time price from Binance
@st.cache_data(ttl=10)  # Update every 10 seconds
def get_binance_price(): # for showing the currect live btc price metrics
    try:
        # url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
        url = "https://data-api.binance.vision/api/v3/ticker/24hr?symbol=BTCUSDT"
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
def get_binance_historical(hours=48): # getting full btc data from binance for chart
    try:
        #url = "https://api.binance.com/api/v3/klines"
        # https://data-api.binance.vision/api/v3/ticker/price?symbol=BTCUSDT
        url = "https://data-api.binance.vision/api/v3/klines"
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
def get_live_24_hours(_scaler): # only fetching close price from full data
    """
    Get current last 24 hours data and normalize it for prediction
    """
    try:
        # Get last 24 hours data
        hist = get_binance_historical(24) # for candlesticks chart getting full data
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

# Predict next hour price (ONLY FOR Random Forest) 
def predict_next_hour(model, scaler, last_24_hours):
    try:
        # Convert last 24 hours from (24,1) → (1,24)
        X_pred = last_24_hours.reshape(1, -1)

        # Predict (scaled value)
        pred_scaled = model.predict(X_pred).reshape(-1, 1)
        predicted_price = scaler.inverse_transform(pred_scaled)
        raw_pred = predicted_price[0][0]  # Model ki asli prediction

        # Get last actual known price and trend
        last_prices = scaler.inverse_transform(last_24_hours.reshape(-1, 1))
        last_price = last_prices[-1][0]
        
        # Calculate Recent Trend (Last 5 hours average movement)
        recent_trend_val = np.mean(np.diff(last_prices[-5:].flatten()))

        # --- VALIDATION LOGIC ---
        
        # Calculate percentage change
        change_pct = abs((raw_pred - last_price) / last_price * 100)
        # 1 ghantay mein 2% ($1800+) change hona bohat rare hai.
        LIMIT = 2.0 

        if change_pct > LIMIT:
            # FIX 2: Better Fallback Logic
            # if model predict too much change, use trend instead
            # but limit it to 0.5 weights
            
            conservative_pred = last_price + (recent_trend_val * 0.5)
            
            if abs((conservative_pred - last_price)/last_price*100) > LIMIT:
                 pred = last_price  # No change (Safest bet)
            else:
                 pred = conservative_pred
                 
        else:
            pred = raw_pred

        return pred

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None
   
def predict_multiple_hours(model, scaler, last_24_hours, hours=12):
    try:
        predictions = []
        current_sequence = last_24_hours.copy()

        # Convert last known price
        last_prices = scaler.inverse_transform(current_sequence.reshape(-1, 1))
        prev_price = last_prices[-1][0]

        for i in range(hours):
            # Shape (1,24)
            X_pred = current_sequence.reshape(1, -1)

            # Predict scaled
            next_pred_scaled = model.predict(X_pred).reshape(-1, 1)
            next_pred = scaler.inverse_transform(next_pred_scaled)[0][0]

            # Validation: ML models sometimes jump — limit to 3%
            change_pct = abs((next_pred - prev_price) / prev_price * 100)

            if change_pct > 3:
                recent_trend = np.mean(np.diff(last_prices[-5:].flatten()))
                next_pred = prev_price + recent_trend * 0.8

            # Store prediction
            predictions.append(next_pred)

            # Update window: normalize new prediction & append it
            next_pred_norm = scaler.transform([[next_pred]])[0][0]
            current_sequence = np.append(current_sequence[1:], next_pred_norm)

            # Update previous price
            prev_price = next_pred

        return predictions

    except Exception as e:
        st.error(f"RF Multi-Hour Prediction Error: {e}")
        return []

# Predict next hour price (ONLY FOR LSTM)
# def predict_next_hour(model, scaler, last_24_hours):
#     try:
#         X_pred = last_24_hours.reshape(1, 24, 1)
#         predicted_scaled = model.predict(X_pred, verbose=0)
#         predicted_price = scaler.inverse_transform(predicted_scaled)
#         pred = predicted_price[0][0]
        
#         # Validation: prediction should be within reasonable range
#         # Get last known price
#         last_prices = scaler.inverse_transform(last_24_hours.reshape(-1, 1))
#         last_price = last_prices[-1][0]
        
#         # Check if prediction is within +/- 5% (reasonable for 1 hour)
#         change_pct = abs((pred - last_price) / last_price * 100)
#         if change_pct > 5:  # More than 5% change in 1 hour is suspicious
#             # Return more conservative estimate
#             st.warning(f"Model prediction was {change_pct:.2f}% change - using conservative estimate")
#             # Use simple moving average trend instead
#             trend = np.mean(np.diff(last_prices[-5:].flatten()))
#             pred = last_price + trend
        
#         return pred
#     except Exception as e:
#         st.error(f"Prediction error: {e}")
#         return None

# # Predict multiple hours
# def predict_multiple_hours(model, scaler, last_24_hours, hours=12):
#     predictions = []
#     current_sequence = last_24_hours.copy()
    
#     # Get last actual price for validation
#     last_prices = scaler.inverse_transform(last_24_hours.reshape(-1, 1))
#     prev_price = last_prices[-1][0]
    
#     for i in range(hours):
#         X_pred = current_sequence.reshape(1, 24, 1)
#         next_pred_scaled = model.predict(X_pred, verbose=0)
#         next_pred = scaler.inverse_transform(next_pred_scaled)[0][0]
        
#         # Validation: each hour should not change more than 3% typically
#         change_pct = abs((next_pred - prev_price) / prev_price * 100)
#         if change_pct > 3:
#             # Use more conservative prediction based on recent trend
#             recent_trend = np.mean(np.diff(last_prices[-5:].flatten()))
#             next_pred = prev_price + recent_trend * 0.8  # 80% of trend
        
#         predictions.append(next_pred)
        
#         # Update sequence with normalized prediction
#         next_pred_normalized = scaler.transform([[next_pred]])[0][0]
#         current_sequence = np.append(current_sequence[1:], next_pred_normalized)
#         prev_price = next_pred
    
#     return predictions

# Main dashboard
def main():
    # Load model
    model, scaler, config = load_model_and_scaler()
    
    if model is None:
        st.stop()
    
    # Prepare live last 24 hours normalized sequence
    live_last_24 = get_live_24_hours(scaler)
    if live_last_24 is None or len(live_last_24) < 24:
        st.error("Not enough live data to make predictions. Try again in a minute.")
        st.stop()
    
    # Header with live indicator
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=130)
    with col2:
        st.title("₿ Bitcoin Price Prediction Dashboard")
        st.markdown("**💎 Real-Time Bitcoin Hourly Forecasting**")
    # with col_live:
    #     st.markdown("<br>", unsafe_allow_html=True)
    #     st.markdown("🟢 **LIVE**", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=100)
        st.header("⚙️ Settings")
        
        chart_hours = st.selectbox("📈 Chart Period", [24, 48, 72, 168], index=1, format_func=lambda x: f"{x} hours ({x//24} days)")
        
        prediction_hours = st.select_slider(
    "🔮 Predict Next Hours",
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
                "- Model: Random Forest Regressor\n"
                "- Model Accuracy: 99.53%\n"
                "- (MAPE): 0.47%\n"
                "- Input: Last 24 hours (Live)\n"
                "- Output: Next 1 hour (Prediction)\n"
                "- Data: Binance API\n")
                #"- Validation: Max 5% change/hr")
        
        st.markdown("---")
    
    # Get current price
    price_data = get_binance_price()
    
    if price_data is None:
        st.error("Unable to fetch live price. Check your internet connection.")
        st.stop()
    
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
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
            label="📈 24h High Price",
            value=f"${price_data['high_24h']:,.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="📉 24h Low Price",
            value=f"${price_data['low_24h']:,.2f}",
            delta=None
        )
    with col5:
        # Get predictions for next 12 hours
        future_predictions = predict_multiple_hours(model, scaler, live_last_24, hours=12)
        
        if future_predictions:
            # Current Price
            current_price = price_data['price']
            
            # Average of future prices (Trends check karne ke liye)
            avg_future_price = sum(future_predictions) / len(future_predictions)
            
            # Change calculation (%)
            pct_change = ((avg_future_price - current_price) / current_price) * 100
            
            # Threshold: 0.5% (Fees aur Slippage ko cover karne ke liye)
            THRESHOLD = 0.5
            
            # --- LOGIC FOR TARGET TIME & PROFIT ---
            
            if pct_change > THRESHOLD:
                status = "Buy (Bullish) 🟢"
                # Buy k liye: Maximum price dhoondo
                best_price = max(future_predictions)
                best_hour_index = future_predictions.index(best_price) + 1
                
                # NEW: Calculate Dollar Profit
                profit = best_price - current_price
                advice = f"Target: {best_price:,.0f} in {best_hour_index} Hrs (Exp. Profit: +${profit:,.0f})"
                
            elif pct_change < -THRESHOLD:
                status = "Sell (Bearish) 🔴"
                # Sell k liye: Minimum price dhoondo
                best_price = min(future_predictions)
                best_hour_index = future_predictions.index(best_price) + 1
                
                # NEW: Calculate Dollar Drop
                drop = current_price - best_price
                advice = f"Bottom: {best_price:,.0f} in {best_hour_index} Hrs (Exp. Drop: -${drop:,.0f})"
                
            else:
                status = "Hold 🟡"
                # Updated advice for Presentation logic
                advice = "Market is choppy/flat (Risk > Reward)"
            
            # --- DISPLAY ---
            st.metric(
                label="🚦 12-Hour Outlook",
                value=status,
                delta=f"{pct_change:.2f}% exp. move"
            )
            # Signal ke neeche chota sa Advice note
            st.caption(f"💡 **Strategy:** {advice}")

        else:
             st.metric(label="🚦 Signal", value="Analyzing...")
    st.markdown("---")
    
    # Main content area
    col_chart, col_predictions = st.columns([2.5, 1])
    
    with col_chart:
        st.subheader(f"📈 BTC-USD Price Chart ({chart_hours} hours)")
        
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
        st.subheader(f"🔮 Next {prediction_hours} hours BTC Predictions")
        
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
    
    # with col_f1:
    #     st.caption(f"⏰ Last updated: {price_data['time'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    with col_f2:
        st.caption(f"⏰ Last updated: {price_data['time'].strftime('%Y-%m-%d %H:%M:%S')}")
        #st.caption(f"📊 24h Volume: {price_data['volume_24h']:,.2f} BTC") %H:%M:%S')}")
    
    # with col_f3:
    #     st.caption(f"📊 24h Volume: {price_data['volume_24h']:,.2f} BTC")
    
    
    st.caption("⚠️ Disclaimer: This application provides predictive analytics for informational and educational purposes only. The forecasts generated are probabilistic estimates and may not accurately reflect future market movements. Cryptocurrency markets are highly volatile and unpredictable, and past performance does not guarantee future results.")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(10)
        st.rerun()

if __name__ == "__main__":
    main()
