"""
Bitcoin Price Analysis & Forecasting Dashboard
Interactive Streamlit application for cryptocurrency time series analysis
with REAL-TIME data updates every 1 minute
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import time
import yfinance as yf
warnings.filterwarnings('ignore')

# Time series libraries
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Deep learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Prophet
from prophet import Prophet

# Page configuration
st.set_page_config(
    page_title="Bitcoin Price Forecasting Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130;
        border-radius: 5px;
        padding: 10px 20px;
    }
    h1, h2, h3 {
        color: #f8a300;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("₿ Bitcoin Real-Time Forecasting Dashboard")
st.markdown("### 🔴 LIVE: Auto-updates every 60 seconds | 7-Day Predictions")

# Display last update time
if 'last_update' not in st.session_state:
    st.session_state['last_update'] = datetime.now()

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown(f"**Last Updated:** {st.session_state['last_update'].strftime('%Y-%m-%d %H:%M:%S')}")
with col2:
    if st.button("🔄 Refresh Now"):
        st.rerun()
with col3:
    auto_refresh = st.checkbox("Auto-Refresh", value=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=100)
    st.title("⚙️ Settings")
    
    # Data source selection
    st.subheader("📊 Data Source")
    data_source = st.radio(
        "Select Data Source",
        ["🔴 Real-Time (Live)", "📁 Upload CSV"],
        index=0
    )
    
    if data_source == "📁 Upload CSV":
        uploaded_file = st.file_uploader("Upload Bitcoin CSV", type=['csv'])
    else:
        uploaded_file = None
        st.info("🔴 Fetching live data from Yahoo Finance")
        
        # Historical data range
        data_period = st.selectbox(
            "Historical Data Range",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            index=2
        )
    
    # Model parameters
    st.subheader("Model Parameters")
    train_split = st.slider("Train/Test Split (%)", 60, 90, 80, 5)
    
    # Fixed to 7 days for weekly prediction
    st.markdown("**Forecast Period:** 7 Days (1 Week)")
    forecast_days = 7
    
    # ARIMA parameters
    with st.expander("ARIMA Parameters"):
        p = st.number_input("p (AR order)", 1, 10, 5)
        d = st.number_input("d (differencing)", 0, 2, 1)
        q = st.number_input("q (MA order)", 1, 10, 2)
    
    # LSTM parameters
    with st.expander("LSTM Parameters"):
        sequence_length = st.number_input("Sequence Length", 30, 120, 60, 10)
        lstm_epochs = st.number_input("Epochs", 10, 100, 50, 10)
        batch_size = st.number_input("Batch Size", 16, 64, 32, 16)
    
    st.markdown("---")
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Helper functions
@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_realtime_data(period="6mo"):
    """Fetch real-time Bitcoin data from Yahoo Finance"""
    
    # Try multiple Bitcoin tickers
    tickers_to_try = [
        ("BTC-USD", "Bitcoin USD"),
        ("BTC-EUR", "Bitcoin EUR"),
        ("BTCUSD=X", "Bitcoin USD (Forex)"),
    ]
    
    for ticker, name in tickers_to_try:
        try:
            st.info(f"🔄 Trying {name} ({ticker})...")
            
            # Calculate date range - ensure we don't go into future
            end_date = datetime.now() - timedelta(days=1)  # Yesterday
            
            if period == "1mo":
                start_date = end_date - timedelta(days=30)
            elif period == "3mo":
                start_date = end_date - timedelta(days=90)
            elif period == "6mo":
                start_date = end_date - timedelta(days=180)
            elif period == "1y":
                start_date = end_date - timedelta(days=365)
            else:  # 2y
                start_date = end_date - timedelta(days=730)
            
            # Try downloading with period first (most reliable)
            df = yf.download(
                ticker, 
                period=period,
                progress=False,
                interval='1d',
                timeout=10
            )
            
            if not df.empty:
                st.success(f"✅ Successfully fetched data using {name}")
                break
                
        except Exception as e:
            st.warning(f"Failed with {name}: {str(e)[:100]}")
            continue
    
    if df.empty:
        st.error("❌ All ticker symbols failed")
        st.warning("Yahoo Finance API may be experiencing issues. Using local CSV data instead.")
        return None
    
    try:
        # Clean and prepare data
        df.index = pd.to_datetime(df.index)
        
        # Handle MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
            return None
        
        df = df[required_cols]
        
        # Add Price column (same as Close)
        df['Price'] = df['Close']
        
        # Remove any NaN values
        df = df.dropna()
        
        st.session_state['last_update'] = datetime.now()
        st.success(f"✅ Loaded {len(df)} days | Latest: {df.index[-1].strftime('%Y-%m-%d')} | Price: ${df['Close'].iloc[-1]:,.2f}")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error processing data: {type(e).__name__}: {str(e)}")
        return None

def load_data(file_path=None):
    """Load and preprocess Bitcoin data from CSV"""
    try:
        if file_path:
            df = pd.read_csv(file_path, header=[0, 1], index_col=0)
        else:
            df = pd.read_csv('BTC_1Jan2025_to_30Nov2025 (1).csv', header=[0, 1], index_col=0)
        
        df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df = df[df.index != 'Ticker']
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def calculate_metrics(actual, predicted):
    """Calculate performance metrics"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)
    return mae, rmse, mape, r2

def create_price_chart(df, title="Bitcoin Price History"):
    """Create interactive price chart with Plotly"""
    fig = go.Figure()
    
    # Add area under curve
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#f8a300', width=2),
        fill='tozeroy',
        fillcolor='rgba(248, 163, 0, 0.1)',
        hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: $%{y:,.2f}<extra></extra>'
    ))
    
    # Add current price annotation
    current_price = df['Close'].iloc[-1]
    fig.add_annotation(
        x=df.index[-1],
        y=current_price,
        text=f"🔴 ${current_price:,.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#f8a300",
        bgcolor="#1e2130",
        bordercolor="#f8a300",
        borderwidth=2,
        font=dict(size=14, color="white")
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_candlestick_chart(df):
    """Create candlestick chart"""
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='#00d09c',
        decreasing_line_color='#e63946'
    )])
    
    fig.update_layout(
        title="Bitcoin OHLC Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_volume_chart(df):
    """Create volume bar chart"""
    colors = ['#00d09c' if close >= open_ else '#e63946' 
              for close, open_ in zip(df['Close'], df['Open'])]
    
    fig = go.Figure(data=[go.Bar(
        x=df.index,
        y=df['Volume'],
        marker_color=colors,
        name='Volume'
    )])
    
    fig.update_layout(
        title="Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        template="plotly_dark",
        height=400
    )
    
    return fig

def train_arima_model(train_data, order, test_size):
    """Train ARIMA model"""
    with st.spinner("Training ARIMA model..."):
        model = ARIMA(train_data['Close'], order=order)
        fit = model.fit()
        predictions = fit.forecast(steps=test_size)
        return fit, predictions

def train_lstm_model(df_ts, train_size, seq_length, epochs, batch_size_param):
    """Train LSTM model"""
    with st.spinner("Training LSTM model... This may take a few minutes."):
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_ts[['Close']].values)
        
        # Create sequences
        X, y = [], []
        for i in range(seq_length, len(scaled_data)):
            X.append(scaled_data[i-seq_length:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        
        # Split
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Reshape
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build model
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size_param,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0
        )
        
        # Predictions
        train_pred = scaler.inverse_transform(model.predict(X_train, verbose=0))
        test_pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
        
        return model, scaler, train_pred, test_pred, history

def train_prophet_model(df_ts, train_size):
    """Train Prophet model"""
    with st.spinner("Training Prophet model..."):
        prophet_data = df_ts.reset_index()
        prophet_data.columns = ['ds', 'y']
        
        prophet_train = prophet_data.iloc[:train_size].copy()
        prophet_test = prophet_data.iloc[train_size:].copy()
        
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10
        )
        
        model.fit(prophet_train)
        forecast = model.predict(prophet_test[['ds']])
        
        return model, forecast, prophet_train, prophet_test

def create_forecast_comparison_chart(df_ts, arima_forecast, lstm_forecast, prophet_forecast, forecast_dates):
    """Create comparison chart for all three models"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df_ts.index, y=df_ts['Close'],
        mode='lines',
        name='Historical Data',
        line=dict(color='#2E86AB', width=2)
    ))
    
    # ARIMA
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=arima_forecast,
        mode='lines+markers',
        name='ARIMA',
        line=dict(color='#3498DB', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # LSTM
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=lstm_forecast.flatten(),
        mode='lines+markers',
        name='LSTM',
        line=dict(color='#E74C3C', width=2, dash='dash'),
        marker=dict(size=6, symbol='square')
    ))
    
    # Prophet
    fig.add_trace(go.Scatter(
        x=prophet_forecast['ds'], y=prophet_forecast['yhat'],
        mode='lines+markers',
        name='Prophet',
        line=dict(color='#2ECC71', width=2, dash='dash'),
        marker=dict(size=6, symbol='triangle-up')
    ))
    
    # Prophet confidence interval
    fig.add_trace(go.Scatter(
        x=prophet_forecast['ds'], y=prophet_forecast['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=prophet_forecast['ds'], y=prophet_forecast['yhat_lower'],
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(46, 204, 113, 0.2)',
        fill='tonexty',
        name='Prophet Confidence',
        hoverinfo='skip'
    ))
    
    # Add vertical line at forecast start using shape (more reliable than add_vline)
    forecast_start_date = df_ts.index[-1]
    
    fig.add_shape(
        type="line",
        x0=forecast_start_date,
        x1=forecast_start_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    
    fig.add_annotation(
        x=forecast_start_date,
        y=1,
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        yshift=10,
        font=dict(color="red")
    )
    
    fig.update_layout(
        title=f"{forecast_days}-Day Forecast Comparison: ARIMA vs LSTM vs Prophet",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        hovermode='x unified',
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

# Main application
if data_source == "🔴 Real-Time (Live)":
    df = load_realtime_data(period=data_period)
    
    # Fallback to local CSV if real-time fails
    if df is None:
        st.warning("⚠️ Falling back to local CSV file...")
        df = load_data()
        if df is not None:
            st.info("✅ Using local CSV data instead")
        
elif uploaded_file:
    df = load_data(uploaded_file)
else:
    df = load_data()

if df is not None:
    df_ts = df[['Close']].copy().dropna()
    
    # Overview section
    st.header("📊 Market Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    current_price = df_ts['Close'].iloc[-1]
    prev_price = df_ts['Close'].iloc[-2]
    price_change_24h = ((current_price / prev_price) - 1) * 100
    
    with col1:
        st.metric(
            "🔴 LIVE Price",
            f"${current_price:,.2f}",
            f"{price_change_24h:+.2f}%",
            delta_color="normal"
        )
    
    with col2:
        st.metric("📈 24h High", f"${df['High'].iloc[-1]:,.2f}")
    
    with col3:
        st.metric("📉 24h Low", f"${df['Low'].iloc[-1]:,.2f}")
    
    with col4:
        price_change = df_ts['Close'].iloc[-1] - df_ts['Close'].iloc[0]
        pct_change = (price_change / df_ts['Close'].iloc[0]) * 100
        st.metric("📊 Period Change", f"${price_change:,.2f}", f"{pct_change:+.2f}%")
    
    with col5:
        st.metric("💹 Avg Volume", f"{df['Volume'].mean()/1e9:.2f}B")
    
    st.markdown("---")
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Price Analysis", 
        "🤖 ARIMA Model", 
        "🧠 LSTM Model", 
        "🔮 Prophet Model",
        "🏆 Model Comparison"
    ])
    
    # Tab 1: Price Analysis
    with tab1:
        st.subheader("Price History & Technical Analysis")
        
        # Price chart
        st.plotly_chart(create_price_chart(df_ts), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Candlestick chart
            st.plotly_chart(create_candlestick_chart(df), use_container_width=True)
        
        with col2:
            # Volume chart
            st.plotly_chart(create_volume_chart(df), use_container_width=True)
        
        # Statistical summary
        st.subheader("📊 Statistical Summary")
        
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
            'Value': [
                f"${df_ts['Close'].mean():,.2f}",
                f"${df_ts['Close'].median():,.2f}",
                f"${df_ts['Close'].std():,.2f}",
                f"${df_ts['Close'].min():,.2f}",
                f"${df_ts['Close'].max():,.2f}",
                f"${df_ts['Close'].max() - df_ts['Close'].min():,.2f}"
            ]
        })
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
        
        with col2:
            # Distribution plot
            fig = px.histogram(
                df_ts, x='Close',
                nbins=50,
                title="Price Distribution",
                template="plotly_dark"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Run analysis when button is clicked
    if run_analysis:
        train_size_rows = int(len(df_ts) * (train_split / 100))
        train_data = df_ts.iloc[:train_size_rows].copy()
        test_data = df_ts.iloc[train_size_rows:].copy()
        
        # Tab 2: ARIMA
        with tab2:
            st.subheader("ARIMA Model Results")
            
            try:
                # Train ARIMA
                arima_order = (p, d, q)
                arima_fit, arima_predictions = train_arima_model(
                    train_data, arima_order, len(test_data)
                )
                
                # Metrics
                arima_mae, arima_rmse, arima_mape, arima_r2 = calculate_metrics(
                    test_data['Close'].values, arima_predictions
                )
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"${arima_mae:,.2f}")
                col2.metric("RMSE", f"${arima_rmse:,.2f}")
                col3.metric("MAPE", f"{arima_mape:.2f}%")
                col4.metric("R² Score", f"{arima_r2:.4f}")
                
                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=train_data.index, y=train_data['Close'],
                    mode='lines', name='Training Data',
                    line=dict(color='#2E86AB', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=test_data.index, y=test_data['Close'],
                    mode='lines', name='Actual Test Data',
                    line=dict(color='#E63946', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=test_data.index, y=arima_predictions,
                    mode='lines', name='ARIMA Predictions',
                    line=dict(color='#F77F00', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title="ARIMA Model: Actual vs Predicted",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    template="plotly_dark",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Future forecast
                st.subheader(f"📅 {forecast_days}-Day Future Forecast")
                future_forecast = arima_fit.forecast(steps=len(test_data) + forecast_days)
                future_dates = pd.date_range(
                    start=df_ts.index[-1] + timedelta(days=1),
                    periods=forecast_days,
                    freq='D'
                )
                arima_future = future_forecast[-forecast_days:]
                
                # Calculate 7-day targets
                day_7_price = arima_future.iloc[-1]
                current_btc_price = df_ts['Close'].iloc[-1]
                week_change = day_7_price - current_btc_price
                week_change_pct = (week_change / current_btc_price) * 100
                
                # Display 7-day prediction
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${current_btc_price:,.2f}")
                with col2:
                    st.metric(
                        "7-Day Prediction",
                        f"${day_7_price:,.2f}",
                        f"{week_change_pct:+.2f}%"
                    )
                with col3:
                    direction = "📈 BULLISH" if week_change > 0 else "📉 BEARISH"
                    st.metric("Trend", direction)
                
                # Store for comparison
                st.session_state['arima_future'] = arima_future
                st.session_state['arima_metrics'] = (arima_mae, arima_rmse, arima_mape, arima_r2)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_ts.index, y=df_ts['Close'],
                    mode='lines', name='Historical Data',
                    line=dict(color='#2E86AB', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=future_dates, y=arima_future,
                    mode='lines+markers', name='ARIMA Forecast',
                    line=dict(color='#06A77D', width=2),
                    marker=dict(size=6)
                ))
                
                # Add vertical line at forecast start using shape
                forecast_start = df_ts.index[-1]
                fig.add_shape(
                    type="line",
                    x0=forecast_start, x1=forecast_start,
                    y0=0, y1=1, yref="paper",
                    line=dict(color="red", width=2, dash="dash")
                )
                
                fig.update_layout(
                    title=f"ARIMA: {forecast_days}-Day Future Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("✓ ARIMA model training complete!")
                
            except Exception as e:
                st.error(f"Error training ARIMA model: {e}")
        
        # Tab 3: LSTM
        with tab3:
            st.subheader("LSTM Model Results")
            
            try:
                # Train LSTM
                lstm_train_size = int((len(df_ts) - sequence_length) * (train_split / 100))
                lstm_model, scaler, lstm_train_pred, lstm_test_pred, history = train_lstm_model(
                    df_ts, lstm_train_size, sequence_length, lstm_epochs, batch_size
                )
                
                # Metrics
                y_test_actual = df_ts['Close'].iloc[lstm_train_size+sequence_length:].values
                lstm_mae, lstm_rmse, lstm_mape, lstm_r2 = calculate_metrics(
                    y_test_actual, lstm_test_pred.flatten()
                )
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"${lstm_mae:,.2f}")
                col2.metric("RMSE", f"${lstm_rmse:,.2f}")
                col3.metric("MAPE", f"{lstm_mape:.2f}%")
                col4.metric("R² Score", f"{lstm_r2:.4f}")
                
                # Training history
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=history.history['loss'],
                        mode='lines',
                        name='Training Loss',
                        line=dict(color='#3498DB', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        y=history.history['val_loss'],
                        mode='lines',
                        name='Validation Loss',
                        line=dict(color='#E74C3C', width=2)
                    ))
                    fig.update_layout(
                        title="Training History",
                        xaxis_title="Epoch",
                        yaxis_title="Loss",
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Future forecast
                with col2:
                    st.markdown("### 🎯 Forecast Generation")
                    st.info(f"""
                    - **Sequence Length**: {sequence_length} days
                    - **Training Epochs**: {len(history.history['loss'])}
                    - **Final Loss**: {history.history['loss'][-1]:.6f}
                    - **Architecture**: 3 LSTM layers + Dropout
                    """)
                
                # Generate future predictions
                scaled_data = scaler.transform(df_ts[['Close']].values)
                last_sequence = scaled_data[-sequence_length:]
                lstm_future_predictions = []
                
                current_sequence = last_sequence.copy()
                for _ in range(forecast_days):
                    current_input = current_sequence.reshape((1, sequence_length, 1))
                    next_pred = lstm_model.predict(current_input, verbose=0)
                    lstm_future_predictions.append(next_pred[0, 0])
                    current_sequence = np.append(current_sequence[1:], next_pred[0])
                
                lstm_future = scaler.inverse_transform(
                    np.array(lstm_future_predictions).reshape(-1, 1)
                )
                
                # Store for comparison
                st.session_state['lstm_future'] = lstm_future
                st.session_state['lstm_metrics'] = (lstm_mae, lstm_rmse, lstm_mape, lstm_r2)
                
                # Calculate 7-day targets
                day_7_price = lstm_future[-1][0]
                current_btc_price = df_ts['Close'].iloc[-1]
                week_change = day_7_price - current_btc_price
                week_change_pct = (week_change / current_btc_price) * 100
                
                st.subheader(f"📅 7-Day Future Forecast")
                
                # Display 7-day prediction
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${current_btc_price:,.2f}")
                with col2:
                    st.metric(
                        "7-Day Prediction",
                        f"${day_7_price:,.2f}",
                        f"{week_change_pct:+.2f}%"
                    )
                with col3:
                    direction = "📈 BULLISH" if week_change > 0 else "📉 BEARISH"
                    st.metric("Trend", direction)
                
                future_dates = pd.date_range(
                    start=df_ts.index[-1] + timedelta(days=1),
                    periods=forecast_days,
                    freq='D'
                )
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_ts.index, y=df_ts['Close'],
                    mode='lines', name='Historical Data',
                    line=dict(color='#2E86AB', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=future_dates, y=lstm_future.flatten(),
                    mode='lines+markers', name='LSTM Forecast',
                    line=dict(color='#E63946', width=2),
                    marker=dict(size=6, symbol='square')
                ))
                
                # Add vertical line at forecast start using shape
                forecast_start = df_ts.index[-1]
                fig.add_shape(
                    type="line",
                    x0=forecast_start, x1=forecast_start,
                    y0=0, y1=1, yref="paper",
                    line=dict(color="red", width=2, dash="dash")
                )
                
                fig.update_layout(
                    title=f"LSTM: {forecast_days}-Day Future Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("✓ LSTM model training complete!")
                
            except Exception as e:
                st.error(f"Error training LSTM model: {e}")
        
        # Tab 4: Prophet
        with tab4:
            st.subheader("Facebook Prophet Model Results")
            
            try:
                # Train Prophet
                prophet_model, prophet_forecast, prophet_train, prophet_test = train_prophet_model(
                    df_ts, train_size_rows
                )
                
                # Metrics
                prophet_mae, prophet_rmse, prophet_mape, prophet_r2 = calculate_metrics(
                    prophet_test['y'].values, prophet_forecast['yhat'].values
                )
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"${prophet_mae:,.2f}")
                col2.metric("RMSE", f"${prophet_rmse:,.2f}")
                col3.metric("MAPE", f"{prophet_mape:.2f}%")
                col4.metric("R² Score", f"{prophet_r2:.4f}")
                
                # Predictions
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=prophet_train['ds'], y=prophet_train['y'],
                    mode='lines', name='Training Data',
                    line=dict(color='#2E86AB', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=prophet_test['ds'], y=prophet_test['y'],
                    mode='lines', name='Actual Test Data',
                    line=dict(color='#E63946', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=prophet_forecast['ds'], y=prophet_forecast['yhat'],
                    mode='lines', name='Prophet Predictions',
                    line=dict(color='#06A77D', width=2, dash='dash')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=prophet_forecast['ds'], y=prophet_forecast['yhat_upper'],
                    mode='lines', line=dict(width=0),
                    showlegend=False, hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=prophet_forecast['ds'], y=prophet_forecast['yhat_lower'],
                    mode='lines', line=dict(width=0),
                    fillcolor='rgba(6, 167, 125, 0.2)',
                    fill='tonexty', name='Confidence Interval',
                    hoverinfo='skip'
                ))
                
                fig.update_layout(
                    title="Prophet Model: Actual vs Predicted",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    template="plotly_dark",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Future forecast
                st.subheader(f"📅 {forecast_days}-Day Future Forecast")
                
                future = prophet_model.make_future_dataframe(periods=forecast_days, freq='D')
                prophet_future_forecast = prophet_model.predict(future)
                prophet_future = prophet_future_forecast.tail(forecast_days)
                
                # Calculate 7-day targets
                day_7_price = prophet_future['yhat'].iloc[-1]
                day_7_lower = prophet_future['yhat_lower'].iloc[-1]
                day_7_upper = prophet_future['yhat_upper'].iloc[-1]
                current_btc_price = df_ts['Close'].iloc[-1]
                week_change = day_7_price - current_btc_price
                week_change_pct = (week_change / current_btc_price) * 100
                
                # Display 7-day prediction with confidence
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${current_btc_price:,.2f}")
                with col2:
                    st.metric(
                        "7-Day Prediction",
                        f"${day_7_price:,.2f}",
                        f"{week_change_pct:+.2f}%"
                    )
                with col3:
                    st.metric("Best Case", f"${day_7_upper:,.2f}")
                with col4:
                    st.metric("Worst Case", f"${day_7_lower:,.2f}")
                
                # Store for comparison
                st.session_state['prophet_future'] = prophet_future
                st.session_state['prophet_metrics'] = (prophet_mae, prophet_rmse, prophet_mape, prophet_r2)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_ts.index, y=df_ts['Close'],
                    mode='lines', name='Historical Data',
                    line=dict(color='#2E86AB', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=prophet_future['ds'], y=prophet_future['yhat'],
                    mode='lines+markers', name='Prophet Forecast',
                    line=dict(color='#9B59B6', width=2),
                    marker=dict(size=6, symbol='triangle-up')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=prophet_future['ds'], y=prophet_future['yhat_upper'],
                    mode='lines', line=dict(width=0),
                    showlegend=False, hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=prophet_future['ds'], y=prophet_future['yhat_lower'],
                    mode='lines', line=dict(width=0),
                    fillcolor='rgba(155, 89, 182, 0.2)',
                    fill='tonexty', name='Confidence Interval'
                ))
                
                # Add vertical line at forecast start using shape
                forecast_start = df_ts.index[-1]
                fig.add_shape(
                    type="line",
                    x0=forecast_start, x1=forecast_start,
                    y0=0, y1=1, yref="paper",
                    line=dict(color="red", width=2, dash="dash")
                )
                
                fig.update_layout(
                    title=f"Prophet: {forecast_days}-Day Future Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Component analysis
                st.subheader("📊 Component Analysis")
                
                # Create matplotlib figure for components
                from matplotlib.figure import Figure
                fig_components = prophet_model.plot_components(prophet_future_forecast)
                st.pyplot(fig_components)
                
                st.success("✓ Prophet model training complete!")
                
            except Exception as e:
                st.error(f"Error training Prophet model: {e}")
        
        # Tab 5: Comparison
        with tab5:
            st.subheader("🏆 Model Performance Comparison")
            
            if all(key in st.session_state for key in ['arima_metrics', 'lstm_metrics', 'prophet_metrics']):
                # Metrics comparison
                comparison_df = pd.DataFrame({
                    'Model': ['ARIMA', 'LSTM', 'Prophet'],
                    'MAE': [
                        st.session_state['arima_metrics'][0],
                        st.session_state['lstm_metrics'][0],
                        st.session_state['prophet_metrics'][0]
                    ],
                    'RMSE': [
                        st.session_state['arima_metrics'][1],
                        st.session_state['lstm_metrics'][1],
                        st.session_state['prophet_metrics'][1]
                    ],
                    'MAPE (%)': [
                        st.session_state['arima_metrics'][2],
                        st.session_state['lstm_metrics'][2],
                        st.session_state['prophet_metrics'][2]
                    ],
                    'R² Score': [
                        st.session_state['arima_metrics'][3],
                        st.session_state['lstm_metrics'][3],
                        st.session_state['prophet_metrics'][3]
                    ]
                })
                
                # Display metrics table
                st.dataframe(
                    comparison_df.style.highlight_min(subset=['MAE', 'RMSE', 'MAPE (%)'], color='lightgreen')
                                       .highlight_max(subset=['R² Score'], color='lightgreen'),
                    use_container_width=True
                )
                
                # Best model
                best_model_idx = comparison_df['RMSE'].idxmin()
                best_model = comparison_df.loc[best_model_idx, 'Model']
                st.success(f"🏆 **Best Performing Model:** {best_model} (Lowest RMSE)")
                
                # Metrics comparison charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        comparison_df, x='Model', y='RMSE',
                        title="RMSE Comparison",
                        color='Model',
                        template="plotly_dark",
                        color_discrete_sequence=['#3498DB', '#E74C3C', '#2ECC71']
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        comparison_df, x='Model', y='R² Score',
                        title="R² Score Comparison",
                        color='Model',
                        template="plotly_dark",
                        color_discrete_sequence=['#3498DB', '#E74C3C', '#2ECC71']
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Combined forecast visualization
                if all(key in st.session_state for key in ['arima_future', 'lstm_future', 'prophet_future']):
                    st.subheader(f"📈 Combined {forecast_days}-Day Forecast")
                    
                    future_dates = pd.date_range(
                        start=df_ts.index[-1] + timedelta(days=1),
                        periods=forecast_days,
                        freq='D'
                    )
                    
                    fig = create_forecast_comparison_chart(
                        df_ts,
                        st.session_state['arima_future'],
                        st.session_state['lstm_future'],
                        st.session_state['prophet_future'],
                        future_dates
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Ensemble forecast - handle both Series and ndarray
                    arima_values = st.session_state['arima_future'].values if hasattr(st.session_state['arima_future'], 'values') else st.session_state['arima_future'].flatten()
                    lstm_values = st.session_state['lstm_future'].flatten()
                    prophet_values = st.session_state['prophet_future']['yhat'].values
                    
                    ensemble_forecast = (arima_values + lstm_values + prophet_values) / 3
                    
                    st.subheader("🎯 Forecast Summary")
                    
                    # Get values safely
                    arima_day1 = st.session_state['arima_future'].iloc[0] if hasattr(st.session_state['arima_future'], 'iloc') else st.session_state['arima_future'][0]
                    arima_last = st.session_state['arima_future'].iloc[-1] if hasattr(st.session_state['arima_future'], 'iloc') else st.session_state['arima_future'][-1]
                    
                    summary_df = pd.DataFrame({
                        'Model': ['ARIMA', 'LSTM', 'Prophet', 'Ensemble Average'],
                        'Day 1': [
                            f"${arima_day1:,.2f}",
                            f"${st.session_state['lstm_future'][0][0]:,.2f}",
                            f"${st.session_state['prophet_future']['yhat'].iloc[0]:,.2f}",
                            f"${ensemble_forecast[0]:,.2f}"
                        ],
                        f'Day {forecast_days}': [
                            f"${arima_last:,.2f}",
                            f"${st.session_state['lstm_future'][-1][0]:,.2f}",
                            f"${st.session_state['prophet_future']['yhat'].iloc[-1]:,.2f}",
                            f"${ensemble_forecast[-1]:,.2f}"
                        ],
                        'Change (%)': [
                            f"{((arima_last / df_ts['Close'].iloc[-1]) - 1) * 100:+.2f}%",
                            f"{((st.session_state['lstm_future'][-1][0] / df_ts['Close'].iloc[-1]) - 1) * 100:+.2f}%",
                            f"{((st.session_state['prophet_future']['yhat'].iloc[-1] / df_ts['Close'].iloc[-1]) - 1) * 100:+.2f}%",
                            f"{((ensemble_forecast[-1] / df_ts['Close'].iloc[-1]) - 1) * 100:+.2f}%"
                        ]
                    })
                    
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    # 7-Day Summary Box
                    st.subheader("🎯 7-Day Forecast Summary")
                    
                    avg_7day = ensemble_forecast[-1]
                    week_change_ensemble = avg_7day - df_ts['Close'].iloc[-1]
                    week_pct_ensemble = (week_change_ensemble / df_ts['Close'].iloc[-1]) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"""
                        **📅 Current Date:** {df_ts.index[-1].strftime('%Y-%m-%d')}  
                        **💰 Current Price:** ${df_ts['Close'].iloc[-1]:,.2f}
                        """)
                    with col2:
                        if week_change_ensemble > 0:
                            st.success(f"""
                            **🎯 7-Day Target:** ${avg_7day:,.2f}  
                            **📈 Expected Gain:** ${week_change_ensemble:,.2f}  
                            **📊 Percentage:** +{week_pct_ensemble:.2f}%  
                            **🚀 Signal:** BULLISH
                            """)
                        else:
                            st.error(f"""
                            **🎯 7-Day Target:** ${avg_7day:,.2f}  
                            **📉 Expected Loss:** ${week_change_ensemble:,.2f}  
                            **📊 Percentage:** {week_pct_ensemble:.2f}%  
                            **⚠️ Signal:** BEARISH
                            """)
                    with col3:
                        arima_pct = ((arima_last / df_ts['Close'].iloc[-1]) - 1) * 100
                        lstm_pct = ((st.session_state['lstm_future'][-1][0] / df_ts['Close'].iloc[-1]) - 1) * 100
                        prophet_pct = ((st.session_state['prophet_future']['yhat'].iloc[-1] / df_ts['Close'].iloc[-1]) - 1) * 100
                        
                        st.warning(f"""
                        **🤖 Models Agreement:**  
                        - ARIMA: {arima_pct:+.2f}%  
                        - LSTM: {lstm_pct:+.2f}%  
                        - Prophet: {prophet_pct:+.2f}%
                        """)
                    
            else:
                st.info("👆 Please run all models first to see the comparison!")
    
    else:
        st.info("👈 Click 'Run Analysis' in the sidebar to start forecasting!")
    
    # Auto-refresh functionality
    if auto_refresh and data_source == "🔴 Real-Time (Live)":
        time.sleep(60)  # Wait 60 seconds
        st.rerun()

else:
    st.error("❌ Unable to load data. Please check the file path or upload a CSV file.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>🔴 Real-Time Bitcoin Forecasting Dashboard | Auto-Updates Every 60 Seconds</p>
        <p>Models: ARIMA • LSTM • Facebook Prophet | 7-Day Predictions</p>
        <p style='font-size: 12px;'>Data Source: Yahoo Finance (BTC-USD) | Built with Streamlit & yfinance</p>
    </div>
""", unsafe_allow_html=True)
