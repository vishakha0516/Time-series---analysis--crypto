# Bitcoin Price Forecasting Dashboard

## 🚀 Quick Start

### Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

### Running the Dashboard

Run the Streamlit application:
```bash
streamlit run streamlit_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 Features

### 1. **Interactive Price Analysis**
   - Real-time price charts with Plotly
   - Candlestick OHLC visualization
   - Trading volume analysis
   - Statistical summaries and distributions

### 2. **ARIMA Model**
   - Auto-Regressive Integrated Moving Average
   - Configurable (p, d, q) parameters
   - Test predictions and future forecasting
   - Performance metrics (MAE, RMSE, MAPE, R²)

### 3. **LSTM Neural Network**
   - Deep learning time series forecasting
   - 3-layer LSTM architecture with Dropout
   - Configurable sequence length and epochs
   - Training history visualization

### 4. **Facebook Prophet**
   - Automated trend detection
   - Seasonality decomposition
   - Confidence intervals
   - Component analysis (trend, weekly, daily)

### 5. **Model Comparison**
   - Side-by-side performance metrics
   - Combined forecast visualization
   - Ensemble predictions
   - Best model identification

## 🎛️ Dashboard Controls

**Sidebar Settings:**
- **File Upload**: Upload your own Bitcoin CSV file
- **Train/Test Split**: Adjust the percentage (60-90%)
- **Forecast Days**: Set prediction horizon (7-60 days)
- **ARIMA Parameters**: Customize p, d, q values
- **LSTM Parameters**: Set sequence length, epochs, batch size

## 📈 Usage Tips

1. **Start with Default Settings**: Click "Run Analysis" to use pre-configured parameters
2. **Experiment with Parameters**: Adjust settings in the sidebar to optimize models
3. **Compare Models**: Check the "Model Comparison" tab to see which performs best
4. **Export Results**: Take screenshots or export forecast data for reporting

## 🔧 Customization

The dashboard uses:
- **Plotly** for interactive charts
- **TensorFlow/Keras** for LSTM modeling
- **Statsmodels** for ARIMA
- **Prophet** for time series forecasting
- **Streamlit** for the web interface

## 📊 Data Format

Expected CSV format (with multi-level headers):
```
Date,Price,Open,High,Low,Close,Volume
```

Or use the default file: `BTC_1Jan2025_to_30Nov2025 (1).csv`

## 💡 Performance

- **ARIMA**: Fast training, good for short-term predictions
- **LSTM**: Slower training, captures complex patterns
- **Prophet**: Balance of speed and accuracy, includes seasonality

## 🐛 Troubleshooting

- **Slow performance?** Reduce LSTM epochs or sequence length
- **Memory errors?** Use smaller batch size or reduce data size
- **Import errors?** Ensure all packages in requirements.txt are installed

## 📝 Notes

- Models are retrained each time you click "Run Analysis"
- Adjust parameters based on your data characteristics
- Use ensemble predictions for more robust forecasting
