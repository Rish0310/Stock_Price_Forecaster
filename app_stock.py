import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

# Title
st.markdown('<h1 style="text-align: center; color: #1f77b4;">📈 Stock Price Forecaster</h1>', unsafe_allow_html=True)
st.markdown("**Machine Learning Stock Price Forecasting using ARIMA Time Series Analysis**")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader("📁 Upload Stock CSV", type=['csv'])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    df = df.set_index('Date')
    
    st.sidebar.success("✅ Data loaded successfully!")
    
    # Prediction settings
    st.sidebar.subheader("🔮 Prediction Settings")
    forecast_days = st.sidebar.slider("Days to Predict into Future", 1, 30, 7)
    
    # Data info
    st.sidebar.metric("📊 Total Days", len(df))
    st.sidebar.metric("📅 Date Range", f"{df.index[0]} to {df.index[-1]}")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🔮 Future Predictions", "📈 Model Performance"])
    
    # ========== TAB 1: DATA OVERVIEW ==========
    with tab1:
        st.header("📊 Historical Stock Data")
        
        st.subheader("📋 Recent Data (Last 10 Days)")
        st.dataframe(df.tail(10), use_container_width=True)
        
        st.divider()
        
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        price_change = current_price - prev_price
        pct_change = (price_change / prev_price) * 100
        
        with col1:
            st.metric("📍 Current Price", f"₹{current_price:.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
        with col2:
            st.metric("📈 All-Time High", f"₹{df['High'].max():.2f}")
        with col3:
            st.metric("📉 All-Time Low", f"₹{df['Low'].min():.2f}")
        with col4:
            st.metric("📊 Avg Volume", f"{df['Volume'].mean():,.0f}")
        
        st.divider()
        
        st.subheader("📈 Price History (Last 180 Days)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-180:], y=df['Close'][-180:], mode='lines', name='Closing Price', line=dict(color='#1f77b4', width=2)))
        fig.update_layout(xaxis_title="Date", yaxis_title="Price (₹)", height=450, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    # ========== TAB 2: FUTURE PREDICTIONS ==========
    with tab2:
        st.header(f"🔮 Next {forecast_days} Days Price Predictions")
        
        with st.spinner("🔄 Preparing data and engineering features..."):
            dataframe = df.drop(columns=['Trades'], errors='ignore')
            dataframe = dataframe.dropna()
            
            lag_features = ['High', 'Low', 'Volume', 'Turnover']
            window1, window2 = 3, 7
            
            for col in lag_features:
                dataframe[f'{col}rolling_mean_3'] = dataframe[col].rolling(window=window1).mean()
                dataframe[f'{col}rolling_mean_7'] = dataframe[col].rolling(window=window2).mean()
                dataframe[f'{col}rolling_std_3'] = dataframe[col].rolling(window=window1).std()
                dataframe[f'{col}rolling_std_7'] = dataframe[col].rolling(window=window2).std()
            
            dataframe = dataframe.dropna()
            ind_features = [col for col in dataframe.columns if 'rolling' in col]
            
            st.success(f"✅ Engineered {len(ind_features)} features from {len(dataframe)} days of data")
        
        with st.spinner("🤖 Training ARIMA model (2-3 mins)..."):
            progress_bar = st.progress(0)
            
            try:
                # Cloud optimization
                if len(dataframe) > 1500:
                    st.info(f"📊 Using last 1500 days for training (Cloud optimization)")
                    df_train = dataframe[-1500:]
                else:
                    df_train = dataframe
                
                model = auto_arima(
                    y=df_train['VWAP'],
                    X=df_train[ind_features],
                    start_p=1, max_p=2,
                    start_q=1, max_q=2,
                    max_d=1,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    trace=False,
                    n_jobs=1,
                    maxiter=50
                )
                
                progress_bar.progress(100)
                st.success(f"✅ Model trained! Best Model: ARIMA{model.order}")
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.info("💡 Try uploading a smaller dataset")
                st.stop()
        
        with st.spinner(f"🔮 Generating {forecast_days}-day forecast..."):
            last_7_features = dataframe[ind_features].iloc[-7:].mean().values
            future_features = np.tile(last_7_features, (forecast_days, 1))
            future_forecast = model.predict(n_periods=forecast_days, X=future_features)
            
            last_date = pd.to_datetime(dataframe.index[-1])
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
            
            future_df = pd.DataFrame({'Date': future_dates, 'Predicted_VWAP': future_forecast})
            future_df['Date'] = future_df['Date'].dt.strftime('%Y-%m-%d')
            future_df.set_index('Date', inplace=True)
        
        st.success(f"✅ Generated {forecast_days}-day forecast!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Historical + Future Price Chart")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(dataframe.index[-90:]), y=dataframe['VWAP'][-90:].values, mode='lines', name='Historical VWAP', line=dict(color='#1f77b4', width=2)))
            fig.add_trace(go.Scatter(x=[dataframe.index[-1]], y=[dataframe['VWAP'].iloc[-1]], mode='markers', name='Today', marker=dict(color='green', size=12, symbol='star')))
            fig.add_trace(go.Scatter(x=list(future_df.index), y=future_df['Predicted_VWAP'].values, mode='lines+markers', name=f'Future Predictions ({forecast_days}d)', line=dict(color='red', width=2, dash='dash'), marker=dict(size=8)))
            
            fig.update_layout(title=f"Stock Price: Last 90 Days + Next {forecast_days} Days", xaxis_title="Date", yaxis_title="VWAP (₹)", height=500, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Prediction Summary")
            
            current_price = dataframe['VWAP'].iloc[-1]
            predicted_price_7d = future_df['Predicted_VWAP'].iloc[min(6, len(future_df)-1)]
            predicted_price_final = future_df['Predicted_VWAP'].iloc[-1]
            
            price_change_7d = predicted_price_7d - current_price
            pct_change_7d = (price_change_7d / current_price) * 100
            
            st.metric("📍 Current Price", f"₹{current_price:.2f}")
            st.metric(f"🔮 Predicted (Day 7)", f"₹{predicted_price_7d:.2f}", f"{price_change_7d:+.2f} ({pct_change_7d:+.2f}%)")
            
            if forecast_days > 7:
                price_change_final = predicted_price_final - current_price
                pct_change_final = (price_change_final / current_price) * 100
                st.metric(f"🔮 Predicted (Day {forecast_days})", f"₹{predicted_price_final:.2f}", f"{price_change_final:+.2f} ({pct_change_final:+.2f}%)")
            
            st.divider()
            
            if pct_change_7d > 2:
                st.success("📈 **Bullish Trend** - Price expected to rise")
            elif pct_change_7d < -2:
                st.error("📉 **Bearish Trend** - Price expected to fall")
            else:
                st.info("➡️ **Neutral Trend** - Price relatively stable")
            
            st.divider()
            st.subheader("📅 Daily Predictions")
            
            display_df = future_df.copy()
            display_df['Day'] = [f"Day {i+1}" for i in range(len(display_df))]
            display_df = display_df[['Day', 'Predicted_VWAP']]
            st.dataframe(display_df, height=300)
            
            csv = future_df.to_csv()
            st.download_button("📥 Download Predictions CSV", csv, f"predictions_{forecast_days}days.csv", "text/csv")
    
    # ========== TAB 3: MODEL PERFORMANCE ==========
    with tab3:
        st.header("📈 Model Validation & Performance")
        st.info("🔍 Testing model accuracy on historical data (80-20 split)")
        
        with st.spinner("📊 Running backtest..."):
            train_size = int(len(dataframe) * 0.8)
            training_data = dataframe[:train_size]
            testing_data = dataframe[train_size:].copy()
            
            val_model = auto_arima(
                y=training_data['VWAP'], X=training_data[ind_features],
                start_p=1, max_p=2, start_q=1, max_q=2, max_d=1,
                seasonal=False, stepwise=True, suppress_warnings=True,
                trace=False, n_jobs=1, maxiter=50
            )
            
            val_forecast = val_model.predict(n_periods=len(testing_data), X=testing_data[ind_features])
            testing_data['Forecast_ARIMA'] = val_forecast.values
        
        mae = mean_absolute_error(testing_data['VWAP'], testing_data['Forecast_ARIMA'])
        rmse = np.sqrt(mean_squared_error(testing_data['VWAP'], testing_data['Forecast_ARIMA']))
        mape = np.mean(np.abs((testing_data['VWAP'] - testing_data['Forecast_ARIMA']) / testing_data['VWAP'])) * 100
        accuracy = 100 - mape
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 MAE", f"₹{mae:.2f}")
        with col2:
            st.metric("📊 RMSE", f"₹{rmse:.2f}")
        with col3:
            st.metric("📊 MAPE", f"{mape:.2f}%")
        with col4:
            st.metric("✅ Accuracy", f"{accuracy:.2f}%")
        
        st.divider()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(testing_data.index), y=testing_data['VWAP'].values, mode='lines', name='Actual', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=list(testing_data.index), y=testing_data['Forecast_ARIMA'].values, mode='lines', name='Predicted', line=dict(color='red', width=2, dash='dash')))
        fig.update_layout(title="Predicted vs Actual Prices", xaxis_title="Date", yaxis_title="VWAP (₹)", height=450, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 **Please upload a stock CSV file to get started**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📋 Expected CSV Format
        
        - **Date** (index)
        - **Open, High, Low, Close** (prices)
        - **Volume, Turnover** (trading metrics)
        - **VWAP** (Volume Weighted Average Price)
        - **Trades** (optional)
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 What This App Does
        
        1. ✅ Loads historical stock data
        2. ✅ Engineers 16 technical indicators
        3. ✅ Trains Fast ARIMA model (2-3 mins)
        4. ✅ Predicts real future prices (7-30 days)
        5. ✅ Shows accuracy metrics
        6. ✅ Downloads predictions as CSV
        """)

st.divider()
st.markdown('<div style="text-align: center; color: gray;"><p><strong>Built with ❤️ using Streamlit & Machine Learning</strong></p></div>', unsafe_allow_html=True)
