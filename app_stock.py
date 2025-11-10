import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📈 Stock Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("**AI-Powered Future Stock Price Forecasting using ARIMA**")

st.sidebar.header("⚙️ Configuration")

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
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview", 
        "🔮 Future Predictions", 
        "📈 Model Performance", 
        "💡 Insights & Analysis"
    ])
    
    # ========== TAB 1: DATA OVERVIEW ==========
    with tab1:
        st.header("📊 Historical Stock Data")
        
        # Recent Data - Full Width
        st.subheader("📋 Recent Data (Last 10 Days)")
        st.dataframe(df.tail(10).style.format({
            'Open': '₹{:.2f}',
            'High': '₹{:.2f}',
            'Low': '₹{:.2f}',
            'Close': '₹{:.2f}',
            'VWAP': '₹{:.2f}'
        }), use_container_width=True)
        
        st.divider()
       
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        price_change = current_price - prev_price
        pct_change = (price_change / prev_price) * 100
        
        with col1:
            st.metric(
                "📍 Current Price", 
                f"₹{current_price:.2f}", 
                f"{price_change:+.2f} ({pct_change:+.2f}%)"
            )
        
        with col2:
            st.metric("📈 All-Time High", f"₹{df['High'].max():.2f}")
        
        with col3:
            st.metric("📉 All-Time Low", f"₹{df['Low'].min():.2f}")
        
        with col4:
            avg_volume = df['Volume'].mean()
            st.metric("📊 Avg Volume", f"{avg_volume:,.0f}")
        
        st.divider()
       
        st.subheader("📈 Price History (Last 180 Days)")
        
        fig = go.Figure()
       
        fig.add_trace(go.Scatter(
            x=df.index[-180:], 
            y=df['Close'][-180:],
            mode='lines',
            name='Closing Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            height=450,
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
       
        st.subheader("📊 Candlestick Chart (Last 50 Days)")
        
        fig2 = go.Figure(data=[go.Candlestick(
            x=df.index[-50:],
            open=df['Open'][-50:],
            high=df['High'][-50:],
            low=df['Low'][-50:],
            close=df['Close'][-50:]
        )])
        
        fig2.update_layout(
            xaxis_rangeslider_visible=False,
            height=450
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # ========== TAB 2: FUTURE PREDICTIONS ==========
    with tab2:
        st.header(f"🔮 Next {forecast_days} Days Price Predictions")
       
        with st.spinner("🔄 Preparing data and engineering features..."):
            # Drop Trades column and NaN
            dataframe = df.drop(columns=['Trades'], errors='ignore')
            dataframe = dataframe.dropna()
           
            lag_features = ['High', 'Low', 'Volume', 'Turnover']
            window1 = 3
            window2 = 7
            
            for col in lag_features:
                dataframe[f'{col}rolling_mean_3'] = dataframe[col].rolling(window=window1).mean()
                dataframe[f'{col}rolling_mean_7'] = dataframe[col].rolling(window=window2).mean()
                dataframe[f'{col}rolling_std_3'] = dataframe[col].rolling(window=window1).std()
                dataframe[f'{col}rolling_std_7'] = dataframe[col].rolling(window=window2).std()
           
            dataframe = dataframe.dropna()
         
            ind_features = [col for col in dataframe.columns if 'rolling' in col]
            
            st.success(f"✅ Engineered {len(ind_features)} features from {len(dataframe)} days of data")
      
        with st.spinner("🤖 Training optimized ARIMA model..."):
            progress_bar = st.progress(0)
          
            model = auto_arima(
                y=dataframe['VWAP'],
                X=dataframe[ind_features],
               
                start_p=1, max_p=3,
                start_q=1, max_q=3,
                max_d=2,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=False,
                n_jobs=-1
            )
            
            progress_bar.progress(100)
        
        st.success(f"✅ Model trained successfully! Best Model: ARIMA{model.order}")
      
        with st.spinner(f"🔮 Generating {forecast_days}-day forecast..."):
            # Prepare future features (use average of last 7 days)
            last_7_features = dataframe[ind_features].iloc[-7:].mean().values
            future_features = np.tile(last_7_features, (forecast_days, 1))
         
            future_forecast = model.predict(n_periods=forecast_days, X=future_features)
         
            last_date = pd.to_datetime(dataframe.index[-1])
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
         
            future_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_VWAP': future_forecast
            })
            future_df['Date'] = future_df['Date'].dt.strftime('%Y-%m-%d')
            future_df.set_index('Date', inplace=True)
        
        st.success(f"✅ Generated {forecast_days}-day forecast!")
     
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Historical + Future Price Chart")
          
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=list(dataframe.index[-90:]),
                y=dataframe['VWAP'][-90:].values,
                mode='lines',
                name='Historical VWAP',
                line=dict(color='#1f77b4', width=2)
            ))
        
            fig.add_trace(go.Scatter(
                x=[dataframe.index[-1]],
                y=[dataframe['VWAP'].iloc[-1]],
                mode='markers',
                name='Today',
                marker=dict(color='green', size=12, symbol='star')
            ))
         
            fig.add_trace(go.Scatter(
                x=list(future_df.index),
                y=future_df['Predicted_VWAP'].values,
                mode='lines+markers',
                name=f'Future Predictions ({forecast_days}d)',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=8, symbol='circle')
            ))
           
            fig.add_annotation(
                x=dataframe.index[-1],
                y=dataframe['VWAP'].iloc[-1],
                text="← Today | Future →",
                showarrow=False,
                yshift=30,
                font=dict(size=12, color="gray")
            )
            
            fig.update_layout(
                title=f"Stock Price: Last 90 Days + Next {forecast_days} Days",
                xaxis_title="Date",
                yaxis_title="VWAP Price (₹)",
                height=500,
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Prediction Summary")
           
            current_price = dataframe['VWAP'].iloc[-1]
            predicted_price_7d = future_df['Predicted_VWAP'].iloc[min(6, len(future_df)-1)]
            predicted_price_final = future_df['Predicted_VWAP'].iloc[-1]
            
            price_change_7d = predicted_price_7d - current_price
            pct_change_7d = (price_change_7d / current_price) * 100
            
            price_change_final = predicted_price_final - current_price
            pct_change_final = (price_change_final / current_price) * 100
            
            st.metric("📍 Current Price (Today)", f"₹{current_price:.2f}")
            
            st.metric(
                f"🔮 Predicted Price (Day 7)", 
                f"₹{predicted_price_7d:.2f}",
                f"{price_change_7d:+.2f} ({pct_change_7d:+.2f}%)"
            )
            
            if forecast_days > 7:
                st.metric(
                    f"🔮 Predicted Price (Day {forecast_days})", 
                    f"₹{predicted_price_final:.2f}",
                    f"{price_change_final:+.2f} ({pct_change_final:+.2f}%)"
                )
            
            st.divider()
          
            if pct_change_7d > 2:
                st.success("📈 **Bullish Trend**")
                st.write("Price expected to rise significantly")
            elif pct_change_7d < -2:
                st.error("📉 **Bearish Trend**")
                st.write("Price expected to fall significantly")
            else:
                st.info("➡️ **Neutral Trend**")
                st.write("Price relatively stable")
            
            st.divider()
      
            st.subheader("📅 Daily Predictions")
         
            display_df = future_df.copy()
            display_df['Day'] = [f"Day {i+1}" for i in range(len(display_df))]
            display_df = display_df[['Day', 'Predicted_VWAP']]
            display_df['Change from Today'] = display_df['Predicted_VWAP'] - current_price
            display_df['% Change'] = ((display_df['Predicted_VWAP'] - current_price) / current_price * 100)
            
            st.dataframe(
                display_df.style.format({
                    'Predicted_VWAP': '₹{:.2f}',
                    'Change from Today': '{:+.2f}',
                    '% Change': '{:+.2f}%'
                }).background_gradient(subset=['% Change'], cmap='RdYlGn', vmin=-5, vmax=5),
                height=300
            )
       
            csv = future_df.to_csv()
            st.download_button(
                "📥 Download Predictions CSV",
                csv,
                f"stock_predictions_{forecast_days}days.csv",
                "text/csv",
                key='download-csv'
            )
    
    # ========== TAB 3: MODEL PERFORMANCE ==========
    with tab3:
        st.header("📈 Model Validation & Performance")
        
        st.info("🔍 Testing model accuracy on historical data (80-20 split)")
        
        with st.spinner("📊 Running backtest validation..."):
            # Split for validation (80-20)
            train_size = int(len(dataframe) * 0.8)
            training_data = dataframe[:train_size]
            testing_data = dataframe[train_size:].copy()
  
            val_model = auto_arima(
                y=training_data['VWAP'],
                X=training_data[ind_features],
                start_p=1, max_p=3,
                start_q=1, max_q=3,
                max_d=2,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                trace=False,
                n_jobs=-1
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
            st.caption("Mean Absolute Error")
        
        with col2:
            st.metric("📊 RMSE", f"₹{rmse:.2f}")
            st.caption("Root Mean Squared Error")
        
        with col3:
            st.metric("📊 MAPE", f"{mape:.2f}%")
            st.caption("Mean Absolute % Error")
        
        with col4:
            st.metric("✅ Accuracy", f"{accuracy:.2f}%")
            st.caption("Model Accuracy")
        
        st.divider()
   
        st.subheader("📈 Predicted vs Actual Prices (Test Set)")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(testing_data.index),
            y=testing_data['VWAP'].values,
            mode='lines',
            name='Actual Price',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=list(testing_data.index),
            y=testing_data['Forecast_ARIMA'].values,
            mode='lines',
            name='Predicted Price',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="VWAP (₹)",
            height=450,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
     
        st.subheader("📊 Error Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            errors = testing_data['VWAP'] - testing_data['Forecast_ARIMA']
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=errors,
                nbinsx=30,
                name='Prediction Errors',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                xaxis_title="Error (₹)",
                yaxis_title="Frequency",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Error Statistics")
            st.metric("Mean Error", f"₹{errors.mean():.2f}")
            st.metric("Std Error", f"₹{errors.std():.2f}")
            st.metric("Min Error", f"₹{errors.min():.2f}")
            st.metric("Max Error", f"₹{errors.max():.2f}")
    
    # ========== TAB 4: INSIGHTS ==========
    with tab4:
        st.header("💡 Model Insights & Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Information")
            
            st.markdown(f"""
            **Model Type:** ARIMA (AutoRegressive Integrated Moving Average)
            
            **Best Parameters:** ARIMA{model.order}
            - **p (AR order):** {model.order[0]} - Uses last {model.order[0]} price points
            - **d (Differencing):** {model.order[1]} - Removes trend {model.order[1]} time(s)
            - **q (MA order):** {model.order[2]} - Uses last {model.order[2]} forecast errors
            
            **Training Details:**
            - Total data points: {len(dataframe)}
            - Training set: {train_size} days (80%)
            - Test set: {len(testing_data)} days (20%)
            - Features used: {len(ind_features)} rolling indicators
            
            **Model Quality:**
            - AIC Score: {model.aic():.2f} (lower is better)
            - Training time: ~2-3 minutes (Fast ARIMA)
            - Prediction accuracy: {accuracy:.2f}%
            """)
        
        with col2:
            st.subheader("🎯 Key Features Used")
            
            st.markdown("""
            The model uses **16 technical indicators**:
            
            **📈 Trend Indicators (Moving Averages):**
            - 3-day rolling mean (High, Low, Volume, Turnover)
            - 7-day rolling mean (High, Low, Volume, Turnover)
            
            **📊 Volatility Indicators (Standard Deviation):**
            - 3-day rolling std (High, Low, Volume, Turnover)
            - 7-day rolling std (High, Low, Volume, Turnover)
            
            **Why these features?**
            - **Short-term trends (3-day):** Capture recent momentum
            - **Medium-term trends (7-day):** Capture weekly patterns
            - **Volatility measures:** Identify price stability/risk
            - **Volume/Turnover:** Market activity indicators
            """)
        
        st.divider()
        
        st.subheader("⚠️ Important Disclaimers")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.warning("""
            **Not Financial Advice**
            
            These predictions are based on historical patterns and ML algorithms. 
            They should NOT be used as sole basis for investment decisions.
            """)
        
        with col2:
            st.info("""
            **Model Limitations**
            
            - Cannot predict unexpected events (news, crashes)
            - Assumes past patterns continue
            - Most accurate for short-term (7-14 days)
            - Accuracy decreases for longer horizons
            """)
        
        with col3:
            st.success("""
            **Best Practices**
            
            - Use as one of many analysis tools
            - Combine with fundamental analysis
            - Consider market sentiment
            - Diversify your investments
            - Consult financial advisors
            """)
        
        st.divider()
        
        st.subheader("🚀 Future Improvements")
        
        st.markdown("""
        **Potential enhancements for this model:**
        
        1. **📊 Confidence Intervals** - Show prediction uncertainty ranges
        2. **🤖 Ensemble Models** - Combine ARIMA with LSTM, Prophet
        3. **📰 Sentiment Analysis** - Integrate news sentiment scores
        4. **📈 More Features** - Add RSI, MACD, Bollinger Bands
        5. **🔄 Auto-Retraining** - Update model with new data daily
        6. **📱 Real-time Data** - Connect to live stock APIs
        7. **🎯 Multi-Stock** - Compare predictions across stocks
        8. **⚡ Alert System** - Notify on predicted price movements
        """)

else:
    # Landing page
    st.info("👈 **Please upload a stock CSV file to get started**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📋 Expected CSV Format
        
        Your CSV should contain these columns:
        - **Date** (will be set as index)
        - **Open** - Opening price
        - **High** - Highest price of the day
        - **Low** - Lowest price of the day
        - **Close** - Closing price
        - **Volume** - Number of shares traded
        - **Turnover** - Total value traded
        - **VWAP** - Volume Weighted Average Price
        - **Trades** (optional - will be dropped)
        
        **Example:** BAJAJFINSV.csv format
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 What This App Does
        
        1. ✅ **Loads** your historical stock data
        2. ✅ **Engineers** 16 technical indicators
        3. ✅ **Trains** Fast ARIMA model (2-3 mins)
        4. ✅ **Predicts** real future prices (1-30 days)
        5. ✅ **Validates** accuracy on historical data
        6. ✅ **Visualizes** trends and predictions
        7. ✅ **Downloads** predictions as CSV
        
        ### 🚀 Key Features
        
        - 🔮 **True future forecasting** (not just backtesting!)
        - ⚡ **Fast training** using optimized ARIMA
        - 📊 **Interactive charts** with Plotly
        - 📈 **Accuracy metrics** (MAE, RMSE, MAPE)
        - 📥 **Export predictions** to CSV
        """)
    
    st.divider()
    
    st.markdown("""
    ### 🎓 How It Works
    
    **Step 1: Feature Engineering**
    - Calculates 3-day and 7-day rolling averages
    - Computes volatility (standard deviation)
    - Creates 16 predictive features
    
    **Step 2: Model Training**
    - Uses Fast ARIMA (optimized for speed)
    - Tests ~30-40 parameter combinations
    - Finds best model automatically
    
    **Step 3: Future Prediction**
    - Trains on ALL available data
    - Projects features into the future
    - Generates day-by-day predictions
    
    **Step 4: Validation**
    - Tests accuracy on 20% holdout data
    - Calculates error metrics
    - Shows prediction vs actual charts
    """)

st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Built with ❤️ using Streamlit, pmdarima & Plotly</strong></p>
    <p>🔮 AI-Powered Stock Price Forecasting | ⚡ Fast ARIMA Model | 📊 Real Future Predictions</p>
</div>

""", unsafe_allow_html=True)
