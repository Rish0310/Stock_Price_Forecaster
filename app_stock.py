import streamlit as st
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(page_title="Stock Price Forecaster", page_icon="📈", layout="wide")

# Title and description
st.title("📈 Stock Price Forecasting with ARIMA")
st.markdown("""
This application uses ARIMA (AutoRegressive Integrated Moving Average) model to forecast stock prices.
Upload your historical stock data and select parameters to generate predictions.
""")

# Sidebar for inputs
st.sidebar.header("📊 Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload Stock Data (CSV)", type=['csv'])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Display data info
    st.sidebar.success(f"✅ Loaded {len(df)} rows of data")
    
    # Ticker selection
    if 'Ticker' in df.columns:
        tickers = df['Ticker'].unique()
        selected_ticker = st.sidebar.selectbox("Select Stock Ticker", tickers)
    else:
        selected_ticker = st.sidebar.text_input("Enter Stock Ticker", "STOCK")
    
    # Forecast parameters
    forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 7)
    train_size = st.sidebar.slider("Training Data (%)", 50, 90, 80)
    
    # Run forecast button
    if st.sidebar.button("🚀 Run Forecast", type="primary"):
        
        with st.spinner("Processing data and training model..."):
            
            # Filter data for selected ticker
            if 'Ticker' in df.columns:
                df_ticker = df[df['Ticker'] == selected_ticker].copy()
            else:
                df_ticker = df.copy()
            
            # Convert date column
            date_col = [col for col in df_ticker.columns if 'date' in col.lower()][0]
            df_ticker[date_col] = pd.to_datetime(df_ticker[date_col])
            df_ticker = df_ticker.sort_values(date_col).reset_index(drop=True)
            
            # Calculate rolling features
            df_ticker['Rolling_Mean_3'] = df_ticker['VWAP'].rolling(window=3).mean()
            df_ticker['Rolling_Mean_7'] = df_ticker['VWAP'].rolling(window=7).mean()
            df_ticker['Rolling_Std_3'] = df_ticker['VWAP'].rolling(window=3).std()
            df_ticker['Rolling_Std_7'] = df_ticker['VWAP'].rolling(window=7).std()
            
            # Drop NaN values
            df_ticker = df_ticker.dropna()
            
            # Split data
            split_idx = int(len(df_ticker) * (train_size / 100))
            df_train = df_ticker[:split_idx]
            df_test = df_ticker[split_idx:]
            
            # Features for modeling
            ind_features = ['Rolling_Mean_3', 'Rolling_Mean_7', 'Rolling_Std_3', 'Rolling_Std_7']
            
            # Train ARIMA model
            st.info("🔄 Training ARIMA model... This may take a moment.")
            model = auto_arima(
                df_train['VWAP'],
                exogenous=df_train[ind_features],
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_order=5,
                trace=False
            )
            
            # Make predictions on test set
            test_predictions = model.predict(
                n_periods=len(df_test),
                exogenous=df_test[ind_features]
            )
            df_test['Forecast_ARIMA'] = test_predictions
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(df_test['VWAP'], df_test['Forecast_ARIMA']))
            mae = mean_absolute_error(df_test['VWAP'], df_test['Forecast_ARIMA'])
            
            # Future forecast
            last_date = df_ticker[date_col].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
            
            # Create future features (using last known values)
            last_values = df_ticker[ind_features].iloc[-1:].values
            future_features = pd.DataFrame(
                np.repeat(last_values, forecast_days, axis=0),
                columns=ind_features
            )
            
            # Future predictions
            future_forecast = model.predict(n_periods=forecast_days, exogenous=future_features)
            
            # Create future dataframe
            df_future = pd.DataFrame({
                date_col: future_dates,
                'Forecast_ARIMA': future_forecast
            })
            
            # Display results
            st.success("✅ Model Training Complete!")
            
            # Metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 RMSE", f"{rmse:.2f}")
            with col2:
                st.metric("📉 MAE", f"{mae:.2f}")
            with col3:
                st.metric("🎯 Train Size", f"{len(df_train)} days")
            with col4:
                st.metric("🔮 Forecast", f"{forecast_days} days")
            
            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["📈 Overview", "📊 Detailed Analysis", "📋 Data"])
            
            with tab1:
                st.subheader(f"Stock Overview: {selected_ticker}")
                
                # Candlestick chart
                fig_candle = go.Figure(data=[go.Candlestick(
                    x=df_ticker[date_col],
                    open=df_ticker['Open'],
                    high=df_ticker['High'],
                    low=df_ticker['Low'],
                    close=df_ticker['Close'],
                    name='Price'
                )])
                
                fig_candle.update_layout(
                    title=f"{selected_ticker} - Candlestick Chart",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    height=500,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig_candle, use_container_width=True)
                
                # Forecast overview
                st.subheader("Forecast Overview")
                fig_overview = go.Figure()
                
                # Historical data
                fig_overview.add_trace(go.Scatter(
                    x=df_ticker[date_col],
                    y=df_ticker['VWAP'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Test predictions
                fig_overview.add_trace(go.Scatter(
                    x=df_test[date_col],
                    y=df_test['Forecast_ARIMA'],
                    mode='lines',
                    name='Test Predictions',
                    line=dict(color='orange', dash='dash')
                ))
                
                # Future forecast
                fig_overview.add_trace(go.Scatter(
                    x=df_future[date_col],
                    y=df_future['Forecast_ARIMA'],
                    mode='lines+markers',
                    name='Future Forecast',
                    line=dict(color='red', width=3),
                    marker=dict(size=8)
                ))
                
                fig_overview.update_layout(
                    title="Price Forecast",
                    xaxis_title="Date",
                    yaxis_title="VWAP Price",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_overview, use_container_width=True)
            
            with tab2:
                st.subheader("Detailed Analysis")
                
                # Training vs Testing
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Training Data Performance**")
                    fig_train = go.Figure()
                    fig_train.add_trace(go.Scatter(
                        x=df_train[date_col],
                        y=df_train['VWAP'],
                        mode='lines',
                        name='Training Data',
                        line=dict(color='green')
                    ))
                    fig_train.update_layout(height=300, showlegend=True)
                    st.plotly_chart(fig_train, use_container_width=True)
                
                with col2:
                    st.markdown("**Testing Data Performance**")
                    fig_test = go.Figure()
                    fig_test.add_trace(go.Scatter(
                        x=df_test[date_col],
                        y=df_test['VWAP'],
                        mode='lines',
                        name='Actual',
                        line=dict(color='blue')
                    ))
                    fig_test.add_trace(go.Scatter(
                        x=df_test[date_col],
                        y=df_test['Forecast_ARIMA'],
                        mode='lines',
                        name='Predicted',
                        line=dict(color='orange', dash='dash')
                    ))
                    fig_test.update_layout(height=300, showlegend=True)
                    st.plotly_chart(fig_test, use_container_width=True)
                
                # Prediction Error Analysis
                st.markdown("**Prediction Error Analysis**")
                df_test['Error'] = df_test['VWAP'] - df_test['Forecast_ARIMA']
                df_test['Error_Percent'] = (df_test['Error'] / df_test['VWAP']) * 100
                
                fig_error = go.Figure()
                fig_error.add_trace(go.Scatter(
                    x=df_test[date_col],
                    y=df_test['Error_Percent'],
                    mode='lines+markers',
                    name='Prediction Error %',
                    line=dict(color='red')
                ))
                fig_error.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_error.update_layout(
                    title="Prediction Error Over Time",
                    xaxis_title="Date",
                    yaxis_title="Error (%)",
                    height=300
                )
                st.plotly_chart(fig_error, use_container_width=True)
                
                # Rolling Statistics
                st.markdown("**Rolling Statistics**")
                fig_rolling = go.Figure()
                fig_rolling.add_trace(go.Scatter(
                    x=df_ticker[date_col],
                    y=df_ticker['VWAP'],
                    mode='lines',
                    name='VWAP',
                    line=dict(color='blue')
                ))
                fig_rolling.add_trace(go.Scatter(
                    x=df_ticker[date_col],
                    y=df_ticker['Rolling_Mean_3'],
                    mode='lines',
                    name='3-Day MA',
                    line=dict(color='orange', dash='dash')
                ))
                fig_rolling.add_trace(go.Scatter(
                    x=df_ticker[date_col],
                    y=df_ticker['Rolling_Mean_7'],
                    mode='lines',
                    name='7-Day MA',
                    line=dict(color='green', dash='dash')
                ))
                fig_rolling.update_layout(
                    title="Price with Moving Averages",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    height=400
                )
                st.plotly_chart(fig_rolling, use_container_width=True)
            
            with tab3:
                st.subheader("📋 Forecast Data")
                
                # Display future predictions
                st.markdown("**Future Predictions**")
                display_future = df_future.copy()
                display_future[date_col] = display_future[date_col].dt.strftime('%Y-%m-%d')
                display_future.columns = ['Date', 'Predicted Price']
                st.dataframe(display_future, use_container_width=True, hide_index=True)
                
                # Display test predictions
                st.markdown("**Test Set Predictions vs Actual**")
                display_test = df_test[[date_col, 'VWAP', 'Forecast_ARIMA', 'Error_Percent']].copy()
                display_test[date_col] = display_test[date_col].dt.strftime('%Y-%m-%d')
                display_test.columns = ['Date', 'Actual Price', 'Predicted Price', 'Error %']
                display_test['Error %'] = display_test['Error %'].round(2)
                st.dataframe(display_test, use_container_width=True, hide_index=True)
                
                # Model Summary
                st.markdown("**Model Summary**")
                st.text(model.summary())

else:
    # Instructions when no file is uploaded
    st.info("👈 Please upload a CSV file from the sidebar to begin")
    
    st.markdown("""
    ### 📝 Required CSV Format:
    
    Your CSV file should contain the following columns:
    - **Date** (any date format)
    - **Open** (opening price)
    - **High** (highest price)
    - **Low** (lowest price)
    - **Close** (closing price)
    - **VWAP** (Volume Weighted Average Price)
    - **Ticker** (optional, for multiple stocks)
    
    ### 🎯 How to Use:
    1. Upload your stock data CSV file
    2. Select the stock ticker (if applicable)
    3. Adjust forecast parameters
    4. Click "Run Forecast" to generate predictions
    
    ### 📊 Features:
    - ✅ Automatic ARIMA model optimization
    - ✅ Candlestick chart visualization
    - ✅ Rolling window features (3-day & 7-day)
    - ✅ Performance metrics (RMSE, MAE)
    - ✅ Interactive charts with Plotly
    - ✅ Future price predictions
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, pmdarima, and Plotly")
