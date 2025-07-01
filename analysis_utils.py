import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from prophet import Prophet
import streamlit as st

def plot_stock(df, symbol):
    """Create an interactive stock price chart with enhanced styling"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(f'{symbol} Stock Price & Technical Analysis', 'Trading Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick chart with better colors
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="OHLC",
            increasing_line_color='#00C851',
            decreasing_line_color='#FF4444',
            increasing_fillcolor='#00C851',
            decreasing_fillcolor='#FF4444'
        ),
        row=1, col=1
    )
    
    # Add moving averages if enough data
    if len(df) >= 20:
        ma_20 = df['close'].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=ma_20,
                mode='lines',
                name='MA 20',
                line=dict(color='orange', width=2),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    if len(df) >= 50:
        ma_50 = df['close'].rolling(window=50).mean()
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=ma_50,
                mode='lines',
                name='MA 50',
                line=dict(color='purple', width=2),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    # Volume chart with color coding
    colors = ['red' if close < open else 'green' for close, open in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Update layout with better styling
    fig.update_layout(
        title={
            'text': f"{symbol} - Comprehensive Stock Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1f77b4'}
        },
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update axes
    fig.update_xaxes(
        title_text="Date",
        rangeslider_visible=False,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    fig.update_yaxes(
        title_text="Price (NPR)",
        row=1, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    fig.update_yaxes(
        title_text="Volume",
        row=2, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    return fig

def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators"""
    df = df.copy()
    
    # Moving averages
    df['MA_5'] = df['close'].rolling(window=5).mean()
    df['MA_10'] = df['close'].rolling(window=10).mean()
    df['MA_20'] = df['close'].rolling(window=20).mean()
    df['MA_50'] = df['close'].rolling(window=50).mean()
    df['MA_200'] = df['close'].rolling(window=200).mean()
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['%K'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    return df

def plot_technical_indicators(df, symbol):
    """Plot comprehensive technical indicators"""
    df_tech = calculate_technical_indicators(df)
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'{symbol} - Price with Moving Averages',
            'Bollinger Bands',
            'RSI & Stochastic',
            'MACD'
        ),
        row_heights=[0.4, 0.25, 0.2, 0.15]
    )
    
    # Price with moving averages
    fig.add_trace(
        go.Scatter(x=df_tech['date'], y=df_tech['close'], name='Close Price', 
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    if len(df_tech) >= 20:
        fig.add_trace(
            go.Scatter(x=df_tech['date'], y=df_tech['MA_20'], name='MA 20', 
                      line=dict(color='orange', width=1.5)),
            row=1, col=1
        )
    
    if len(df_tech) >= 50:
        fig.add_trace(
            go.Scatter(x=df_tech['date'], y=df_tech['MA_50'], name='MA 50', 
                      line=dict(color='red', width=1.5)),
            row=1, col=1
        )
    
    # Bollinger Bands
    if len(df_tech) >= 20:
        fig.add_trace(
            go.Scatter(x=df_tech['date'], y=df_tech['BB_upper'], name='BB Upper', 
                      line=dict(color='gray', dash='dash'), opacity=0.7),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_tech['date'], y=df_tech['BB_middle'], name='BB Middle', 
                      line=dict(color='blue', width=1)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_tech['date'], y=df_tech['BB_lower'], name='BB Lower', 
                      line=dict(color='gray', dash='dash'), opacity=0.7),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_tech['date'], y=df_tech['close'], name='Close', 
                      line=dict(color='black', width=1.5)),
            row=2, col=1
        )
    
    # RSI and Stochastic
    if len(df_tech) >= 14:
        fig.add_trace(
            go.Scatter(x=df_tech['date'], y=df_tech['RSI'], name='RSI', 
                      line=dict(color='purple', width=2)),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, opacity=0.7)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, opacity=0.7)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1, opacity=0.5)
        
        # Add Stochastic if available
        if '%K' in df_tech.columns:
            fig.add_trace(
                go.Scatter(x=df_tech['date'], y=df_tech['%K'], name='%K', 
                          line=dict(color='orange', width=1.5)),
                row=3, col=1
            )
    
    # MACD
    if 'MACD' in df_tech.columns:
        fig.add_trace(
            go.Scatter(x=df_tech['date'], y=df_tech['MACD'], name='MACD', 
                      line=dict(color='blue', width=2)),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_tech['date'], y=df_tech['MACD_signal'], name='Signal', 
                      line=dict(color='red', width=1.5)),
            row=4, col=1
        )
        fig.add_trace(
            go.Bar(x=df_tech['date'], y=df_tech['MACD_histogram'], name='Histogram', 
                  marker_color='green', opacity=0.6),
            row=4, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=1000, 
        showlegend=True,
        title={
            'text': f"{symbol} - Technical Analysis Dashboard",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1f77b4'}
        },
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update y-axes titles
    fig.update_yaxes(title_text="Price (NPR)", row=1, col=1)
    fig.update_yaxes(title_text="Price (NPR)", row=2, col=1)
    fig.update_yaxes(title_text="RSI / %K", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    
    return fig

def forecast_stock(df, symbol, periods=30):
    """Enhanced stock price forecasting using Prophet with better error handling"""
    try:
        # Prepare data for Prophet
        prophet_df = df[['date', 'close']].copy()
        prophet_df = prophet_df.rename(columns={'date': 'ds', 'close': 'y'})
        
        # Remove any NaN values and ensure we have enough data
        prophet_df = prophet_df.dropna()
        
        if len(prophet_df) < 30:
            st.warning("⚠️ Insufficient data for reliable forecasting (need at least 30 days)")
            return None, None
        
        # Initialize Prophet model with optimized parameters
        model = Prophet(
            daily_seasonality=False,  # Disable for stock data
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            seasonality_mode='multiplicative',
            interval_width=0.8
        )
        
        # Add custom seasonalities for stock market
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
        
        # Fit the model
        with st.spinner("Training AI model..."):
            model.fit(prophet_df)
        
        # Make future predictions
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        # Create enhanced forecast plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=prophet_df['ds'],
            y=prophet_df['y'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='Date: %{x}<br>Price: NPR %{y:.2f}<extra></extra>'
        ))
        
        # Forecast line
        forecast_data = forecast.tail(periods)
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat'],
            mode='lines',
            name='AI Forecast',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            hovertemplate='Date: %{x}<br>Forecast: NPR %{y:.2f}<extra></extra>'
        ))
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Confidence Interval (80%)',
            fillcolor='rgba(255,127,14,0.2)',
            hovertemplate='Date: %{x}<br>Lower: NPR %{y:.2f}<extra></extra>'
        ))
        
        # Add vertical line to separate historical and forecast
        last_date = prophet_df['ds'].iloc[-1]
        fig.add_vline(
            x=last_date,
            line_dash="dot",
            line_color="gray",
            annotation_text="Forecast Start",
            annotation_position="top"
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"{symbol} - AI-Powered Price Forecast ({periods} days)",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#1f77b4'}
            },
            xaxis_title="Date",
            yaxis_title="Price (NPR)",
            height=600,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig, forecast_data
        
    except Exception as e:
        st.error(f"❌ Error in forecasting: {str(e)}")
        st.info("💡 This might be due to insufficient data or data quality issues. Try selecting a different stock or time period.")
        return None, None

def calculate_stock_metrics(df):
    """Calculate comprehensive stock metrics"""
    if df.empty:
        return {}
    
    current_price = df['close'].iloc[-1]
    previous_price = df['close'].iloc[-2] if len(df) > 1 else current_price
    
    # Price change calculations
    price_change = current_price - previous_price
    price_change_pct = (price_change / previous_price * 100) if previous_price != 0 else 0
    
    # 52-week high/low
    high_52w = df['high'].max()
    low_52w = df['low'].min()
    
    # Volume metrics
    avg_volume = df['volume'].mean()
    latest_volume = df['volume'].iloc[-1]
    
    # Volatility (standard deviation of returns)
    returns = df['close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
    
    # Price performance
    if len(df) >= 30:
        price_30d_ago = df['close'].iloc[-30]
        change_30d = ((current_price - price_30d_ago) / price_30d_ago * 100)
    else:
        change_30d = 0
    
    # Format metrics
    metrics = {
        'Current Price': f"NPR {current_price:.2f}",
        'Previous Close': f"NPR {previous_price:.2f}",
        'Change': f"NPR {price_change:.2f}",
        'Change %': f"{price_change_pct:.2f}%",
        '52W High': f"NPR {high_52w:.2f}",
        '52W Low': f"NPR {low_52w:.2f}",
        'Average Volume': f"{avg_volume:,.0f}",
        'Latest Volume': f"{latest_volume:,.0f}",
        'Volatility': f"{volatility:.2f}%",
        '30D Change': f"{change_30d:.2f}%",
        'Price Range': f"NPR {high_52w - low_52w:.2f}",
        'Market Cap': "N/A"  # Would need shares outstanding data
    }
    
    return metrics

def generate_trading_signals(df):
    """Generate basic trading signals based on technical indicators"""
    if len(df) < 50:
        return "Insufficient data for signal generation"
    
    df_tech = calculate_technical_indicators(df)
    latest = df_tech.iloc[-1]
    
    signals = []
    
    # RSI signals
    if 'RSI' in df_tech.columns and not pd.isna(latest['RSI']):
        if latest['RSI'] > 70:
            signals.append("RSI: Overbought (Consider Sell)")
        elif latest['RSI'] < 30:
            signals.append("RSI: Oversold (Consider Buy)")
        else:
            signals.append("RSI: Neutral")
    
    # Moving average signals
    if 'MA_20' in df_tech.columns and 'MA_50' in df_tech.columns:
        if not pd.isna(latest['MA_20']) and not pd.isna(latest['MA_50']):
            if latest['close'] > latest['MA_20'] > latest['MA_50']:
                signals.append("MA: Strong Bullish Trend")
            elif latest['close'] < latest['MA_20'] < latest['MA_50']:
                signals.append("MA: Strong Bearish Trend")
            else:
                signals.append("MA: Mixed Signals")
    
    # MACD signals
    if 'MACD' in df_tech.columns and 'MACD_signal' in df_tech.columns:
        if not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_signal']):
            if latest['MACD'] > latest['MACD_signal']:
                signals.append("MACD: Bullish Momentum")
            else:
                signals.append("MACD: Bearish Momentum")
    
    return signals if signals else ["No clear signals available"]