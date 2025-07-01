import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from prophet import Prophet
import streamlit as st

def plot_stock(df, symbol):
    """Create an interactive stock price chart"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{symbol} Stock Price', 'Volume'),
        row_width=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name="Volume",
            marker_color='rgba(158,202,225,0.8)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"{symbol} Stock Analysis",
        xaxis_title="Date",
        yaxis_title="Price (NPR)",
        height=600,
        showlegend=False
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    df = df.copy()
    
    # Moving averages
    df['MA_20'] = df['close'].rolling(window=20).mean()
    df['MA_50'] = df['close'].rolling(window=50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    return df

def plot_technical_indicators(df, symbol):
    """Plot technical indicators"""
    df_tech = calculate_technical_indicators(df)
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'{symbol} Price with Moving Averages',
            'Bollinger Bands',
            'RSI'
        )
    )
    
    # Price with moving averages
    fig.add_trace(
        go.Scatter(x=df_tech['date'], y=df_tech['close'], name='Close Price', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_tech['date'], y=df_tech['MA_20'], name='MA 20', line=dict(color='orange')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_tech['date'], y=df_tech['MA_50'], name='MA 50', line=dict(color='red')),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df_tech['date'], y=df_tech['BB_upper'], name='BB Upper', line=dict(color='gray', dash='dash')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_tech['date'], y=df_tech['BB_middle'], name='BB Middle', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_tech['date'], y=df_tech['BB_lower'], name='BB Lower', line=dict(color='gray', dash='dash')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_tech['date'], y=df_tech['close'], name='Close', line=dict(color='black')),
        row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df_tech['date'], y=df_tech['RSI'], name='RSI', line=dict(color='purple')),
        row=3, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True)
    fig.update_yaxes(title_text="Price (NPR)", row=1, col=1)
    fig.update_yaxes(title_text="Price (NPR)", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    return fig

def forecast_stock(df, symbol, periods=30):
    """Forecast stock prices using Prophet"""
    try:
        # Prepare data for Prophet
        prophet_df = df[['date', 'close']].copy()
        prophet_df = prophet_df.rename(columns={'date': 'ds', 'close': 'y'})
        
        # Remove any NaN values
        prophet_df = prophet_df.dropna()
        
        # Initialize and fit Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        model.fit(prophet_df)
        
        # Make future predictions
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        # Create forecast plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=prophet_df['ds'],
            y=prophet_df['y'],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast
        forecast_data = forecast.tail(periods)
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Confidence Interval',
            fillcolor='rgba(255,0,0,0.2)'
        ))
        
        fig.update_layout(
            title=f"{symbol} Stock Price Forecast ({periods} days)",
            xaxis_title="Date",
            yaxis_title="Price (NPR)",
            height=500
        )
        
        return fig, forecast_data
        
    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")
        return None, None

def calculate_stock_metrics(df):
    """Calculate key stock metrics"""
    if df.empty:
        return {}
    
    current_price = df['close'].iloc[-1]
    previous_price = df['close'].iloc[-2] if len(df) > 1 else current_price
    
    # Calculate metrics
    metrics = {
        'Current Price': f"NPR {current_price:.2f}",
        'Previous Close': f"NPR {previous_price:.2f}",
        'Change': f"NPR {current_price - previous_price:.2f}",
        'Change %': f"{((current_price - previous_price) / previous_price * 100):.2f}%",
        '52W High': f"NPR {df['high'].max():.2f}",
        '52W Low': f"NPR {df['low'].min():.2f}",
        'Average Volume': f"{df['volume'].mean():.0f}",
        'Market Cap': "N/A"  # Would need shares outstanding
    }
    
    return metrics