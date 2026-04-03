import plotly.graph_objects as go
import pandas as pd
import streamlit as st

def render_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str = "Line Chart"):
    """Renders a simple Plotly line chart."""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        st.warning(f"Insufficient data for {title}.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='lines', name=y_col, line=dict(color='#3b82f6', width=2)))
    fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

def render_candlestick(df: pd.DataFrame, title: str="Price History"):
    """Renders a Plotly candlestick chart."""
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        st.warning("Insufficient specific column data for Candlestick chart.")
        return

    fig = go.Figure(data=[go.Candlestick(
                    x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close']
                )])
    fig.update_layout(title=title, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

def render_advanced_chart(df: pd.DataFrame, title: str="Advanced Analysis"):
    from plotly.subplots import make_subplots
    from utils.indicators import compute_rsi
    
    if df.empty or not all(col in df.columns for col in ['Date', 'Open', 'High', 'Low', 'Close']):
        st.warning("Insufficient specific column data for Advanced chart.")
        return

    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(title, 'RSI (14)'),
                        row_width=[0.3, 0.7])

    # Candlestick
    fig.add_trace(go.Candlestick(x=df['Date'],
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='Price'),
                row=1, col=1)

    # Pandas MAs for visualization
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], line=dict(color='orange', width=1.5), name='SMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA200'], line=dict(color='blue', width=1.5), name='SMA 200'), row=1, col=1)

    # RSI
    df['RSI'] = compute_rsi(df['Close'], 14)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], line=dict(color='purple', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, height=550, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
