# utils/indicators.py
import pandas as pd
import ta

def compute_rsi(close_prices: pd.Series, window: int = 14) -> pd.Series:
    """Computes the Relative Strength Index (RSI)."""
    return ta.momentum.rsi(close_prices, window=window)

def compute_macd(close_prices: pd.Series) -> tuple:
    """Computes MACD and its signal line."""
    macd = ta.trend.macd(close_prices)
    signal = ta.trend.macd_signal(close_prices)
    return macd, signal

def compute_sma(close_prices: pd.Series, window: int = 20) -> pd.Series:
    """Computes Simple Moving Average (SMA)."""
    return ta.trend.sma_indicator(close_prices, window=window)

def generate_signal(rsi_value: float) -> str:
    """
    Smart Signal generation based on current RSI value.
    BUY if RSI < 30
    SELL if RSI > 70
    HOLD otherwise
    """
    if pd.isna(rsi_value):
        return "HOLD"
    
    if rsi_value < 30:
        return "BUY"
    elif rsi_value > 70:
        return "SELL"
    else:
        return "HOLD"
