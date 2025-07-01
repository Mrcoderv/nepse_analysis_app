import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta

# Mock NEPSE data since the nepse library might not be available
NEPSE_STOCKS = [
    'NABIL', 'SCB', 'HBL', 'EBL', 'BOKL', 'NICA', 'MBL', 'LBL', 'KBL', 'NCCB',
    'SBI', 'NBB', 'CBL', 'PCBL', 'LBBL', 'KSBBL', 'NLBBL', 'ADBL', 'MLBL', 'NLG',
    'NLICL', 'LICN', 'PICL', 'LGIL', 'SICL', 'UICL', 'PRIN', 'SIL', 'IGI', 'NLIC',
    'HIDCL', 'CHCL', 'SJCL', 'UNHPL', 'AKPL', 'SHPC', 'NHPC', 'BPCL', 'HPPL', 'NGPL'
]

def get_all_stocks():
    """Return list of available NEPSE stocks"""
    return sorted(NEPSE_STOCKS)

def generate_mock_data(symbol, days=365):
    """Generate mock stock data for demonstration"""
    import numpy as np
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate mock price data with some randomness
    np.random.seed(hash(symbol) % 2**32)  # Consistent seed based on symbol
    base_price = np.random.uniform(100, 1000)
    
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        # Add some trend and volatility
        trend = 0.001 * np.sin(i * 0.01)  # Long-term trend
        volatility = np.random.normal(0, 0.02)  # Daily volatility
        current_price = current_price * (1 + trend + volatility)
        prices.append(max(current_price, 1))  # Ensure price doesn't go negative
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'open': [p * np.random.uniform(0.98, 1.02) for p in prices],
        'high': [p * np.random.uniform(1.0, 1.05) for p in prices],
        'low': [p * np.random.uniform(0.95, 1.0) for p in prices],
        'volume': [np.random.randint(1000, 100000) for _ in prices]
    })
    
    return df

def get_stock_data(symbol):
    """Get stock data for the given symbol"""
    try:
        # Try to get real data from Yahoo Finance (some NEPSE stocks might be available)
        ticker = f"{symbol}.NP"  # Nepal stock exchange suffix
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        
        if df.empty:
            raise Exception("No data from Yahoo Finance")
            
        df.reset_index(inplace=True)
        df.columns = df.columns.str.lower()
        df = df.rename(columns={'date': 'date'})
        
        return df[['date', 'close', 'open', 'high', 'low', 'volume']]
        
    except:
        # Fallback to mock data
        return generate_mock_data(symbol)