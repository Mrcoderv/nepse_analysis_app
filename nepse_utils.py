import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
import numpy as np

# Comprehensive NEPSE stocks organized by sectors
NEPSE_SECTORS = {
    'Banking': [
        'NABIL', 'SCB', 'HBL', 'EBL', 'BOKL', 'NICA', 'MBL', 'LBL', 'KBL', 'NCCB',
        'SBI', 'NBB', 'CBL', 'PCBL', 'LBBL', 'KSBBL', 'NLBBL', 'ADBL', 'MLBL', 'SANIMA',
        'MEGA', 'CIVIL', 'PRVU', 'GBIME', 'CZBIL', 'SHINE', 'MNBBL', 'KAMANA', 'JBNL', 'KRBL'
    ],
    'Insurance': [
        'NLG', 'NLICL', 'LICN', 'PICL', 'LGIL', 'SICL', 'UICL', 'PRIN', 'SIL', 'IGI', 
        'NLIC', 'AIL', 'GLICL', 'RILG', 'HGI', 'UNL', 'PFL', 'JLI', 'SLICL', 'GILB'
    ],
    'Hydropower': [
        'HIDCL', 'CHCL', 'SJCL', 'UNHPL', 'AKPL', 'SHPC', 'NHPC', 'BPCL', 'HPPL', 'NGPL',
        'RURU', 'UMHL', 'PMHPL', 'LEMF', 'UMRH', 'HURJA', 'RADHI', 'MHNL', 'AKJCL', 'RHPL',
        'SSHL', 'JOSHI', 'NYADI', 'BARUN', 'MKJC', 'RRHP', 'DHPL', 'AHPC', 'KPCL', 'UPPER'
    ],
    'Manufacturing': [
        'UNL', 'JYOTI', 'SHIVM', 'BNT', 'FLBSL', 'GDBL', 'HDHPC', 'ICFC', 'JALPA', 'JBBL',
        'KKHC', 'LLBS', 'MBJC', 'NHDL', 'NIBL', 'NLBBL', 'RBCL', 'RHPL', 'RSDC', 'SABSL',
        'SAHAS', 'SBCF', 'SFCL', 'SHEL', 'SIFC', 'SLBS', 'SMATA', 'SMFBS', 'SMFDB', 'SNLB'
    ],
    'Hotels & Tourism': [
        'OHL', 'TRHPR', 'SHL', 'CGH', 'KHL', 'PPCL', 'SHEL', 'TPC', 'NHDL', 'CORBL',
        'DDBL', 'FOWAD', 'GRDBL', 'HDHPC', 'HURJA', 'ICFC', 'JALPA', 'JBBL', 'KKHC', 'LLBS'
    ],
    'Finance': [
        'CFCL', 'GFCL', 'MFIL', 'NIFRA', 'SIFC', 'GUFL', 'PROFL', 'ICFC', 'JSLBB', 'MPFL',
        'SWFL', 'GMFIL', 'RSDC', 'NIDC', 'SAPDBL', 'SMFDB', 'SMFBS', 'SABSL', 'SFCL', 'SBCF'
    ],
    'Development Banks': [
        'KSBBL', 'SHINE', 'MLBBL', 'CCBL', 'DDBL', 'FOWAD', 'GRDBL', 'JSLBB', 'LLBS', 'MBJC',
        'NLBBL', 'RBCL', 'SABSL', 'SAPDBL', 'SBCF', 'SFCL', 'SLBS', 'SMFBS', 'SMFDB', 'SNLB'
    ],
    'Microfinance': [
        'CBBL', 'DDBL', 'FOWAD', 'GRDBL', 'JSLBB', 'LLBS', 'MBJC', 'NLBBL', 'RBCL', 'SABSL',
        'SAPDBL', 'SBCF', 'SFCL', 'SLBS', 'SMFBS', 'SMFDB', 'SNLB', 'SWFL', 'GMFIL', 'MPFL'
    ],
    'Trading': [
        'BBC', 'BNHC', 'CEDB', 'CHDC', 'CIT', 'CORBL', 'DDBL', 'FOWAD', 'GRDBL', 'HDHPC',
        'HURJA', 'ICFC', 'JALPA', 'JBBL', 'KKHC', 'LLBS', 'MBJC', 'NHDL', 'NIBL', 'NLBBL'
    ],
    'Others': [
        'NESDO', 'NYADI', 'BARUN', 'MKJC', 'RRHP', 'DHPL', 'AHPC', 'KPCL', 'UPPER', 'LEMF',
        'UMRH', 'HURJA', 'RADHI', 'MHNL', 'AKJCL', 'RHPL', 'SSHL', 'JOSHI', 'NYADI', 'BARUN'
    ]
}

# Base prices for different sectors (in NPR)
SECTOR_BASE_PRICES = {
    'Banking': (200, 800),
    'Insurance': (300, 1200),
    'Hydropower': (150, 600),
    'Manufacturing': (100, 500),
    'Hotels & Tourism': (80, 400),
    'Finance': (120, 480),
    'Development Banks': (150, 600),
    'Microfinance': (800, 2000),
    'Trading': (50, 300),
    'Others': (100, 500)
}

def get_all_stocks():
    """Return list of all available NEPSE stocks"""
    all_stocks = []
    for sector_stocks in NEPSE_SECTORS.values():
        all_stocks.extend(sector_stocks)
    return sorted(list(set(all_stocks)))

def get_stocks_by_sector():
    """Return stocks organized by sector"""
    return NEPSE_SECTORS

def get_stock_sector(symbol):
    """Get the sector of a given stock symbol"""
    for sector, stocks in NEPSE_SECTORS.items():
        if symbol in stocks:
            return sector
    return 'Others'

def generate_realistic_nepse_data(symbol, days=365):
    """Generate realistic NEPSE stock data based on sector and market conditions"""
    sector = get_stock_sector(symbol)
    min_price, max_price = SECTOR_BASE_PRICES.get(sector, (100, 500))
    
    # Create date range (excluding weekends for more realistic trading days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(days * 1.4))  # Extra days to account for weekends
    
    # Generate business days only
    dates = pd.bdate_range(start=start_date, end=end_date)
    dates = dates[:days]  # Limit to requested days
    
    # Set seed based on symbol for consistency
    np.random.seed(hash(symbol) % 2**32)
    
    # Base price within sector range
    base_price = np.random.uniform(min_price, max_price)
    
    # Generate more realistic price movements
    prices = []
    volumes = []
    current_price = base_price
    
    for i, date in enumerate(dates):
        # Market trends based on sector
        if sector == 'Banking':
            trend = 0.0002 + 0.001 * np.sin(i * 0.02)  # Stable with slight growth
            volatility = np.random.normal(0, 0.015)
        elif sector == 'Hydropower':
            trend = 0.0005 + 0.002 * np.sin(i * 0.01)  # Higher growth potential
            volatility = np.random.normal(0, 0.025)
        elif sector == 'Insurance':
            trend = 0.0003 + 0.0015 * np.sin(i * 0.015)
            volatility = np.random.normal(0, 0.02)
        elif sector == 'Microfinance':
            trend = 0.0001 + 0.003 * np.sin(i * 0.008)  # High volatility
            volatility = np.random.normal(0, 0.035)
        else:
            trend = 0.0002 + 0.001 * np.sin(i * 0.012)
            volatility = np.random.normal(0, 0.02)
        
        # Add some market events (random spikes/drops)
        if np.random.random() < 0.05:  # 5% chance of significant event
            volatility += np.random.choice([-0.08, 0.08])  # ±8% event
        
        # Calculate new price
        price_change = trend + volatility
        current_price = current_price * (1 + price_change)
        current_price = max(current_price, min_price * 0.3)  # Floor price
        current_price = min(current_price, max_price * 2)    # Ceiling price
        
        prices.append(current_price)
        
        # Generate volume based on price volatility
        base_volume = np.random.randint(1000, 50000)
        if abs(volatility) > 0.03:  # High volatility = high volume
            volume = base_volume * np.random.uniform(2, 5)
        else:
            volume = base_volume * np.random.uniform(0.5, 1.5)
        
        volumes.append(int(volume))
    
    # Create OHLC data
    df_data = []
    for i, (date, close_price, volume) in enumerate(zip(dates, prices, volumes)):
        # Generate realistic OHLC from close price
        daily_volatility = np.random.uniform(0.005, 0.03)
        
        open_price = close_price * np.random.uniform(0.98, 1.02)
        high_price = max(open_price, close_price) * np.random.uniform(1.0, 1 + daily_volatility)
        low_price = min(open_price, close_price) * np.random.uniform(1 - daily_volatility, 1.0)
        
        df_data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
    
    return pd.DataFrame(df_data)

def get_stock_data(symbol):
    """Get stock data for the given symbol"""
    try:
        # Try to get real data from Yahoo Finance first
        ticker_variants = [f"{symbol}.NP", f"{symbol}.NPL", symbol]
        
        for ticker in ticker_variants:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period="1y")
                
                if not df.empty and len(df) > 10:
                    df.reset_index(inplace=True)
                    df.columns = df.columns.str.lower()
                    
                    # Ensure we have the right column names
                    if 'date' not in df.columns and 'Date' in df.columns:
                        df = df.rename(columns={'Date': 'date'})
                    
                    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                    if all(col in df.columns for col in required_cols):
                        return df[required_cols]
                        
            except Exception:
                continue
                
    except Exception:
        pass
    
    # Fallback to realistic mock data
    return generate_realistic_nepse_data(symbol)

def get_market_summary():
    """Get overall market summary"""
    sectors = list(NEPSE_SECTORS.keys())
    summary = {}
    
    for sector in sectors:
        # Sample a few stocks from each sector for summary
        sample_stocks = NEPSE_SECTORS[sector][:3]
        sector_prices = []
        
        for stock in sample_stocks:
            df = get_stock_data(stock)
            if not df.empty:
                sector_prices.append(df['close'].iloc[-1])
        
        if sector_prices:
            summary[sector] = {
                'avg_price': np.mean(sector_prices),
                'stocks_count': len(NEPSE_SECTORS[sector])
            }
    
    return summary