from nepse import Nepse
import pandas as pd

nepse = Nepse()

def get_all_stocks():
    stocks = nepse.get_all_stocks()
    return sorted([stock['symbol'] for stock in stocks])

def get_stock_data(symbol):
    prices = nepse.get_price(symbol)
    df = pd.DataFrame(prices)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    return df
