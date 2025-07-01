from prophet import Prophet
import pandas as pd
import plotly.graph_objs as go

def plot_stock(df):
    go_fig = go.Figure()
    go_fig.add_trace(go.Scatter(x=df['date'], y=df['close'], name="Close Price"))
    go_fig.update_layout(title="Stock Price History", xaxis_title="Date", yaxis_title="Price")
    return st.plotly_chart(go_fig)

def forecast_stock(df):
    df = df.rename(columns={'date': 'ds', 'close': 'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    fig = model.plot(forecast)
    return fig
