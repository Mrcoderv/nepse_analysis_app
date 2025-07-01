import streamlit as st
from nepse_utils import get_all_stocks, get_stock_data
from analysis_utils import plot_stock, forecast_stock

st.title("📈 NEPSE Stock Analyzer & Predictor")

stock_list = get_all_stocks()
selected_stock = st.selectbox("Select a stock", stock_list)

if selected_stock:
    df = get_stock_data(selected_stock)
    st.subheader(f"{selected_stock} Price Chart")
    plot_stock(df)

    st.subheader("📊 Forecast")
    forecast_fig = forecast_stock(df)
    st.plotly_chart(forecast_fig)
