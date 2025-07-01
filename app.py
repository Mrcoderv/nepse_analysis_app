import streamlit as st
import pandas as pd
from nepse_utils import get_all_stocks, get_stock_data
from analysis_utils import (
    plot_stock, 
    plot_technical_indicators, 
    forecast_stock, 
    calculate_stock_metrics
)

# Page configuration
st.set_page_config(
    page_title="NEPSE Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">📈 NEPSE Stock Analyzer & Predictor</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔧 Controls")
    
    # Stock selection
    stock_list = get_all_stocks()
    selected_stock = st.selectbox(
        "📊 Select a Stock",
        options=stock_list,
        index=0,
        help="Choose a stock from the NEPSE (Nepal Stock Exchange)"
    )
    
    # Analysis options
    st.subheader("📋 Analysis Options")
    show_technical = st.checkbox("Technical Indicators", value=True)
    show_forecast = st.checkbox("Price Forecast", value=True)
    
    if show_forecast:
        forecast_days = st.slider("Forecast Days", min_value=7, max_value=90, value=30)
    
    # Information
    st.subheader("ℹ️ Information")
    st.info("""
    This app provides:
    - Real-time stock data visualization
    - Technical analysis indicators
    - Price forecasting using AI
    - Key financial metrics
    """)

# Main content
if selected_stock:
    # Load data
    with st.spinner(f"Loading data for {selected_stock}..."):
        df = get_stock_data(selected_stock)
    
    if df is not None and not df.empty:
        # Key metrics
        st.subheader(f"📊 {selected_stock} - Key Metrics")
        metrics = calculate_stock_metrics(df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", metrics['Current Price'], metrics['Change'])
        with col2:
            st.metric("Change %", metrics['Change %'])
        with col3:
            st.metric("52W High", metrics['52W High'])
        with col4:
            st.metric("52W Low", metrics['52W Low'])
        
        # Price chart
        st.subheader(f"📈 {selected_stock} - Price Chart")
        price_fig = plot_stock(df, selected_stock)
        st.plotly_chart(price_fig, use_container_width=True)
        
        # Technical indicators
        if show_technical:
            st.subheader(f"🔍 {selected_stock} - Technical Analysis")
            tech_fig = plot_technical_indicators(df, selected_stock)
            st.plotly_chart(tech_fig, use_container_width=True)
            
            # Technical summary
            with st.expander("📝 Technical Analysis Summary"):
                latest_data = df.iloc[-1]
                st.write(f"""
                **Latest Trading Data:**
                - **Date:** {latest_data['date'].strftime('%Y-%m-%d')}
                - **Open:** NPR {latest_data['open']:.2f}
                - **High:** NPR {latest_data['high']:.2f}
                - **Low:** NPR {latest_data['low']:.2f}
                - **Close:** NPR {latest_data['close']:.2f}
                - **Volume:** {latest_data['volume']:,.0f}
                """)
        
        # Forecast
        if show_forecast:
            st.subheader(f"🔮 {selected_stock} - Price Forecast")
            
            with st.spinner("Generating forecast..."):
                forecast_fig, forecast_data = forecast_stock(df, selected_stock, forecast_days)
            
            if forecast_fig is not None:
                st.plotly_chart(forecast_fig, use_container_width=True)
                
                # Forecast summary
                if forecast_data is not None and not forecast_data.empty:
                    with st.expander("📊 Forecast Summary"):
                        last_actual = df['close'].iloc[-1]
                        last_forecast = forecast_data['yhat'].iloc[-1]
                        forecast_change = ((last_forecast - last_actual) / last_actual) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Price", f"NPR {last_actual:.2f}")
                        with col2:
                            st.metric("Forecasted Price", f"NPR {last_forecast:.2f}")
                        with col3:
                            st.metric("Expected Change", f"{forecast_change:.2f}%")
                        
                        st.warning("⚠️ **Disclaimer:** This forecast is based on historical data and should not be used as the sole basis for investment decisions.")
        
        # Data table
        with st.expander("📋 Raw Data"):
            st.dataframe(df.tail(20), use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"{selected_stock}_stock_data.csv",
                mime="text/csv"
            )
    
    else:
        st.error(f"❌ Unable to load data for {selected_stock}. Please try another stock.")

else:
    st.info("👆 Please select a stock from the sidebar to begin analysis.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📈 NEPSE Stock Analyzer & Predictor | Built with Streamlit</p>
    <p><small>⚠️ This tool is for educational purposes only. Always consult with financial advisors before making investment decisions.</small></p>
</div>
""", unsafe_allow_html=True)