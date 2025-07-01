import streamlit as st
import pandas as pd
from nepse_utils import (
    get_all_stocks, 
    get_stock_data, 
    get_stocks_by_sector, 
    get_stock_sector,
    get_market_summary
)
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
    .sector-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin: 1rem 0;
        padding: 0.5rem;
        background-color: #f0f8f0;
        border-left: 4px solid #2e8b57;
        border-radius: 4px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sector-summary {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stock-sector-badge {
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">📈 NEPSE Stock Analyzer & Predictor</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔧 Controls")
    
    # Market Overview
    st.subheader("📊 Market Overview")
    with st.expander("View Market Summary", expanded=False):
        market_summary = get_market_summary()
        for sector, data in market_summary.items():
            st.write(f"**{sector}**: {data['stocks_count']} stocks, Avg: NPR {data['avg_price']:.2f}")
    
    # Sector selection
    sectors_data = get_stocks_by_sector()
    selected_sector = st.selectbox(
        "🏢 Select Sector",
        options=['All'] + list(sectors_data.keys()),
        index=0,
        help="Filter stocks by sector"
    )
    
    # Stock selection based on sector
    if selected_sector == 'All':
        available_stocks = get_all_stocks()
        st.info(f"📊 Total stocks available: {len(available_stocks)}")
    else:
        available_stocks = sorted(sectors_data[selected_sector])
        st.info(f"📊 {selected_sector} stocks: {len(available_stocks)}")
    
    selected_stock = st.selectbox(
        "📊 Select a Stock",
        options=available_stocks,
        index=0,
        help="Choose a stock from the NEPSE (Nepal Stock Exchange)"
    )
    
    # Display stock sector
    if selected_stock:
        stock_sector = get_stock_sector(selected_stock)
        st.markdown(f'<div class="stock-sector-badge">Sector: {stock_sector}</div>', unsafe_allow_html=True)
    
    # Analysis options
    st.subheader("📋 Analysis Options")
    show_technical = st.checkbox("Technical Indicators", value=True)
    show_forecast = st.checkbox("Price Forecast", value=True)
    
    if show_forecast:
        forecast_days = st.slider("Forecast Days", min_value=7, max_value=90, value=30)
    
    # Sector Information
    st.subheader("🏢 Sector Information")
    if selected_sector != 'All':
        st.markdown(f'<div class="sector-header">{selected_sector}</div>', unsafe_allow_html=True)
        sector_stocks = sectors_data[selected_sector]
        st.write(f"**Total Companies**: {len(sector_stocks)}")
        
        # Show some sector stocks
        with st.expander(f"View all {selected_sector} stocks"):
            for i, stock in enumerate(sector_stocks, 1):
                st.write(f"{i}. {stock}")
    
    # Information
    st.subheader("ℹ️ Information")
    st.info("""
    **NEPSE Sectors Available:**
    - 🏦 Banking (30 companies)
    - 🛡️ Insurance (20 companies)  
    - ⚡ Hydropower (30 companies)
    - 🏭 Manufacturing (30 companies)
    - 🏨 Hotels & Tourism (20 companies)
    - 💰 Finance (20 companies)
    - 🏛️ Development Banks (20 companies)
    - 🏪 Microfinance (20 companies)
    - 📈 Trading (20 companies)
    - 🔧 Others (20 companies)
    """)

# Main content
if selected_stock:
    # Load data
    with st.spinner(f"Loading data for {selected_stock}..."):
        df = get_stock_data(selected_stock)
    
    if df is not None and not df.empty:
        # Stock header with sector badge
        stock_sector = get_stock_sector(selected_stock)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"📊 {selected_stock} - Stock Analysis")
        with col2:
            st.markdown(f'<div class="stock-sector-badge" style="text-align: center; margin-top: 1rem;">Sector: {stock_sector}</div>', unsafe_allow_html=True)
        
        # Key metrics
        metrics = calculate_stock_metrics(df)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Current Price", metrics['Current Price'], metrics['Change'])
        with col2:
            st.metric("Change %", metrics['Change %'])
        with col3:
            st.metric("52W High", metrics['52W High'])
        with col4:
            st.metric("52W Low", metrics['52W Low'])
        with col5:
            st.metric("Avg Volume", metrics['Average Volume'])
        
        # Additional metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Previous Close", metrics['Previous Close'])
        with col2:
            latest_volume = df['volume'].iloc[-1] if not df.empty else 0
            st.metric("Latest Volume", f"{latest_volume:,.0f}")
        with col3:
            price_range = df['high'].max() - df['low'].min()
            st.metric("Price Range", f"NPR {price_range:.2f}")
        with col4:
            trading_days = len(df)
            st.metric("Trading Days", f"{trading_days}")
        
        # Price chart
        st.subheader(f"📈 {selected_stock} - Price Chart & Volume")
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
                
                # Calculate some basic technical indicators
                if len(df) >= 20:
                    ma_20 = df['close'].rolling(20).mean().iloc[-1]
                    ma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"""
                        **Latest Trading Data:**
                        - **Date:** {latest_data['date'].strftime('%Y-%m-%d')}
                        - **Open:** NPR {latest_data['open']:.2f}
                        - **High:** NPR {latest_data['high']:.2f}
                        - **Low:** NPR {latest_data['low']:.2f}
                        - **Close:** NPR {latest_data['close']:.2f}
                        - **Volume:** {latest_data['volume']:,.0f}
                        """)
                    
                    with col2:
                        st.write(f"""
                        **Technical Indicators:**
                        - **20-day MA:** NPR {ma_20:.2f}
                        - **50-day MA:** NPR {ma_50:.2f if ma_50 else 'N/A'}
                        - **Price vs MA20:** {'Above' if latest_data['close'] > ma_20 else 'Below'}
                        - **Volatility:** {((df['high'] - df['low']) / df['close'] * 100).mean():.2f}%
                        """)
                else:
                    st.write("Insufficient data for technical analysis (need at least 20 days)")
        
        # Forecast
        if show_forecast:
            st.subheader(f"🔮 {selected_stock} - Price Forecast")
            
            with st.spinner("Generating AI-powered forecast..."):
                forecast_fig, forecast_data = forecast_stock(df, selected_stock, forecast_days)
            
            if forecast_fig is not None:
                st.plotly_chart(forecast_fig, use_container_width=True)
                
                # Forecast summary
                if forecast_data is not None and not forecast_data.empty:
                    with st.expander("📊 Detailed Forecast Analysis"):
                        last_actual = df['close'].iloc[-1]
                        last_forecast = forecast_data['yhat'].iloc[-1]
                        forecast_change = ((last_forecast - last_actual) / last_actual) * 100
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Current Price", f"NPR {last_actual:.2f}")
                        with col2:
                            st.metric("Forecasted Price", f"NPR {last_forecast:.2f}")
                        with col3:
                            st.metric("Expected Change", f"{forecast_change:.2f}%")
                        with col4:
                            trend = "Bullish 📈" if forecast_change > 0 else "Bearish 📉"
                            st.metric("Trend", trend)
                        
                        # Forecast confidence
                        if len(forecast_data) > 0:
                            confidence_range = forecast_data['yhat_upper'].iloc[-1] - forecast_data['yhat_lower'].iloc[-1]
                            st.write(f"""
                            **Forecast Details:**
                            - **Forecast Period:** {forecast_days} days
                            - **Confidence Range:** NPR {confidence_range:.2f}
                            - **Upper Bound:** NPR {forecast_data['yhat_upper'].iloc[-1]:.2f}
                            - **Lower Bound:** NPR {forecast_data['yhat_lower'].iloc[-1]:.2f}
                            - **Sector:** {stock_sector}
                            """)
                        
                        st.warning("⚠️ **Investment Disclaimer:** This forecast is generated using AI and historical data patterns. It should not be used as the sole basis for investment decisions. Always consult with qualified financial advisors and conduct your own research before making investment choices.")
        
        # Sector comparison
        st.subheader(f"🏢 {stock_sector} Sector Analysis")
        sector_stocks = get_stocks_by_sector()[stock_sector]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"""
            **Sector Information:**
            - **Sector:** {stock_sector}
            - **Total Companies:** {len(sector_stocks)}
            - **Selected Stock:** {selected_stock}
            - **Market Position:** {sector_stocks.index(selected_stock) + 1} of {len(sector_stocks)} (alphabetical)
            """)
        
        with col2:
            # Show other stocks in the same sector
            st.write("**Other companies in this sector:**")
            other_stocks = [s for s in sector_stocks if s != selected_stock][:10]
            for stock in other_stocks:
                st.write(f"• {stock}")
            if len(sector_stocks) > 11:
                st.write(f"... and {len(sector_stocks) - 11} more")
        
        # Data table
        with st.expander("📋 Historical Data"):
            # Show data with better formatting
            display_df = df.copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df['open'] = display_df['open'].round(2)
            display_df['high'] = display_df['high'].round(2)
            display_df['low'] = display_df['low'].round(2)
            display_df['close'] = display_df['close'].round(2)
            display_df['volume'] = display_df['volume'].astype(int)
            
            st.dataframe(
                display_df.tail(30), 
                use_container_width=True,
                column_config={
                    "date": "Date",
                    "open": st.column_config.NumberColumn("Open (NPR)", format="%.2f"),
                    "high": st.column_config.NumberColumn("High (NPR)", format="%.2f"),
                    "low": st.column_config.NumberColumn("Low (NPR)", format="%.2f"),
                    "close": st.column_config.NumberColumn("Close (NPR)", format="%.2f"),
                    "volume": st.column_config.NumberColumn("Volume", format="%d")
                }
            )
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Complete Data as CSV",
                data=csv,
                file_name=f"{selected_stock}_{stock_sector}_stock_data.csv",
                mime="text/csv"
            )
    
    else:
        st.error(f"❌ Unable to load data for {selected_stock}. Please try another stock or check your internet connection.")

else:
    # Welcome screen
    st.info("👆 Please select a sector and stock from the sidebar to begin analysis.")
    
    # Show sector overview
    st.subheader("🏢 NEPSE Sectors Overview")
    sectors_data = get_stocks_by_sector()
    
    # Create sector cards
    cols = st.columns(3)
    for i, (sector, stocks) in enumerate(sectors_data.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="sector-summary">
                <h4>{sector}</h4>
                <p><strong>{len(stocks)}</strong> companies</p>
                <p>Examples: {', '.join(stocks[:3])}</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📈 <strong>NEPSE Stock Analyzer & Predictor</strong> | Built with Streamlit & AI</p>
    <p><strong>Sectors:</strong> Banking • Insurance • Hydropower • Manufacturing • Hotels & Tourism • Finance • Development Banks • Microfinance • Trading • Others</p>
    <p><small>⚠️ <strong>Disclaimer:</strong> This tool is for educational and research purposes only. Stock forecasts are AI-generated predictions based on historical data and should not be used as the sole basis for investment decisions. Always consult with qualified financial advisors and conduct thorough research before making any investment choices. Past performance does not guarantee future results.</small></p>
</div>
""", unsafe_allow_html=True)