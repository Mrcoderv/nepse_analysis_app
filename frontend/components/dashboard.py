import streamlit as st
import pandas as pd
from services.api_service import get_live_market, get_top_gainers, get_top_losers, get_nepse_index
from components.tables import render_dataframe

def render_dashboard():
    """Renders the main dashboard of the application."""
    st.title("Market Dashboard")
    st.markdown("Current status of the Nepal Stock Exchange (NEPSE).")
    
    # Check index
    index_df = get_nepse_index()
    if not index_df.empty:
        st.subheader("NEPSE Index Summary")
        render_dataframe(index_df)

    st.markdown("---")

    # Gainers & Losers side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Gainers")
        gainers_df = get_top_gainers()
        render_dataframe(gainers_df)
        
    with col2:
        st.subheader("Top Losers")
        losers_df = get_top_losers()
        render_dataframe(losers_df)
    
    st.markdown("---")
    
    # Live market table
    st.subheader("Live Market")
    live_df = get_live_market()
    render_dataframe(live_df)
