import streamlit as st

def render_sidebar():
    """Renders the main navigation sidebar."""
    with st.sidebar:
        st.title("📈 NEPSE Analyzer")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["Dashboard", "Stock Analysis", "Portfolio Tracker (Beta)"]
        )
        
        st.markdown("---")
        st.info("Data powered by unofficial NEPSE API. \nCaches data natively to respect rate limits.")
        
        st.caption("v1.0.0 Alpha")
        
        return page
