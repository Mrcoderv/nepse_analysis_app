import streamlit as st
from config.settings import APP_NAME, APP_ICON, PAGE_LAYOUT, SIDEBAR_STATE
from services.cache_service import initialize_session_state
from components.sidebar import render_sidebar
from components.dashboard import render_dashboard
from components.stock_detail import render_stock_analysis

def load_css():
    """Loads custom CSS styles."""
    try:
        with open("assets/styles.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=APP_ICON,
        layout=PAGE_LAYOUT,
        initial_sidebar_state=SIDEBAR_STATE
    )

    load_css()
    initialize_session_state()

    # Determine route based on sidebar selection
    page = render_sidebar()

    if page == "Dashboard":
        render_dashboard()
    elif page == "Stock Analysis":
        render_stock_analysis()
    elif page == "Portfolio Tracker (Beta)":
        from components.portfolio import render_portfolio
        render_portfolio()

if __name__ == "__main__":
    main()
