import streamlit as st
import pandas as pd

def render_dataframe(df: pd.DataFrame, title: str = None):
    """
    Renders a Pandas DataFrame nicely in Streamlit.
    """
    if title:
        st.subheader(title)
    
    if df is None or df.empty:
        st.info("No data available.")
        return
    
    st.dataframe(df, use_container_width=True, hide_index=True)
