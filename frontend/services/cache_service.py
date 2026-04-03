import streamlit as st

def initialize_session_state():
    """Initializes necessary variables in the Streamlit session state."""
    # Portfolio tracking
    if 'portfolio' not in st.session_state:
        st.session_state['portfolio'] = []

    # Navigation or selection states
    if 'selected_symbol' not in st.session_state:
        st.session_state['selected_symbol'] = None

    if 'last_update' not in st.session_state:
        st.session_state['last_update'] = None
