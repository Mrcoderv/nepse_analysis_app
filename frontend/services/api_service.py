import requests
import pandas as pd
import streamlit as st
from config.settings import API_BASE_URL, CACHE_TTL, REQUEST_TIMEOUT

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_api_data(endpoint: str, params: dict = None) -> list:
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            st.warning("Rate limit exceeded. Waiting for cache reset.")
        else:
            st.error(f"HTTP Error: {e}")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Network Error: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected Error: {e}")
        return []

@st.cache_data(ttl=CACHE_TTL, show_spinner="Fetching Live Market...")
def get_live_market() -> pd.DataFrame:
    data = fetch_api_data("/LiveMarket")
    return pd.DataFrame(data)

@st.cache_data(ttl=CACHE_TTL, show_spinner="Fetching Gainers...")
def get_top_gainers() -> pd.DataFrame:
    data = fetch_api_data("/TopGainers")
    return pd.DataFrame(data)

@st.cache_data(ttl=CACHE_TTL, show_spinner="Fetching Losers...")
def get_top_losers() -> pd.DataFrame:
    data = fetch_api_data("/TopLosers")
    return pd.DataFrame(data)

@st.cache_data(ttl=CACHE_TTL, show_spinner="Fetching Index...")
def get_nepse_index() -> pd.DataFrame:
    data = fetch_api_data("/NepseIndex")
    return pd.DataFrame(data)

@st.cache_data(ttl=CACHE_TTL * 10, show_spinner=False) # Cache list longer
def get_company_list() -> list:
    return fetch_api_data("/CompanyList")

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_company_details(symbol: str) -> dict:
    data = fetch_api_data(f"/CompanyDetails/{symbol}")
    if isinstance(data, list) and len(data) > 0:
        return data[0]
    return data if isinstance(data, dict) else {}

@st.cache_data(ttl=CACHE_TTL * 2, show_spinner=False)
def get_history(symbol: str) -> pd.DataFrame:
    data = fetch_api_data(f"/history/{symbol}")
    if isinstance(data, dict) and "content" in data:
        df = pd.DataFrame(data["content"])
        if not df.empty and "businessDate" in df.columns:
            df["Date"] = pd.to_datetime(df["businessDate"])
            df = df.sort_values(by="Date").reset_index(drop=True)
            df.rename(columns={
                "closePrice": "Close",
                "openPrice": "Open",
                "highPrice": "High",
                "lowPrice": "Low"
            }, inplace=True, errors="ignore")
            return df
    return pd.DataFrame()

@st.cache_data(ttl=30, show_spinner=False)
def get_portfolio() -> list:
    data = fetch_api_data("/portfolio")
    return data if isinstance(data, list) else []

def add_portfolio_stock(symbol, quantity, buyPrice) -> dict:
    url = f"{API_BASE_URL}/portfolio"
    try:
        response = requests.post(url, json={"symbol": symbol, "quantity": quantity, "buyPrice": buyPrice}, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            return {"error": response.json().get('error', 'Failed to add')}
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=CACHE_TTL, show_spinner="Computing analytics...")
def get_analysis(symbol: str) -> dict:
    data = fetch_api_data(f"/analyze/{symbol}")
    return data if isinstance(data, dict) else {}

@st.cache_data(ttl=21600, show_spinner="Asking Gemini...")
def get_ai_analysis(symbol: str) -> dict:
    data = fetch_api_data(f"/ai-analysis/{symbol}")
    return data if isinstance(data, dict) else {}
