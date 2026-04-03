import streamlit as st
import pandas as pd
import numpy as np
from services.api_service import get_company_list, get_company_details, get_history, get_analysis, get_ai_analysis
from utils.validators import is_valid_symbol
from utils.helpers import format_currency
from components.charts import render_advanced_chart

def render_stock_analysis():
    st.title("Stock Analysis")
    st.markdown("Detailed view and technical analysis for a specific stock.")
    
    company_list = get_company_list()
    if not company_list:
        st.error("Could not fetch company list.")
        return

    # Assuming company list is a dict or list of dicts with 'symbol' key
    symbols = [comp.get('symbol') for comp in company_list if comp.get('symbol')]
    
    selected_symbol = st.selectbox("Search Stock Symbol", options=[""] + sorted(symbols))
    
    if selected_symbol:
        if is_valid_symbol(selected_symbol, company_list):
            with st.spinner(f"Loading data for {selected_symbol}..."):
                details = get_company_details(selected_symbol)
                
            if details:
                sec = details.get('security', {})
                trade_dto = details.get('securityDailyTradeDto', {})
                
                comp_name = sec.get('securityName') or details.get('companyName', 'Unknown')
                sector_name = details.get('sectorName', 'N/A')
                
                st.subheader(f"{selected_symbol} - {comp_name}")
                st.write(f"**Sector:** {sector_name}")
                
                # Fetch key details safely
                last_price = trade_dto.get('lastTradedPrice') or details.get('lastTradedPrice') or details.get('closePrice') or 0
                prev_close = trade_dto.get('previousClose') or last_price
                
                if prev_close and last_price and prev_close > 0:
                    percent_change = round(((last_price - prev_close) / prev_close) * 100, 2)
                else:
                    percent_change = details.get('percentageChange') or 0
                    
                high_price = trade_dto.get('highPrice') or details.get('highPrice') or 0
                low_price = trade_dto.get('lowPrice') or details.get('lowPrice') or 0
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", format_currency(last_price))
                col2.metric("Change %", f"{percent_change}%", delta=percent_change)
                col3.metric("High", format_currency(high_price))
                col4.metric("Low", format_currency(low_price))
                
                st.markdown("---")
                
                # Technical Indicators Section
                st.subheader("Technical Analysis")
                
                # Technical Indicators Section
                st.subheader("Quantitative Analysis Engine")
                
                df_history = get_history(selected_symbol)
                
                if not df_history.empty and 'Close' in df_history.columns:
                    colA, colB = st.columns([2, 1])
                    with colA:
                        render_advanced_chart(df_history, title=f"Advanced Analysis ({selected_symbol})")
                    
                    with colB:
                        st.write("### Smart Signals")
                        analysis_data = get_analysis(selected_symbol)
                        if analysis_data and "trend" in analysis_data:
                            trend = analysis_data["trend"]
                            
                            # Color-coded metric
                            t_color = "red" if trend == "BEARISH" else "green" if trend == "BULLISH" else "orange"
                            st.markdown(f"**Market Trend:** <span style='color:{t_color}; font-weight:bold; font-size:1.2em;'>{trend}</span> ({analysis_data.get('strength', 'N/A')})", unsafe_allow_html=True)
                            
                            ind = analysis_data.get("indicators", {})
                            
                            rsi_val = ind.get('currentRSI')
                            rsi_val = float(rsi_val) if rsi_val is not None else 0.0
                            st.write(f"**Current RSI (14):** {rsi_val:.2f}")
                            
                            macd_data = ind.get('currentMACD')
                            macd_val = float(macd_data.get('MACD', 0)) if isinstance(macd_data, dict) and macd_data.get('MACD') is not None else 0.0
                            st.write(f"**MACD Line:** {macd_val:.2f}")
                            
                            st.markdown("**Detected Events:**")
                            for sig in analysis_data.get("signals", []):
                                st.caption(f"- {sig}")
                        else:
                            st.warning("Analysis engine offline.")
                            
                        st.markdown("---")
                        st.write("### AI Assistant")
                        if st.button(f"Generate Gemini Analysis for {selected_symbol}", type="primary"):
                            with st.spinner("Analyzing market structure with Gemini 1.5 Flash..."):
                                ai_res = get_ai_analysis(selected_symbol)
                                st.markdown("##### 🧠 AI Insights")
                                if ai_res and "error" not in ai_res:
                                    rec = ai_res.get("recommendation", "").upper()
                                    rec_c = "green" if "BUY" in rec else "red" if "SELL" in rec else "orange"
                                    st.markdown(f"**Action:** <span style='color:{rec_c}; font-weight:bold;'>{rec}</span>", unsafe_allow_html=True)
                                    st.write(f"**Risk Profile:** {ai_res.get('risk', 'Unknown')}")
                                    st.write(f"**Outlook:** {ai_res.get('outlook', 'Wait and see')}")
                                    st.info(f"**Reasoning:** {ai_res.get('reason', 'N/A')}")
                                else:
                                    st.error("Failed to communicate with AI Engine. Ensure GEMINI_API_KEY is properly set in the backend `.env`.")
                                
                                st.caption("⚠️ **Disclaimer:** This is AI-generated analysis. Not financial advice.")
                else:
                    st.info("No historical data available to compute technical indicators.")
            else:
                st.warning("No details found for the selected symbol.")
        else:
            st.warning("Invalid symbol selected.")
