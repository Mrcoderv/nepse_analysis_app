import streamlit as st
import pandas as pd
from services.api_service import get_portfolio, add_portfolio_stock
from components.tables import render_dataframe

def render_portfolio():
    st.title("💼 Portfolio Tracker")
    st.markdown("Manage your holdings and track live P&L directly from NEPSE live market quotes.")

    with st.expander("➕ Add New Holding"):
        with st.form("add_stock"):
            colA, colB, colC = st.columns(3)
            with colA:
                symbol = st.text_input("Symbol (e.g. NABIL)").upper().strip()
            with colB:
                qty = st.number_input("Quantity", min_value=1, step=1)
            with colC:
                buy_price = st.number_input("Buy Price (NPR)", min_value=0.0, step=0.1)
            
            submitted = st.form_submit_button("Add to Portfolio", type="primary")
            if submitted:
                if symbol and qty and buy_price:
                    res = add_portfolio_stock(symbol, qty, buy_price)
                    if isinstance(res, dict) and "error" in res:
                        st.error(res["error"])
                    else:
                        st.success(f"Successfully added {qty} shares of {symbol} at {buy_price} NPR.")
                        st.rerun() # Refresh to update view
                else:
                    st.error("Please fill all fields.")

    st.subheader("Your Holdings")
    portfolio_data = get_portfolio()
    
    if not portfolio_data:
        st.info("Your portfolio is empty. Add a stock to get started.")
        return

    df = pd.DataFrame(portfolio_data)
    
    total_inv = df['totalInvestment'].sum() if 'totalInvestment' in df.columns else 0
    total_val = df['currentValue'].sum() if 'currentValue' in df.columns else 0
    tot_pnl = total_val - total_inv
    pct_return = (tot_pnl / total_inv * 100) if total_inv > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Investment", f"Rs {total_inv:,.2f}")
    col2.metric("Current Value", f"Rs {total_val:,.2f}", f"{(total_val - total_inv):,.2f}", border=True)
    
    pnl_color = "normal" if tot_pnl >= 0 else "inverse"
    col3.metric("Profit / Loss", f"Rs {tot_pnl:,.2f}", f"{pct_return:.2f}%", delta_color=pnl_color, border=True)
    
    st.markdown("---")
    
    # Prettify table
    if 'symbol' in df.columns:
        disp_df = df[['symbol', 'quantity', 'buyPrice', 'currentPrice', 'totalInvestment', 'currentValue', 'profitLoss', 'percentReturn']]
        # Style dataframe slightly to enhance P&L readibility
        st.dataframe(
            disp_df.style.applymap(lambda x: 'color: green;' if x > 0 else ('color: red;' if x < 0 else ''), subset=['profitLoss', 'percentReturn']),
            use_container_width=True
        )
