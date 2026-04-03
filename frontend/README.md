# 📈 NEPSE Analyzer

A complete, production-ready Streamlit application for analyzing data from the Nepal Stock Exchange (NEPSE) using an unofficial REST API.

## 🌟 Features
- **Live Market Dashboard**: View Live Market, Top Gainers, Top Losers, and the NEPSE Index natively.
- **Stock Analysis**: Look up specific companies and see price changes, high/low, and technical signals (RSI/MACD simulated locally for demonstration).
- **Responsive UI**: Clean UI, custom CSS styling, and unified layout mimicking modern finance platforms.
- **Efficient Caching**: Built-in Streamlit TTL caching respects strict rate limits (60 req/min).

## 🚀 Setup Instructions

1. **Navigate to the directory**:
   ```bash
   cd NepseAi
   ```

2. **Create a Virtual Environment** (Optional but Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure
- `app.py`: Main entry point.
- `config/`: App and API configurations.
- `components/`: UI pieces (Dashboard, Sidebar, Charts, Tables, Stock Detail).
- `services/`: API integration and Cache/WebSocket state management.
- `utils/`: Helpers, Validators, and Indicators.
- `assets/`: Custom CSS styles.
