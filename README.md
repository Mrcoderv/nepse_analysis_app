# NEPSE Analyzer Platform

An AI-powered, two-tier fintech platform for analyzing the Nepal Stock Exchange (NEPSE). Features live market data, advanced technical trading signals, persistent portfolio tracking, and sophisticated Gemini AI intelligence for market insights.

## Architecture
- **Backend**: Node.js + Express
  - Integrates `@rumess/nepse-api` to pull accurate live and historical market data natively.
  - Implements local JSON persistence for Portfolio Tracking APIs.
  - Contains an automated mathematical analysis engine computing SMA, MACD, Bollinger Bands, and RSI over historic pricing.
  - Plugs into Gemini 1.5 Flash to automatically dictate structured JSON Buy/Sell/Hold recommendations.
- **Frontend**: Streamlit
  - Consumes APIs from the Node Backend.
  - Renders professional Plotly Candlestick and RSI subplot charts cleanly.
  - Designed with an aggressive dark theme utilizing custom configuration protocols.

## Getting Started

### 1. Requirements
Ensure you have `Node.js` (v18+) and `Python` (v3.10+) installed.

### 2. Backend Setup
```bash
cd backend
npm install
```
Configure your `.env`:
```
GEMINI_API_KEY=your_gemini_api_key_here
PORT=5000
```
Run the API Service:
```bash
node index.js
```
The server will boot locally on Port `5000`.

### 3. Frontend Setup
```bash
cd frontend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Run the Interface:
```bash
streamlit run app.py
```
You can then view the platform seamlessly over your web browser at Port `8501`.
