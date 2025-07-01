# 📈 NEPSE Stock Analyzer & Predictor

A comprehensive web application for analyzing Nepal Stock Exchange (NEPSE) stocks with advanced technical analysis and AI-powered price forecasting.

## 🌟 Features

- **Real-time Stock Data**: Access to NEPSE stock prices and trading data
- **Interactive Charts**: Candlestick charts with volume analysis
- **Technical Indicators**: 
  - Moving Averages (20-day, 50-day)
  - Bollinger Bands
  - Relative Strength Index (RSI)
- **AI-Powered Forecasting**: Price predictions using Facebook Prophet
- **Key Metrics**: Current price, changes, 52-week high/low, volume
- **Data Export**: Download historical data as CSV

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nepse-stock-analyzer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

## 📊 How to Use

1. **Select a Stock**: Choose from the dropdown list of available NEPSE stocks
2. **View Key Metrics**: See current price, changes, and trading statistics
3. **Analyze Charts**: Interactive price charts with technical indicators
4. **Generate Forecasts**: AI-powered price predictions for up to 90 days
5. **Export Data**: Download historical data for further analysis

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Forecasting**: Facebook Prophet
- **Data Source**: Yahoo Finance (with fallback to mock data)

## 📈 Available Stocks

The application includes major NEPSE stocks including:
- Banking sector: NABIL, SCB, HBL, EBL, BOKL
- Insurance: NLICL, LICN, PICL, LGIL
- Hydropower: HIDCL, CHCL, SJCL, UNHPL
- And many more...

## ⚠️ Disclaimer

This application is for educational and informational purposes only. The forecasts and analysis provided should not be used as the sole basis for investment decisions. Always consult with qualified financial advisors before making investment choices.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🔧 Troubleshooting

If you encounter issues:

1. Ensure all dependencies are installed correctly
2. Check your internet connection for data fetching
3. Try refreshing the application
4. Check the console for error messages

For additional support, please open an issue in the repository.