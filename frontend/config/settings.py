# config/settings.py

# API Configuration
API_BASE_URL = "http://localhost:5000/api"
CACHE_TTL = 60  # Cache TTL in seconds to respect rate limits (60 req/min)
REQUEST_TIMEOUT = 10  # Seconds before timing out an API request

# App Configuration
APP_NAME = "NEPSE Analyzer"
APP_ICON = "📈"

# UI Configuration
PAGE_LAYOUT = "wide"
SIDEBAR_STATE = "expanded"
