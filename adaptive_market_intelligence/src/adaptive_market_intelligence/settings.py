import os

APP_ENV = os.getenv("APP_ENV", "dev")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./adaptive_market_intelligence.db")
