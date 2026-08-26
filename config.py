import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "supportai-secure-dev-key-2026")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "support_tickets.csv")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    DATABASE_PATH = os.path.join(BASE_DIR, "instance", "supportai.db")
