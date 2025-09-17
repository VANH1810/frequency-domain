from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DB_URL = "sqlite:///alerts.db"  # hoặc PostgreSQL URL

engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
