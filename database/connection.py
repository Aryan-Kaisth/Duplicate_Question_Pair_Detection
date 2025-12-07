import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table, select

# Load environment variables
load_dotenv()

# Get DB credentials
DB_USER = os.getenv("DB_USER")
raw_password = os.getenv("DB_PASS")
if raw_password is None:
    raise ValueError("DB_PASS not found in environment variables")
DB_PASSWORD = quote_plus(raw_password)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME")

# Create SQLAlchemy engine
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

# Metadata object for table reflection
metadata = MetaData()

def get_connection():
    """
    Returns a SQLAlchemy connection object.
    Use with 'with' statement for safe connection handling.
    Example:
        with get_connection() as conn:
            result = conn.execute(text("SELECT 1"))
    """
    return engine.connect()

def get_table(table_name: str):
    """
    Reflect and return an existing table object.
    """
    return Table(table_name, metadata, autoload_with=engine)