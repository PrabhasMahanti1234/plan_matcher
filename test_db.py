import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_pass = os.getenv("DB_PASSWORD")

print(f"Connecting to host='{db_host}', port='{db_port}' as user='{db_user}'...")
if not db_host or not db_port:
    print("ERROR: DB_HOST or DB_PORT is missing from .env!")
try:
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_pass,
        connect_timeout=5
    )
    print("OK: Database connection successful!")
    conn.close()
except Exception as e:
    print(f"ERROR: {e}")
