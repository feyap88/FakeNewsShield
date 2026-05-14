import sqlite3
from datetime import datetime

def get_db_connection():
    conn = sqlite3.connect('fakenewsshield.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS evaluation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            text_content TEXT,
            prediction TEXT,
            confidence REAL,
            explanation TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database
init_database()
print("✅ Database initialized: fakenewsshield.db")