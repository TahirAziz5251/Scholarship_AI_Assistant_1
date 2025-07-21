import hashlib
import sqlite3
# from backend.database import SessionLocal
from models import User, Base
from sqlalchemy.orm import Session
import bcrypt
from pathlib import Path

# Path to SQLite DB file
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "users.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def hash_password(password):
    """Hash plain text password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    """Initialize DB with users table if it doesn't exist"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL
            )
        """)
        conn.commit()

def register_user(username, password):
    """Register new user; return True if successful, False if user exists"""
    hashed = hash_password(password)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hashed))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # username already exists

def check_login(username, password):
    """Check login credentials"""
    hashed = hash_password(password)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        return row and row[0] == hashed

def change_password(username, new_password):
    """Change password for existing user"""
    new_hashed = hash_password(new_password)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET password_hash = ? WHERE username = ?", (new_hashed, username))
        conn.commit()
        return cursor.rowcount > 0  # True if user existed and password changed

def get_user(username):
    """Fetch user by username"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        return row[0] if row else None

  