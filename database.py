"""SQLite-based authentication, prediction history, and admin analytics."""
import hashlib
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "app_data.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            full_name TEXT,
            email TEXT,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            pregnancies REAL, glucose REAL, blood_pressure REAL,
            skin_thickness REAL, insulin REAL, bmi REAL,
            pedigree REAL, age REAL,
            prediction INTEGER NOT NULL,
            risk_score REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """)
        # Migration: add is_admin if missing (for older DB files)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
        if "is_admin" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 200_000
    ).hex()


def signup(username: str, password: str, full_name: str = "", email: str = ""):
    if len(username) < 3 or len(password) < 6:
        return False, "Username ≥3 chars, password ≥6 chars."
    salt = secrets.token_hex(16)
    pw_hash = _hash(password, salt)
    try:
        with get_conn() as conn:
            # First registered user becomes admin automatically
            count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            is_admin = 1 if count == 0 else 0
            conn.execute(
                "INSERT INTO users (username, full_name, email, password_hash, salt, is_admin, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (username, full_name, email, pw_hash, salt, is_admin,
                 datetime.utcnow().isoformat()),
            )
        msg = ("🎉 Account created. You are the system admin." if is_admin
               else "Account created. Please log in.")
        return True, msg
    except sqlite3.IntegrityError:
        return False, "Username already taken."


def login(username: str, password: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?",
                           (username,)).fetchone()
    if not row or _hash(password, row["salt"]) != row["password_hash"]:
        return None
    return {"id": row["id"], "username": row["username"],
            "full_name": row["full_name"], "email": row["email"],
            "is_admin": bool(row["is_admin"])}


def save_prediction(user_id: int, inputs: dict, prediction: int, risk: float):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO predictions
              (user_id, pregnancies, glucose, blood_pressure, skin_thickness,
               insulin, bmi, pedigree, age, prediction, risk_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, inputs["Pregnancies"], inputs["Glucose"],
              inputs["BloodPressure"], inputs["SkinThickness"],
              inputs["Insulin"], inputs["BMI"],
              inputs["DiabetesPedigreeFunction"], inputs["Age"],
              int(prediction), float(risk), datetime.utcnow().isoformat()))


def get_user_predictions(user_id: int):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)).fetchall()
    return [dict(r) for r in rows]


# ── Admin queries ──
def admin_stats():
    with get_conn() as conn:
        users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        preds = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        diabetic = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction=1").fetchone()[0]
        avg_risk = conn.execute(
            "SELECT COALESCE(AVG(risk_score),0) FROM predictions").fetchone()[0]
    return {"users": users, "predictions": preds,
            "diabetic": diabetic, "avg_risk": avg_risk}


def admin_all_users():
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT u.id, u.username, u.full_name, u.email, u.is_admin, u.created_at,
                   COUNT(p.id) AS prediction_count,
                   COALESCE(AVG(p.risk_score), 0) AS avg_risk
            FROM users u LEFT JOIN predictions p ON p.user_id = u.id
            GROUP BY u.id ORDER BY u.created_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def admin_all_predictions():
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT p.*, u.username FROM predictions p
            JOIN users u ON u.id = p.user_id
            ORDER BY p.created_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


init_db()
