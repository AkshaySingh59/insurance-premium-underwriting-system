import sqlite3
import pandas as pd

def create_db():
    conn = sqlite3.connect("predictions.db")

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        age INTEGER,
        bmi REAL,
        premium REAL,
        risk TEXT
    )
    """)

    conn.commit()
    conn.close()


def save_prediction(age, bmi, premium, risk):

    conn = sqlite3.connect("predictions.db")

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO predictions(
            age,bmi,premium,risk
        )
        VALUES(?,?,?,?)
        """,
        (age, bmi, premium, risk)
    )

    conn.commit()
    conn.close()


def get_predictions():

    conn = sqlite3.connect("predictions.db")

    df = pd.read_sql(
        "SELECT * FROM predictions ORDER BY id DESC",
        conn
    )

    conn.close()

    return df