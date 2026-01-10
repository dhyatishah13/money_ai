import sqlite3

# ---------------- DATABASE CONNECTION ----------------
def get_connection():
    return sqlite3.connect("moneyleak.db", check_same_thread=False)


# ---------------- CREATE TABLES ----------------
def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS budgets (
        user_email TEXT,
        monthly_budget INTEGER,
        food INTEGER,
        rent INTEGER,
        entertainment INTEGER,
        savings INTEGER
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS expenses (
        user_email TEXT,
        category TEXT,
        amount INTEGER,
        expense_date TEXT
    )
    """)

    conn.commit()
    conn.close()


# ---------------- USER ----------------
def save_user(username, email, password):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT OR IGNORE INTO users (username, email, password)
    VALUES (?, ?, ?)
    """, (username, email, password))

    conn.commit()
    conn.close()


# ---------------- BUDGET ----------------
def save_budget(email, total, food, rent, entertainment, savings):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM budgets WHERE user_email = ?", (email,))
    cursor.execute("""
    INSERT INTO budgets VALUES (?, ?, ?, ?, ?, ?)
    """, (email, total, food, rent, entertainment, savings))

    conn.commit()
    conn.close()


def get_budget(email):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT monthly_budget, food, rent, entertainment, savings
    FROM budgets WHERE user_email = ?
    """, (email,))

    data = cursor.fetchone()
    conn.close()
    return data


# ---------------- EXPENSES ----------------
def save_expense(email, category, amount, expense_date):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO expenses VALUES (?, ?, ?, ?)
    """, (email, category, amount, str(expense_date)))

    conn.commit()
    conn.close()


def get_expenses(email):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT category, amount, expense_date
    FROM expenses
    WHERE user_email = ?
    """, (email,))   # ✅ FIXED

    data = cursor.fetchall()
    conn.close()
    return data
