import sqlite3

conn = sqlite3.connect("signup.db")

cur = conn.cursor()

cur.execute("""

CREATE TABLE IF NOT EXISTS users(

id INTEGER PRIMARY KEY AUTOINCREMENT,

name TEXT,

username TEXT UNIQUE,

email TEXT UNIQUE,

phone TEXT,

password TEXT

)

""")

conn.commit()

conn.close()

print("Database Created")

