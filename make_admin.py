import sqlite3
import os

BASE_DIR = os.path.join(os.getcwd(), 'app')
DATABASE = os.path.join(BASE_DIR, 'users.db')

conn = sqlite3.connect(DATABASE)
conn.execute("UPDATE users SET role = 'admin' WHERE username = 'admin'")
conn.commit()
conn.close()

print('Admin role assigned to user "admin"')