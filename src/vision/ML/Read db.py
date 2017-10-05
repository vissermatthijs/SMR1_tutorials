import sqlite3
conn = sqlite3.connect('example.db')
c = conn.cursor()

for row in c.execute('SELECT * FROM poep'):
        print row