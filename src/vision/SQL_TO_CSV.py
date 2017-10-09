import sqlite3
import csv
conn = sqlite3.connect('plant_db')
c = conn.cursor()
c.execute('SELECT * from features')
csvWriter = csv.writer(open("output.csv", "w"))

rows = c.fetchall()
csvWriter.writerows(rows)





for row in c.execute('SELECT * FROM features'):
        print row