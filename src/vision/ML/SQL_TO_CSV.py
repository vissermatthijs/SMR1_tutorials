import csv
import sqlite3

csvWriter = csv.writer(open("plant_db.csv", "w"))

conn = sqlite3.connect('scripts/plant_db_1_v2')
c = conn.cursor()
c.execute('SELECT * from features')

rows = c.fetchall()
for row in rows:
    #print(row)
    rowsliced=row[1:11]+row[12:18]
    a=(0,)
    rowindex=a+rowsliced
    print(rowindex)
    # do your stuff
    csvWriter.writerow(rowindex)

conn = sqlite3.connect('scripts/plant_db_2_v2')
c = conn.cursor()
c.execute('SELECT * from features')

rows = c.fetchall()
for row in rows:
    #print(row)
    rowsliced = row[1:11] + row[12:18]
    a=(1,)
    rowindex=a+rowsliced
    print(rowindex)
    # do your stuff
    csvWriter.writerow(rowindex)

conn = sqlite3.connect('scripts/plant_db_3_v2')
c = conn.cursor()
c.execute('SELECT * from features')

rows = c.fetchall()
for row in rows:
    #print(row)
    rowsliced = row[1:11] + row[12:18]
    a=(2,)
    rowindex=a+rowsliced
    print(rowindex)
    # do your stuff
    csvWriter.writerow(rowindex)


'''
conn = sqlite3.connect('scripts/plant_db_4')
c = conn.cursor()
c.execute('SELECT * from features')

rows = c.fetchall()
for row in rows:
    #print(row)
    rowsliced = row[1:11] + row[12:18]
    a=(3,)
    rowindex=a+rowsliced
    print(rowindex)
    # do your stuff
    csvWriter.writerow(rowindex)

'''

