#!/usr/bin/python
# -*- coding: utf-8 -*-

import sqlite3  as lite
import sys

con = None
try:
    con = lite.connect('plant.db')

    cur = con.cursor()
    cur.execute('SELECT SQLITE_VERSION()')
    data = cur.fetchone()

    print("SQLite version", data)

except lite.Error, e:

    print("Error", e.args[0])
    sys.exit(1)

finally:

    if con:
        con.close()
