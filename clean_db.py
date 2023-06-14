"""
Clean sqlite database: remove all images which were detected two or more weeks ago.
"""

import sqlite3
import datetime

def old_datetime(date):
    image_datetime = datetime.datetime.strptime("{}".format(date), "%Y.%m.%d %H:%M:%S")
    now  = datetime.datetime.now()
    duration = now - image_datetime
    duration_in_s = duration.total_seconds()
    if duration_in_s > 1209600:     # 1209600 sec in two weeks. Set 0 for clean all data from db.
        return date
    else:
        return None

con = sqlite3.connect('SavedFaces.db')
cur = con.cursor()

sql_exract = """SELECT * from SavedFaces"""
sql_delete = """DELETE FROM SavedFaces WHERE datetime = (?);"""
vacuum = """VACUUM;"""

# cleaning
try:
    cur.execute(sql_exract)
    record = cur.fetchall()
    print('Database size before cleaning:', len(record))
    len_before = len(record)
    for row in record:
        date = row[2]
        del_datetime = old_datetime(date)
        if del_datetime is not None:
            cur.execute(sql_delete, (del_datetime,))
except sqlite3.Error as error:
    print("Error1: ", error)
else:
    con.commit()

# after cleaning
try:
    cur.execute(vacuum)     # for clean useless memory in database
    cur.execute(sql_exract)
    record = cur.fetchall()
    len_after = len(record)
except sqlite3.Error as error:
    print("Error2: ", error)
else:
    len_deleted = len_before-len_after
    print('\nSuccess. Database size after cleaning:', len(record), '\n{} objects deleted!'.format(len_deleted))

cur.close()
con.close()