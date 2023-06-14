"""
Extract all images from sqlite database to saved_folder.
"""

import sqlite3

saved_folder = 'saved_faces/'
con = sqlite3.connect('SavedFaces.db')
cur = con.cursor()

sql_exract = """SELECT * from SavedFaces"""

try:
    cur.execute(sql_exract)
    record = cur.fetchall()
    print('Database size:', len(record))
    for row in record:
        name  = row[0]
        camera = row[1]
        datetime = row[2]
        photo = row[3]
        photo_path = saved_folder + name + '_' + camera + '.jpg'
        with open(photo_path, 'wb') as f:
            f.write(photo)
except sqlite3.Error as error:
    print("Error: ", error)
else:
    print('\nSuccess. Images have been exported to "{}" forder'.format(saved_folder))

cur.close()
con.close()