import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from PeopleTracking import Ui_Dialog


class PeopleTrack(QtWidgets.QDialog):
    def __init__(self):
        super(PeopleTrack, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.buttonBox.accepted.connect(self.getValues)
        self.ui.buttonBox.rejected.connect(self.cancel_program)

    def getValues(self):
        self.input_image_size = self.ui.comboBox_1.currentText()
        self.num_streams = self.ui.comboBox_2.currentText()
        self.threshold = self.ui.comboBox_3.currentText()
        self.camera1_name = self.ui.lineEdit_1.text()
        self.camera2_name = self.ui.lineEdit_4.text()
        self.camera3_name = self.ui.lineEdit_2.text()
        self.camera4_name = self.ui.lineEdit_6.text()
        self.camera1_url = self.ui.lineEdit_5.text()
        self.camera2_url = self.ui.lineEdit_3.text()
        self.camera3_url = self.ui.lineEdit_7.text()
        self.camera4_url = self.ui.lineEdit_8.text()

    def cancel_program(self):
        sys.exit()


app = QtWidgets.QApplication([])
application = PeopleTrack()
application.show()
app.exec()


if application.num_streams == '1':
    if application.camera1_url == '':
        raise ValueError('Empty first camera URL value! Please set URL or reduce "Number of streams" value')
if application.num_streams == '2':
    if application.camera1_url == '':
        raise ValueError('Empty first camera URL value! Please set URL or reduce "Number of streams" value')
    if application.camera2_url == '':
        raise ValueError('Empty second camera URL value! Please set URL or reduce "Number of streams" value')
if application.num_streams == '3':
    if application.camera1_url == '':
        raise ValueError('Empty first camera URL value! Please set URL or reduce "Number of streams" value')
    if application.camera2_url == '':
        raise ValueError('Empty second camera URL value! Please set URL or reduce "Number of streams" value')
    if application.camera3_url == '':
        raise ValueError('Empty third camera URL value! Please set URL or reduce "Number of streams" value')
if application.num_streams == '4':
    if application.camera1_url == '':
        raise ValueError('Empty first camera URL value! Please set URL or reduce "Number of streams" value')
    if application.camera2_url == '':
        raise ValueError('Empty second camera URL value! Please set URL or reduce "Number of streams" value')
    if application.camera3_url == '':
        raise ValueError('Empty third camera URL value! Please set URL or reduce "Number of streams" value')
    if application.camera4_url == '':
        raise ValueError('Empty fourth camera URL value! Please set URL or reduce "Number of streams" value')


import cv2
from scipy.spatial import distance
import Object
import os
from datetime import datetime
from vision.ssd.config.fd_config import define_img_size
import sqlite3


# creating or opening sqlite database for saving detected faces
con = sqlite3.connect('SavedFaces.db')
cur = con.cursor()
sqlite_create_db = """\
CREATE TABLE IF NOT EXISTS SavedFaces (
person TEXT,
camera TEXT,
datetime TEXT,
photo BLOB
);
"""

vacuum = """VACUUM;"""    # for clean useless memory in database

try:
    cur.execute(sqlite_create_db)
    cur.execute(vacuum)
except sqlite3.Error as err:
    print('Error:', err)
else:
    print('sqlite database initialization')

sqlite_insert = """ INSERT INTO SavedFaces
                    (person, camera, datetime, photo) VALUES (?, ?, ?, ?)"""


net_type = 'RFB'   # The network architecture ,optional: RFB (higher precision) or slim (faster)
input_size = int(application.input_image_size)   # define network input size,default optional value 128/160/320/480/640/1280
                   # less value detect faster, but reduce accuracy.
threshold = float(application.threshold)
candidate_size = 80   # nms candidate size (max detections size)
test_device = 'cuda:0'  # cuda:0 or cpu

define_img_size(input_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

label_path = "./models/voc-model-labels.txt"
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)


if net_type == 'slim':
    model_path = "models/pretrained/version-slim-320.pth"
    # model_path = "models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
elif net_type == 'RFB':
    # model_path = "models/pretrained/version-RFB-320.pth"  # 320-version recommend for medium-distance, large face, and small number of faces
    model_path = "models/pretrained/version-RFB-640.pth"    # 640-version recommend for medium or long distance, medium or small face and large face number
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
net.load(model_path)


if application.num_streams == '1':

    cap1 = cv2.VideoCapture(application.camera1_url)
    ret, frame1 = cap1.read()
    hcap1, wcap1 = frame1.shape[:2]
    hcap1, wcap1 = int(hcap1), int(wcap1)

    objects1 = []
    max_o_age = 10  # number of frames that Object can exist without owner(point)
    obj_id1 = 1  # first Id
    add_id1 = 1  # additional var for ID-overflow prevention
    r = 15  # additional radius for expand boxes
    r2 = 15  # additional radius for expand saved photo
    # saved_folder = "saved_faces"    # folder for saving detected faces
    camera1 = application.camera1_name  # current camera name

    print("\n\tClick 'q' for stop process!!!")

    while True:
        ret1, frame1 = cap1.read()

        for i in objects1:  # check if object have owner. If no, over <max_o_age> frames it will be deleted
            i.age_one()

        frame_copy1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        boxes1, labels, probs1 = predictor.predict(frame_copy1, candidate_size / 2, threshold)

        freezedIds = []  # for freeze id that already have owner
        for i in range(boxes1.size(0)):
            box = boxes1[i, :]
            x, y, w, h = int(box[0] - r), int(box[1] - r), int(box[2] - box[0] + 2 * r), int(box[3] - box[1] + 2 * r)
            cx = int(x + w / 2)  # centroids
            cy = int(y + h / 2)

            if cy in range(50, hcap1-50) and cx in range(50, wcap1-50):
                new = True  # for define new object

                # for gather all objects in area aroud predicted point for select nearest for its.
                CandidatesForPoint = []
                for i in objects1:
                    if abs(cx - i.getX()) <= w and abs(cy - i.getY()) <= h:
                        if i not in freezedIds:
                            CandidatesForPoint.append(i)
                if len(CandidatesForPoint) > 0:
                    min = 1000000
                    for c in CandidatesForPoint:
                        dist = distance.euclidean((cx, cy), (c.getX(), c.getY()))
                        if dist < min:
                            min = dist
                            i = c  # i - suitable object for current point

                    freezedIds.append(i)  # for freeze id that already have owner
                    new = False  # object with this id already exist
                    i.updateCoords(cx, cy)

                    # saving faces to sqlite database 'SavesFaces.db'
                    if y - r2 > 0 and x - r2 > 0 and y + h + r2 < hcap1 and x + w + r2 < wcap1:
                        if i.saved == False:
                            crop_person = frame1[y - r2: y + h + r2, x - r2: x + w + r2]
                            crop_person = cv2.resize(crop_person, (160, 240))  # 160x240 px - size of saved face frame
                            _, crop_person = cv2.imencode(".jpg", crop_person)
                            crop_person = crop_person.tobytes()  # convert img to bytes for saving in database
                            now = datetime.now()
                            cur_date = now.strftime("%Y.%m.%d %H:%M:%S")  # detection datetime
                            person_id = 'person_{}-{}'.format(add_id1, i.getId())
                            data_tuple = (person_id, camera1, cur_date, crop_person)
                            try:
                                cur.execute(sqlite_insert, data_tuple)
                            except sqlite3.Error as error:
                                print("Error: ", error)
                            else:
                                con.commit()
                            i.setSaved()  # for saving face at once

                    # for delete object
                    if (i.getX() < 50 or i.getX() > (wcap1-50) or
                            i.getY() > (hcap1-50) or i.getY() < 50):
                        i.setDone()
                # set a new object
                if new == True:
                    obj = Object.MyObject(obj_id1, cx, cy, max_o_age)
                    objects1.append(obj)
                    obj_id1 += 1
                # for prevent ID-overflow
                if obj_id1 == 10000:
                    obj_id1 = 1
                    add_id1 += 1

        # visualize detected faces
        for i in range(boxes1.size(0)):
            box = boxes1[i, :]
            cv2.rectangle(frame1, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # delete all objects that have object.done == True
        deletes = []
        for o in objects1:
            if o.timedOut():
                deletes.append(o)
        if len(deletes) != 0:
            for d in deletes:
                index = objects1.index(d)
                objects1.pop(index)
                del d

        frame1 = cv2.resize(frame1, (640, 480))
        cv2.imshow('output1', frame1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cv2.destroyAllWindows()
    cur.close()
    con.close()
    print('Finish session.')


if application.num_streams == '2':

    cap1 = cv2.VideoCapture(application.camera1_url)
    cap2 = cv2.VideoCapture(application.camera2_url)
    ret, frame1 = cap1.read()
    hcap1, wcap1 = frame1.shape[:2]
    hcap1, wcap1 = int(hcap1), int(wcap1)
    ret, frame2 = cap2.read()
    hcap2, wcap2 = frame2.shape[:2]
    hcap2, wcap2 = int(hcap2), int(wcap2)


    objects1 = []
    objects2 = []
    max_o_age = 10  # number of frames that Object can exist without owner(point)
    obj_id1 = 1  # first Id
    obj_id2 = 1
    add_id1 = 1  # additional var for ID-overflow prevention
    add_id2 = 1
    r = 15  # additional radius for expand boxes
    r2 = 15  # additional radius for expand saved photo
    # saved_folder = "saved_faces"    # folder for saving detected faces
    camera1 = application.camera1_name  # current camera name
    camera2 = application.camera2_name

    print("\n\tClick 'q' for stop process!!!")

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        for i in objects1:  # check if object have owner. If no, over <max_o_age> frames it will be deleted
            i.age_one()

        for i in objects2:  # check if object have owner. If no, over <max_o_age> frames it will be deleted
            i.age_one()

        frame_copy1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        boxes1, labels, probs1 = predictor.predict(frame_copy1, candidate_size / 2, threshold)

        frame_copy2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        boxes2, labels2, probs2 = predictor.predict(frame_copy2, candidate_size / 2, threshold)

        # cap1
        freezedIds = []  # for freeze id that already have owner
        for i in range(boxes1.size(0)):
            box = boxes1[i, :]
            x, y, w, h = int(box[0] - r), int(box[1] - r), int(box[2] - box[0] + 2 * r), int(box[3] - box[1] + 2 * r)
            cx = int(x + w / 2)  # centroids
            cy = int(y + h / 2)

            if cy in range(50, hcap1-50) and cx in range(50, wcap1-50):
                new = True  # for define new object

                # for gather all objects in area aroud predicted point for select nearest for its.
                CandidatesForPoint = []
                for i in objects1:
                    if abs(cx - i.getX()) <= w and abs(cy - i.getY()) <= h:
                        if i not in freezedIds:
                            CandidatesForPoint.append(i)
                if len(CandidatesForPoint) > 0:
                    min = 1000000
                    for c in CandidatesForPoint:
                        dist = distance.euclidean((cx, cy), (c.getX(), c.getY()))
                        if dist < min:
                            min = dist
                            i = c  # i - suitable object for current point

                    freezedIds.append(i)  # for freeze id that already have owner
                    new = False  # object with this id already exist
                    i.updateCoords(cx, cy)

                    # saving faces to sqlite database 'SavesFaces.db'
                    if y - r2 > 0 and x - r2 > 0 and y + h + r2 < hcap1 and x + w + r2 < wcap1:
                        if i.saved == False:
                            crop_person = frame1[y - r2: y + h + r2, x - r2: x + w + r2]
                            crop_person = cv2.resize(crop_person, (160, 240))  # 160x240 px - size of saved face frame
                            _, crop_person = cv2.imencode(".jpg", crop_person)
                            crop_person = crop_person.tobytes()  # convert img to bytes for saving in database
                            now = datetime.now()
                            cur_date = now.strftime("%Y.%m.%d %H:%M:%S")  # detection datetime
                            person_id = 'person_{}-{}'.format(add_id1, i.getId())
                            data_tuple = (person_id, camera1, cur_date, crop_person)
                            try:
                                cur.execute(sqlite_insert, data_tuple)
                            except sqlite3.Error as error:
                                print("Error: ", error)
                            else:
                                con.commit()
                            i.setSaved()  # for saving face at once

                    # for delete object
                    if (i.getX() < 50 or i.getX() > (wcap1-50) or
                            i.getY() > (hcap1-50) or i.getY() < 50):
                        i.setDone()
                # set a new object
                if new == True:
                    obj = Object.MyObject(obj_id1, cx, cy, max_o_age)
                    objects1.append(obj)
                    obj_id1 += 1
                # for prevent ID-overflow
                if obj_id1 == 10000:
                    obj_id1 = 1
                    add_id1 += 1

        # visualize detected faces
        for i in range(boxes1.size(0)):
            box = boxes1[i, :]
            cv2.rectangle(frame1, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # delete all objects that have object.done == True
        deletes = []
        for o in objects1:
            if o.timedOut():
                deletes.append(o)
        if len(deletes) != 0:
            for d in deletes:
                index = objects1.index(d)
                objects1.pop(index)
                del d

        # cap2
        freezedIds = []  # for freeze id that already have owner
        for i in range(boxes2.size(0)):
            box = boxes2[i, :]
            x, y, w, h = int(box[0] - r), int(box[1] - r), int(box[2] - box[0] + 2 * r), int(box[3] - box[1] + 2 * r)
            cx = int(x + w / 2)  # centroids
            cy = int(y + h / 2)

            if cy in range(50, hcap2-50) and cx in range(50, wcap2-50):
                new = True  # for define new object

                # for gather all objects in area aroud predicted point for select nearest for its.
                CandidatesForPoint = []
                for i in objects2:
                    if abs(cx - i.getX()) <= w and abs(cy - i.getY()) <= h:
                        if i not in freezedIds:
                            CandidatesForPoint.append(i)
                if len(CandidatesForPoint) > 0:
                    min = 1000000
                    for c in CandidatesForPoint:
                        dist = distance.euclidean((cx, cy), (c.getX(), c.getY()))
                        if dist < min:
                            min = dist
                            i = c  # i - suitable object for current point

                    freezedIds.append(i)  # for freeze id that already have owner
                    new = False  # object with this id already exist
                    i.updateCoords(cx, cy)

                    # saving faces to sqlite database 'SavesFaces.db'
                    if y - r2 > 0 and x - r2 > 0 and y + h + r2 < hcap2 and x + w + r2 < wcap2:
                        if i.saved == False:
                            crop_person = frame2[y - r2: y + h + r2, x - r2: x + w + r2]
                            crop_person = cv2.resize(crop_person, (160, 240))  # 160x240 px - size of saved face frame
                            _, crop_person = cv2.imencode(".jpg", crop_person)
                            crop_person = crop_person.tobytes()  # convert img to bytes for saving in database
                            now = datetime.now()
                            cur_date = now.strftime("%Y.%m.%d %H:%M:%S")  # detection datetime
                            person_id = 'person_{}-{}'.format(add_id2, i.getId())
                            data_tuple = (person_id, camera2, cur_date, crop_person)
                            try:
                                cur.execute(sqlite_insert, data_tuple)
                            except sqlite3.Error as error:
                                print("Error: ", error)
                            else:
                                con.commit()
                            i.setSaved()  # for saving face at once

                    # for delete object
                    if (i.getX() < 50 or i.getX() > (wcap2-50) or
                            i.getY() > (hcap2-50) or i.getY() < 50):
                        i.setDone()
                # set a new object
                if new == True:
                    obj = Object.MyObject(obj_id2, cx, cy, max_o_age)
                    objects2.append(obj)
                    obj_id2 += 1
                # for prevent ID-overflow
                if obj_id2 == 10000:
                    obj_id2 = 1
                    add_id2 += 1

        # visualize detected faces
        for i in range(boxes2.size(0)):
            box = boxes2[i, :]
            cv2.rectangle(frame2, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # delete all objects that have object.done == True
        deletes = []
        for o in objects2:
            if o.timedOut():
                deletes.append(o)
        if len(deletes) != 0:
            for d in deletes:
                index = objects2.index(d)
                objects2.pop(index)
                del d

        frame1 = cv2.resize(frame1, (640, 480))
        frame2 = cv2.resize(frame2, (640, 480))
        cv2.imshow('output1', frame1)
        cv2.imshow('output2', frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    cur.close()
    con.close()
    print('Finish session.')

if application.num_streams == '3':

    cap1 = cv2.VideoCapture(application.camera1_url)
    cap2 = cv2.VideoCapture(application.camera2_url)
    cap3 = cv2.VideoCapture(application.camera3_url)
    ret, frame1 = cap1.read()
    hcap1, wcap1 = frame1.shape[:2]
    hcap1, wcap1 = int(hcap1), int(wcap1)
    ret, frame2 = cap2.read()
    hcap2, wcap2 = frame2.shape[:2]
    hcap2, wcap2 = int(hcap2), int(wcap2)
    ret, frame3 = cap3.read()
    hcap3, wcap3 = frame3.shape[:2]
    hcap3, wcap3 = int(hcap3), int(wcap3)

    objects1 = []
    objects2 = []
    objects3 = []
    max_o_age = 10  # number of frames that Object can exist without owner(point)
    obj_id1 = 1  # first Id
    obj_id2 = 1
    obj_id3 = 1
    add_id1 = 1  # additional var for ID-overflow prevention
    add_id2 = 1
    add_id3 = 1
    r = 15  # additional radius for expand boxes
    r2 = 15  # additional radius for expand saved photo
    # saved_folder = "saved_faces"    # folder for saving detected faces
    camera1 = application.camera1_name  # current camera name
    camera2 = application.camera2_name
    camera3 = application.camera3_name

    print("\n\tClick 'q' for stop process!!!")

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()

        for i in objects1:  # check if object have owner. If no, over <max_o_age> frames it will be deleted
            i.age_one()

        for i in objects2:  # check if object have owner. If no, over <max_o_age> frames it will be deleted
            i.age_one()

        for i in objects3:  # check if object have owner. If no, over <max_o_age> frames it will be deleted
            i.age_one()

        frame_copy1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        boxes1, labels, probs1 = predictor.predict(frame_copy1, candidate_size / 2, threshold)

        frame_copy2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        boxes2, labels2, probs2 = predictor.predict(frame_copy2, candidate_size / 2, threshold)

        frame_copy3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
        boxes3, labels3, probs3 = predictor.predict(frame_copy3, candidate_size / 2, threshold)

        # cap1
        freezedIds = []  # for freeze id that already have owner
        for i in range(boxes1.size(0)):
            box = boxes1[i, :]
            x, y, w, h = int(box[0] - r), int(box[1] - r), int(box[2] - box[0] + 2 * r), int(box[3] - box[1] + 2 * r)
            cx = int(x + w / 2)  # centroids
            cy = int(y + h / 2)

            if cy in range(50, hcap1-50) and cx in range(50, wcap1-50):
                new = True  # for define new object

                # for gather all objects in area aroud predicted point for select nearest for its.
                CandidatesForPoint = []
                for i in objects1:
                    if abs(cx - i.getX()) <= w and abs(cy - i.getY()) <= h:
                        if i not in freezedIds:
                            CandidatesForPoint.append(i)
                if len(CandidatesForPoint) > 0:
                    min = 1000000
                    for c in CandidatesForPoint:
                        dist = distance.euclidean((cx, cy), (c.getX(), c.getY()))
                        if dist < min:
                            min = dist
                            i = c  # i - suitable object for current point

                    freezedIds.append(i)  # for freeze id that already have owner
                    new = False  # object with this id already exist
                    i.updateCoords(cx, cy)

                    # saving faces to sqlite database 'SavesFaces.db'
                    if y - r2 > 0 and x - r2 > 0 and y + h + r2 < hcap1 and x + w + r2 < wcap1:
                        if i.saved == False:
                            crop_person = frame1[y - r2: y + h + r2, x - r2: x + w + r2]
                            crop_person = cv2.resize(crop_person, (160, 240))  # 160x240 px - size of saved face frame
                            _, crop_person = cv2.imencode(".jpg", crop_person)
                            crop_person = crop_person.tobytes()  # convert img to bytes for saving in database
                            now = datetime.now()
                            cur_date = now.strftime("%Y.%m.%d %H:%M:%S")  # detection datetime
                            person_id = 'person_{}-{}'.format(add_id1, i.getId())
                            data_tuple = (person_id, camera1, cur_date, crop_person)
                            try:
                                cur.execute(sqlite_insert, data_tuple)
                            except sqlite3.Error as error:
                                print("Error: ", error)
                            else:
                                con.commit()
                            i.setSaved()  # for saving face at once

                    # for delete object
                    if (i.getX() < 50 or i.getX() > (wcap1-50) or
                            i.getY() > (hcap1-50) or i.getY() < 50):
                        i.setDone()
                # set a new object
                if new == True:
                    obj = Object.MyObject(obj_id1, cx, cy, max_o_age)
                    objects1.append(obj)
                    obj_id1 += 1
                # for prevent ID-overflow
                if obj_id1 == 10000:
                    obj_id1 = 1
                    add_id1 += 1

        # visualize detected faces
        for i in range(boxes1.size(0)):
            box = boxes1[i, :]
            cv2.rectangle(frame1, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # delete all objects that have object.done == True
        deletes = []
        for o in objects1:
            if o.timedOut():
                deletes.append(o)
        if len(deletes) != 0:
            for d in deletes:
                index = objects1.index(d)
                objects1.pop(index)
                del d

        # cap2
        freezedIds = []  # for freeze id that already have owner
        for i in range(boxes2.size(0)):
            box = boxes2[i, :]
            x, y, w, h = int(box[0] - r), int(box[1] - r), int(box[2] - box[0] + 2 * r), int(box[3] - box[1] + 2 * r)
            cx = int(x + w / 2)  # centroids
            cy = int(y + h / 2)

            if cy in range(50, hcap2-50) and cx in range(50, wcap2-50):
                new = True  # for define new object

                # for gather all objects in area aroud predicted point for select nearest for its.
                CandidatesForPoint = []
                for i in objects2:
                    if abs(cx - i.getX()) <= w and abs(cy - i.getY()) <= h:
                        if i not in freezedIds:
                            CandidatesForPoint.append(i)
                if len(CandidatesForPoint) > 0:
                    min = 1000000
                    for c in CandidatesForPoint:
                        dist = distance.euclidean((cx, cy), (c.getX(), c.getY()))
                        if dist < min:
                            min = dist
                            i = c  # i - suitable object for current point

                    freezedIds.append(i)  # for freeze id that already have owner
                    new = False  # object with this id already exist
                    i.updateCoords(cx, cy)

                    # saving faces to sqlite database 'SavesFaces.db'
                    if y - r2 > 0 and x - r2 > 0 and y + h + r2 < hcap2 and x + w + r2 < wcap2:
                        if i.saved == False:
                            crop_person = frame2[y - r2: y + h + r2, x - r2: x + w + r2]
                            crop_person = cv2.resize(crop_person, (160, 240))  # 160x240 px - size of saved face frame
                            _, crop_person = cv2.imencode(".jpg", crop_person)
                            crop_person = crop_person.tobytes()  # convert img to bytes for saving in database
                            now = datetime.now()
                            cur_date = now.strftime("%Y.%m.%d %H:%M:%S")  # detection datetime
                            person_id = 'person_{}-{}'.format(add_id2, i.getId())
                            data_tuple = (person_id, camera2, cur_date, crop_person)
                            try:
                                cur.execute(sqlite_insert, data_tuple)
                            except sqlite3.Error as error:
                                print("Error: ", error)
                            else:
                                con.commit()
                            i.setSaved()  # for saving face at once

                    # for delete object
                    if (i.getX() < 50 or i.getX() > (wcap2-50) or
                            i.getY() > (hcap2-50) or i.getY() < 50):
                        i.setDone()
                # set a new object
                if new == True:
                    obj = Object.MyObject(obj_id2, cx, cy, max_o_age)
                    objects2.append(obj)
                    obj_id2 += 1
                # for prevent ID-overflow
                if obj_id2 == 10000:
                    obj_id2 = 1
                    add_id2 += 1

        # visualize detected faces
        for i in range(boxes2.size(0)):
            box = boxes2[i, :]
            cv2.rectangle(frame2, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # delete all objects that have object.done == True
        deletes = []
        for o in objects2:
            if o.timedOut():
                deletes.append(o)
        if len(deletes) != 0:
            for d in deletes:
                index = objects2.index(d)
                objects2.pop(index)
                del d

        # cap3
        freezedIds = []  # for freeze id that already have owner
        for i in range(boxes3.size(0)):
            box = boxes3[i, :]
            x, y, w, h = int(box[0] - r), int(box[1] - r), int(box[2] - box[0] + 2 * r), int(box[3] - box[1] + 2 * r)
            cx = int(x + w / 2)  # centroids
            cy = int(y + h / 2)

            if cy in range(50, (hcap3-50)) and cx in range(50, (wcap3-50)):
                new = True  # for define new object

                # for gather all objects in area aroud predicted point for select nearest for its.
                CandidatesForPoint = []
                for i in objects3:
                    if abs(cx - i.getX()) <= w and abs(cy - i.getY()) <= h:
                        if i not in freezedIds:
                            CandidatesForPoint.append(i)
                if len(CandidatesForPoint) > 0:
                    min = 1000000
                    for c in CandidatesForPoint:
                        dist = distance.euclidean((cx, cy), (c.getX(), c.getY()))
                        if dist < min:
                            min = dist
                            i = c  # i - suitable object for current point

                    freezedIds.append(i)  # for freeze id that already have owner
                    new = False  # object with this id already exist
                    i.updateCoords(cx, cy)

                    # saving faces to sqlite database 'SavesFaces.db'
                    if y - r2 > 0 and x - r2 > 0 and y + h + r2 < hcap3 and x + w + r2 < wcap3:
                        if i.saved == False:
                            crop_person = frame3[y - r2: y + h + r2, x - r2: x + w + r2]
                            crop_person = cv2.resize(crop_person, (160, 240))  # 160x240 px - size of saved face frame
                            _, crop_person = cv2.imencode(".jpg", crop_person)
                            crop_person = crop_person.tobytes()  # convert img to bytes for saving in database
                            now = datetime.now()
                            cur_date = now.strftime("%Y.%m.%d %H:%M:%S")  # detection datetime
                            person_id = 'person_{}-{}'.format(add_id3, i.getId())
                            data_tuple = (person_id, camera3, cur_date, crop_person)
                            try:
                                cur.execute(sqlite_insert, data_tuple)
                            except sqlite3.Error as error:
                                print("Error: ", error)
                            else:
                                con.commit()
                            i.setSaved()  # for saving face at once

                    # for delete object
                    if (i.getX() < 50 or i.getX() > (wcap3-50) or
                            i.getY() > (hcap3-50) or i.getY() < 50):
                        i.setDone()
                # set a new object
                if new == True:
                    obj = Object.MyObject(obj_id3, cx, cy, max_o_age)
                    objects3.append(obj)
                    obj_id3 += 1
                # for prevent ID-overflow
                if obj_id3 == 10000:
                    obj_id3 = 1
                    add_id3 += 1

        # visualize detected faces
        for i in range(boxes3.size(0)):
            box = boxes3[i, :]
            cv2.rectangle(frame3, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # delete all objects that have object.done == True
        deletes = []
        for o in objects3:
            if o.timedOut():
                deletes.append(o)
        if len(deletes) != 0:
            for d in deletes:
                index = objects3.index(d)
                objects3.pop(index)
                del d

        frame1 = cv2.resize(frame1, (640, 480))
        frame2 = cv2.resize(frame2, (640, 480))
        frame3 = cv2.resize(frame3, (640, 480))
        cv2.imshow('output1', frame1)
        cv2.imshow('output2', frame2)
        cv2.imshow('output3', frame3)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cap3.release()
    cv2.destroyAllWindows()
    cur.close()
    con.close()
    print('Finish session.')

if application.num_streams == '4':

    cap1 = cv2.VideoCapture(application.camera1_url)
    cap2 = cv2.VideoCapture(application.camera2_url)
    cap3 = cv2.VideoCapture(application.camera3_url)
    cap4 = cv2.VideoCapture(application.camera4_url)
    ret, frame1 = cap1.read()
    hcap1, wcap1 = frame1.shape[:2]
    hcap1, wcap1 = int(hcap1), int(wcap1)
    ret, frame2 = cap2.read()
    hcap2, wcap2 = frame2.shape[:2]
    hcap2, wcap2 = int(hcap2), int(wcap2)
    ret, frame3 = cap3.read()
    hcap3, wcap3 = frame3.shape[:2]
    hcap3, wcap3 = int(hcap3), int(wcap3)
    ret, frame4 = cap4.read()
    hcap4, wcap4 = frame4.shape[:2]
    hcap4, wcap4 = int(hcap4), int(wcap4)

    objects1 = []
    objects2 = []
    objects3 = []
    objects4 = []
    max_o_age = 10  # number of frames that Object can exist without owner(point)
    obj_id1 = 1  # first Id
    obj_id2 = 1
    obj_id3 = 1
    obj_id4 = 1
    add_id1 = 1  # additional var for ID-overflow prevention
    add_id2 = 1
    add_id3 = 1
    add_id4 = 1
    r = 15  # additional radius for expand boxes
    r2 = 15  # additional radius for expand saved photo
    # saved_folder = "saved_faces"    # folder for saving detected faces
    camera1 = application.camera1_name  # current camera name
    camera2 = application.camera2_name
    camera3 = application.camera3_name
    camera4 = application.camera4_name

    print("\n\tClick 'q' for stop process!!!")

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        ret3, frame4 = cap4.read()

        for i in objects1:  # check if object have owner. If no, over <max_o_age> frames it will be deleted
            i.age_one()

        for i in objects2:  # check if object have owner. If no, over <max_o_age> frames it will be deleted
            i.age_one()

        for i in objects3:  # check if object have owner. If no, over <max_o_age> frames it will be deleted
            i.age_one()

        for i in objects4:  # check if object have owner. If no, over <max_o_age> frames it will be deleted
            i.age_one()

        frame_copy1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        boxes1, labels, probs1 = predictor.predict(frame_copy1, candidate_size / 2, threshold)

        frame_copy2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        boxes2, labels2, probs2 = predictor.predict(frame_copy2, candidate_size / 2, threshold)

        frame_copy3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
        boxes3, labels3, probs3 = predictor.predict(frame_copy3, candidate_size / 2, threshold)

        frame_copy4 = cv2.cvtColor(frame4, cv2.COLOR_BGR2RGB)
        boxes4, labels4, probs4 = predictor.predict(frame_copy4, candidate_size / 2, threshold)

        # cap1
        freezedIds = []  # for freeze id that already have owner
        for i in range(boxes1.size(0)):
            box = boxes1[i, :]
            x, y, w, h = int(box[0] - r), int(box[1] - r), int(box[2] - box[0] + 2 * r), int(box[3] - box[1] + 2 * r)
            cx = int(x + w / 2)  # centroids
            cy = int(y + h / 2)

            if cy in range(50, (wcap1-50)) and cx in range(50, (hcap1-50)):
                new = True  # for define new object

                # for gather all objects in area aroud predicted point for select nearest for its.
                CandidatesForPoint = []
                for i in objects1:
                    if abs(cx - i.getX()) <= w and abs(cy - i.getY()) <= h:
                        if i not in freezedIds:
                            CandidatesForPoint.append(i)
                if len(CandidatesForPoint) > 0:
                    min = 1000000
                    for c in CandidatesForPoint:
                        dist = distance.euclidean((cx, cy), (c.getX(), c.getY()))
                        if dist < min:
                            min = dist
                            i = c  # i - suitable object for current point

                    freezedIds.append(i)  # for freeze id that already have owner
                    new = False  # object with this id already exist
                    i.updateCoords(cx, cy)

                    # saving faces to sqlite database 'SavesFaces.db'
                    if y - r2 > 0 and x - r2 > 0 and y + h + r2 < hcap1 and x + w + r2 < wcap1:
                        if i.saved == False:
                            crop_person = frame1[y - r2: y + h + r2, x - r2: x + w + r2]
                            crop_person = cv2.resize(crop_person, (160, 240))  # 160x240 px - size of saved face frame
                            _, crop_person = cv2.imencode(".jpg", crop_person)
                            crop_person = crop_person.tobytes()  # convert img to bytes for saving in database
                            now = datetime.now()
                            cur_date = now.strftime("%Y.%m.%d %H:%M:%S")  # detection datetime
                            person_id = 'person_{}-{}'.format(add_id1, i.getId())
                            data_tuple = (person_id, camera1, cur_date, crop_person)
                            try:
                                cur.execute(sqlite_insert, data_tuple)
                            except sqlite3.Error as error:
                                print("Error: ", error)
                            else:
                                con.commit()
                            i.setSaved()  # for saving face at once

                    # for delete object
                    if (i.getX() < 50 or i.getX() > (wcap1-50) or
                            i.getY() > (hcap1-50) or i.getY() < 50):
                        i.setDone()
                # set a new object
                if new == True:
                    obj = Object.MyObject(obj_id1, cx, cy, max_o_age)
                    objects1.append(obj)
                    obj_id1 += 1
                # for prevent ID-overflow
                if obj_id1 == 10000:
                    obj_id1 = 1
                    add_id1 += 1

        # visualize detected faces
        for i in range(boxes1.size(0)):
            box = boxes1[i, :]
            cv2.rectangle(frame1, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # delete all objects that have object.done == True
        deletes = []
        for o in objects1:
            if o.timedOut():
                deletes.append(o)
        if len(deletes) != 0:
            for d in deletes:
                index = objects1.index(d)
                objects1.pop(index)
                del d

        # cap2
        freezedIds = []  # for freeze id that already have owner
        for i in range(boxes2.size(0)):
            box = boxes2[i, :]
            x, y, w, h = int(box[0] - r), int(box[1] - r), int(box[2] - box[0] + 2 * r), int(box[3] - box[1] + 2 * r)
            cx = int(x + w / 2)  # centroids
            cy = int(y + h / 2)

            if cy in range(50, (hcap2-50)) and cx in range(50, (wcap2-50)):
                new = True  # for define new object

                # for gather all objects in area aroud predicted point for select nearest for its.
                CandidatesForPoint = []
                for i in objects2:
                    if abs(cx - i.getX()) <= w and abs(cy - i.getY()) <= h:
                        if i not in freezedIds:
                            CandidatesForPoint.append(i)
                if len(CandidatesForPoint) > 0:
                    min = 1000000
                    for c in CandidatesForPoint:
                        dist = distance.euclidean((cx, cy), (c.getX(), c.getY()))
                        if dist < min:
                            min = dist
                            i = c  # i - suitable object for current point

                    freezedIds.append(i)  # for freeze id that already have owner
                    new = False  # object with this id already exist
                    i.updateCoords(cx, cy)

                    # saving faces to sqlite database 'SavesFaces.db'
                    if y - r2 > 0 and x - r2 > 0 and y + h + r2 < hcap2 and x + w + r2 < wcap2:
                        if i.saved == False:
                            crop_person = frame2[y - r2: y + h + r2, x - r2: x + w + r2]
                            crop_person = cv2.resize(crop_person, (160, 240))  # 160x240 px - size of saved face frame
                            _, crop_person = cv2.imencode(".jpg", crop_person)
                            crop_person = crop_person.tobytes()  # convert img to bytes for saving in database
                            now = datetime.now()
                            cur_date = now.strftime("%Y.%m.%d %H:%M:%S")  # detection datetime
                            person_id = 'person_{}-{}'.format(add_id2, i.getId())
                            data_tuple = (person_id, camera2, cur_date, crop_person)
                            try:
                                cur.execute(sqlite_insert, data_tuple)
                            except sqlite3.Error as error:
                                print("Error: ", error)
                            else:
                                con.commit()
                            i.setSaved()  # for saving face at once

                    # for delete object
                    if (i.getX() < 50 or i.getX() > (wcap2-50) or
                            i.getY() > (hcap2-50) or i.getY() < 50):
                        i.setDone()
                # set a new object
                if new == True:
                    obj = Object.MyObject(obj_id2, cx, cy, max_o_age)
                    objects2.append(obj)
                    obj_id2 += 1
                # for prevent ID-overflow
                if obj_id2 == 10000:
                    obj_id2 = 1
                    add_id2 += 1

        # visualize detected faces
        for i in range(boxes2.size(0)):
            box = boxes2[i, :]
            cv2.rectangle(frame2, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # delete all objects that have object.done == True
        deletes = []
        for o in objects2:
            if o.timedOut():
                deletes.append(o)
        if len(deletes) != 0:
            for d in deletes:
                index = objects2.index(d)
                objects2.pop(index)
                del d

        # cap3
        freezedIds = []  # for freeze id that already have owner
        for i in range(boxes3.size(0)):
            box = boxes3[i, :]
            x, y, w, h = int(box[0] - r), int(box[1] - r), int(box[2] - box[0] + 2 * r), int(box[3] - box[1] + 2 * r)
            cx = int(x + w / 2)  # centroids
            cy = int(y + h / 2)

            if cy in range(50, (hcap3-50)) and cx in range(50, (wcap3-50)):
                new = True  # for define new object

                # for gather all objects in area aroud predicted point for select nearest for its.
                CandidatesForPoint = []
                for i in objects3:
                    if abs(cx - i.getX()) <= w and abs(cy - i.getY()) <= h:
                        if i not in freezedIds:
                            CandidatesForPoint.append(i)
                if len(CandidatesForPoint) > 0:
                    min = 1000000
                    for c in CandidatesForPoint:
                        dist = distance.euclidean((cx, cy), (c.getX(), c.getY()))
                        if dist < min:
                            min = dist
                            i = c  # i - suitable object for current point

                    freezedIds.append(i)  # for freeze id that already have owner
                    new = False  # object with this id already exist
                    i.updateCoords(cx, cy)

                    # saving faces to sqlite database 'SavesFaces.db'
                    if y - r2 > 0 and x - r2 > 0 and y + h + r2 < hcap3 and x + w + r2 < wcap3:
                        if i.saved == False:
                            crop_person = frame3[y - r2: y + h + r2, x - r2: x + w + r2]
                            crop_person = cv2.resize(crop_person, (160, 240))  # 160x240 px - size of saved face frame
                            _, crop_person = cv2.imencode(".jpg", crop_person)
                            crop_person = crop_person.tobytes()  # convert img to bytes for saving in database
                            now = datetime.now()
                            cur_date = now.strftime("%Y.%m.%d %H:%M:%S")  # detection datetime
                            person_id = 'person_{}-{}'.format(add_id3, i.getId())
                            data_tuple = (person_id, camera3, cur_date, crop_person)
                            try:
                                cur.execute(sqlite_insert, data_tuple)
                            except sqlite3.Error as error:
                                print("Error: ", error)
                            else:
                                con.commit()
                            i.setSaved()  # for saving face at once

                    # for delete object
                    if (i.getX() < 50 or i.getX() > (wcap3-50) or
                            i.getY() > (hcap3-50) or i.getY() < 50):
                        i.setDone()
                # set a new object
                if new == True:
                    obj = Object.MyObject(obj_id3, cx, cy, max_o_age)
                    objects3.append(obj)
                    obj_id3 += 1
                # for prevent ID-overflow
                if obj_id3 == 10000:
                    obj_id3 = 1
                    add_id3 += 1

        # visualize detected faces
        for i in range(boxes3.size(0)):
            box = boxes3[i, :]
            cv2.rectangle(frame3, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # delete all objects that have object.done == True
        deletes = []
        for o in objects3:
            if o.timedOut():
                deletes.append(o)
        if len(deletes) != 0:
            for d in deletes:
                index = objects3.index(d)
                objects3.pop(index)
                del d

        # cap4
        freezedIds = []  # for freeze id that already have owner
        for i in range(boxes4.size(0)):
            box = boxes4[i, :]
            x, y, w, h = int(box[0] - r), int(box[1] - r), int(box[2] - box[0] + 2 * r), int(
                box[3] - box[1] + 2 * r)
            cx = int(x + w / 2)  # centroids
            cy = int(y + h / 2)

            if cy in range(50, hcap4) and cx in range(50, wcap4-50):
                new = True  # for define new object

                # for gather all objects in area aroud predicted point for select nearest for its.
                CandidatesForPoint = []
                for i in objects4:
                    if abs(cx - i.getX()) <= w and abs(cy - i.getY()) <= h:
                        if i not in freezedIds:
                            CandidatesForPoint.append(i)
                if len(CandidatesForPoint) > 0:
                    min = 1000000
                    for c in CandidatesForPoint:
                        dist = distance.euclidean((cx, cy), (c.getX(), c.getY()))
                        if dist < min:
                            min = dist
                            i = c  # i - suitable object for current point

                    freezedIds.append(i)  # for freeze id that already have owner
                    new = False  # object with this id already exist
                    i.updateCoords(cx, cy)

                    # saving faces to sqlite database 'SavesFaces.db'
                    if y - r2 > 0 and x - r2 > 0 and y + h + r2 < hcap4 and x + w + r2 < wcap4:
                        if i.saved == False:
                            crop_person = frame4[y - r2: y + h + r2, x - r2: x + w + r2]
                            crop_person = cv2.resize(crop_person, (160, 240))  # 160x240 px - size of saved face frame
                            _, crop_person = cv2.imencode(".jpg", crop_person)
                            crop_person = crop_person.tobytes()  # convert img to bytes for saving in database
                            now = datetime.now()
                            cur_date = now.strftime("%Y.%m.%d %H:%M:%S")  # detection datetime
                            person_id = 'person_{}-{}'.format(add_id4, i.getId())
                            data_tuple = (person_id, camera4, cur_date, crop_person)
                            try:
                                cur.execute(sqlite_insert, data_tuple)
                            except sqlite3.Error as error:
                                print("Error: ", error)
                            else:
                                con.commit()
                            i.setSaved()  # for saving face at once

                    # for delete object
                    if (i.getX() < 50 or i.getX() > (wcap4 - 50) or
                            i.getY() > (hcap4 - 50) or i.getY() < 50):
                        i.setDone()
                # set a new object
                if new == True:
                    obj = Object.MyObject(obj_id4, cx, cy, max_o_age)
                    objects4.append(obj)
                    obj_id4 += 1
                # for prevent ID-overflow
                if obj_id4 == 10000:
                    obj_id4 = 1
                    add_id4 += 1

        # visualize detected faces
        for i in range(boxes4.size(0)):
            box = boxes4[i, :]
            cv2.rectangle(frame4, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # delete all objects that have object.done == True
        deletes = []
        for o in objects4:
            if o.timedOut():
                deletes.append(o)
        if len(deletes) != 0:
            for d in deletes:
                index = objects4.index(d)
                objects4.pop(index)
                del d

        cv2.imshow('output1', frame1)
        cv2.imshow('output2', frame2)
        cv2.imshow('output3', frame3)
        cv2.imshow('output4', frame4)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cap3.release()
    cap4.release()
    cv2.destroyAllWindows()
    cur.close()
    con.close()
    print('Finish session.')