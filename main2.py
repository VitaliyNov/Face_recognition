import cv2
from scipy.spatial import distance
import Object
import os
from datetime import datetime
from vision.ssd.config.fd_config import define_img_size
import sqlite3
import time


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


net_type = 'slim'   # The network architecture ,optional: RFB (higher precision) or slim (faster)
input_size = 1280   # define network input size,default optional value 128/160/320/480/640/1280
                   # less value detect faster, but reduce accuracy.
threshold = 0.5
candidate_size = 100   # nms candidate size (max detections size)
test_device = 'cuda:0'  # cuda:0 or cpu

define_img_size(input_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

label_path = "./models/voc-model-labels.txt"
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

cap = cv2.VideoCapture(r"../samples/video9.mp4")
ret, frame = cap.read()
hcap, wcap = frame.shape[:2]
hcap, wcap = int(hcap), int(wcap)

# fourcc = cv2.VideoWriter_fourcc('X','V','I','D')        # write result in output
# out = cv2.VideoWriter("output.avi", fourcc, 20.0, (wcap,hcap))

# for i in range(1200):    # for trimming video
#    r, f = cap.read()

up_limit = 50
down_limit = hcap-50
left_limit = 50
right_limit = wcap-50

if net_type == 'slim':
    #model_path = "models/pretrained/version-slim-320.pth"
    model_path = "models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
elif net_type == 'RFB':
    # model_path = "models/pretrained/version-RFB-320.pth"  # 320-version recommend for medium-distance, large face, and small number of faces
    model_path = "models/pretrained/version-RFB-640.pth"    # 640-version recommend for medium or long distance, medium or small face and large face number
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
net.load(model_path)

objects = []
max_o_age = 10     # number of frames that Object can exist without owner(point)
obj_id = 1           # first Id
add_id = 1         # additional var for ID-overflow prevention
r = 15             # additional radius for expand boxes
r2 = 15            # additional radius for expand saved photo
# saved_folder = "saved_faces"    # folder for saving detected faces
camera_name = 'cam1'    # current camera name

prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()

    for i in objects: # check if object have owner. If no, over <max_o_age> frames it will be deleted
        i.age_one()

    frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(frame_copy, candidate_size / 2, threshold)

    freezedIds = []          # for freeze id that already have owner
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        x, y, w, h = int(box[0] - r), int(box[1] - r), int(box[2]-box[0] + 2*r), int(box[3]-box[1] + 2*r)
        cx = int(x + w / 2)  # centroids
        cy = int(y + h / 2)

        if cy in range(up_limit, down_limit) and cx in range(left_limit, right_limit):
            new = True  # for define new object

            # for gather all objects in area aroud predicted point for select nearest for its.
            CandidatesForPoint = []
            for i in objects:
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
                new = False         # object with this id already exist
                i.updateCoords(cx, cy)

                # saving faces to saved_folder
                # if y-r2 > 0 and x-r2 > 0 and y+h+r2 < hcap and x+w+r2 < wcap:
                #     if i.saved == False:
                #         crop_person = frame[y-r2: y+h+r2, x-r2: x+w+r2]
                #         crop_person = cv2.resize(crop_person, (240, 320))
                #         now = datetime.now()
                #         cur_date = now.strftime("%Y.%m.%d_%H-%M-%S")
                #         cv2.imwrite(os.path.join(saved_folder, 'person_{}-{}_[{}].jpg'.format(add_id, i.getId(), cur_date)), crop_person)
                #         i.setSaved()

                # saving faces to sqlite database 'SavesFaces.db'
                if y - r2 > 0 and x - r2 > 0 and y + h + r2 < hcap and x + w + r2 < wcap:
                    if i.saved == False:
                        crop_person = frame[y - r2: y + h + r2, x - r2: x + w + r2]
                        crop_person = cv2.resize(crop_person, (160, 240))  # 160x240 px - size of saved face frame
                        _, crop_person = cv2.imencode(".jpg", crop_person)
                        crop_person = crop_person.tobytes()  # convert img to bytes for saving in database
                        now = datetime.now()
                        cur_date = now.strftime("%Y.%m.%d %H:%M:%S")  # detection datetime
                        person_id = 'person_{}-{}'.format(add_id, i.getId())
                        data_tuple = (person_id, camera_name, cur_date, crop_person)
                        try:
                            cur.execute(sqlite_insert, data_tuple)
                        except sqlite3.Error as error:
                            print("Error: ", error)
                        else:
                            con.commit()
                        i.setSaved()  # for saving face at once

                # for delete object
                if (i.getX() < left_limit or i.getX() > right_limit or
                i.getY() > down_limit or i.getY() < up_limit):
                    i.setDone()
            # set a new object
            if new == True:
                obj = Object.MyObject(obj_id, cx, cy, max_o_age)
                objects.append(obj)
                obj_id += 1
            # for prevent ID-overflow
            if obj_id == 10000:
                obj_id = 1
                add_id += 1

    # visualize detected faces
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

    # delete all objects that have object.done == True
    deletes = []
    for o in objects:
        if o.timedOut():
            deletes.append(o)
    if len(deletes) != 0:
        for d in deletes:
            index = objects.index(d)
            objects.pop(index)
            del d

    # for i in objects:
    #     cv2.putText(frame, str(i.getId()), (i.getX(), i.getY()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(int(fps))
    cv2.putText(frame, 'fps: ' + fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

    # out.write(frame)
    frame = cv2.resize(frame, (1240, 720))
    cv2.imshow('out', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
cur.close()
con.close()
print('Finish session.')