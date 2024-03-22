import pickle

import cv2
import os

import cvzone
import face_recognition
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL' : "https://faceattendance-3feef-default-rtdb.firebaseio.com/",
    'storageBucket': 'faceattendance-3feef.appspot.com'
})
bucket = storage.bucket()







def mark_attendance():
    cap = cv2.VideoCapture(0)
    cap.set(3, 600)
    cap.set(4, 480)
    running = True
    imgBackground = cv2.imread('Resources/background.png')

    floderModePath = 'Resources/Modes'
    modePathList = os.listdir(floderModePath)
    imgModeList = []

    for path in modePathList:
        imgModeList.append(cv2.imread(os.path.join(floderModePath, path)))

    # load the encoding file
    print('Loading encode file')
    file = open('EncodeFile.p', 'rb')
    encodeListKnownWithIds = pickle.load(file)
    file.close()
    encodeListKnown, studentIds = encodeListKnownWithIds
    # print(studentIds)
    print("Encode File loaded")

    modeType = 0
    counter = 0
    id = -1
    imgStudent = []

    while running:
        success, img = cap.read()

        imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
        # Place the webcam feed at a specific location on the background image
        imgBackground[260:260 + 480, 1:1 + img.shape[1]] = img
        imgBackground[208:208 + imgModeList[modeType].shape[0], 920:920 + imgModeList[modeType].shape[1]] = imgModeList[
            modeType]

        if faceCurFrame:
            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                # print("matches", matches)
                # print("faceDistance", faceDis)

                matchIndex = np.argmin(faceDis)
                if matches[matchIndex]:
                    y1, x2, y2, x1 = faceLoc
                    # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = 275 + x1, 10 + y1, x2 - x1, y2 - y1
                    cvzone.cornerRect(imgBackground, bbox, rt=0)
                    # print("Known face detected")
                    # print(studentIds[matchIndex])
                    id = studentIds[matchIndex]
                    if counter == 0:
                        cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                        cv2.imshow("Face attendance", imgBackground)
                        cv2.waitKey(1)
                        counter = 1
                        modeType = 1
                if counter != 0:
                    if counter == 1:
                        # get the data
                        studentInfo = db.reference(f'students/{id}').get()
                        print(studentInfo)
                        blob = bucket.get_blob(f'Images/{id}.jpg')
                        array = np.frombuffer(blob.download_as_string(), np.uint8)
                        imgStudent = cv2.imdecode(array, cv2.COLOR_BGR2RGB)
                        # update data of attendance
                        datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],
                                                           "%Y-%m-%d %H:%M:%S")
                        secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                        # print(secondsElapsed)
                        if secondsElapsed > 120:
                            ref = db.reference(f'students/{id}')
                            studentInfo['total_attendance'] += 1
                            ref.child('total_attendance').set(studentInfo['total_attendance'])
                            ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        else:
                            modeType = 3
                            counter = 0
                            imgBackground[180:180 + imgModeList[modeType].shape[0],
                            822:822 + imgModeList[modeType].shape[1]] = \
                                imgModeList[modeType]
                    if modeType != 3:
                        if 10 < counter < 20:
                            # switch mode type to marked
                            modeType = 2

                        imgBackground[180:180 + imgModeList[modeType].shape[0],
                        822:822 + imgModeList[modeType].shape[1]] = \
                            imgModeList[modeType]
                        if counter <= 10:
                            cv2.putText(imgBackground, str(studentInfo['firstName']), (1000, 440),
                                        cv2.FONT_HERSHEY_COMPLEX, 1,
                                        (255, 255, 255), 1)

                            cv2.putText(imgBackground, str(studentInfo['last_attendance_time']), (1000, 640),
                                        cv2.FONT_HERSHEY_COMPLEX,
                                        1,
                                        (255, 255, 255), 1)
                            imgBackground[180:180 + imgStudent.shape[0], 822:822 + imgStudent.shape[1]] = imgStudent

                        counter += 1

                        if counter >= 20:
                            counter = 0
                            modeType = 0
                            studentInfo = []
                            imgStudent = []
                            imgBackground[180:180 + imgModeList[modeType].shape[0],
                            822:822 + imgModeList[modeType].shape[1]] = \
                                imgModeList[modeType]
        else:
            modeType = 0
            counter = 0

        if not success:
            print("Failed to grab frame")
            break
        if imgBackground is not None:

            cv2.imshow("Face attendance", imgBackground)
        else:
            cv2.imshow("Webcam", img)
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break
    cap.release()
    cv2.destroyAllWindows()
