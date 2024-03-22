import pickle

import cv2
import os

import face_recognition

cap = cv2.VideoCapture(0)
cap.set(3, 600)
cap.set(4, 480)

imgBackground = cv2.imread('../Resources/background.png')

floderModePath =  '../Resources/Modes'
modePathList  = os.listdir(floderModePath)
imgModeList= []

for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(floderModePath, path)))


# load the encoding file
print('Loading encode file')
file = open('../EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
# print(studentIds)
print("Encode File loaded")

while True:
    success, img = cap.read()


    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faceCurFrame  = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print("matches", matches)
        print("faceDistance", faceDis)

    if not success:
        print("Failed to grab frame")
        break
    if imgBackground is not None:
        # Place the webcam feed at a specific location on the background image
        imgBackground[220:220 + img.shape[0], 10:10 + img.shape[1]] = img
        imgBackground[180:180 + imgModeList[3].shape[0], 822:822 + imgModeList[3].shape[1]] = imgModeList[3]
        cv2.imshow("Face attendance", imgBackground)
    else:
        cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
