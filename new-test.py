import cvzone
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, session
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import pickle
import os
import cv2
import face_recognition



app = Flask(__name__)




cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL' : "https://faceattendance-3feef-default-rtdb.firebaseio.com/",
    'storageBucket': 'faceattendance-3feef.appspot.com'
})
bucket = storage.bucket()

def findEncodings(imagesList):
    encodeList= []
    for img in imagesList:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        AdminInfo = db.reference(f'Admins').get()

        if AdminInfo is not None:
            for admin_data in AdminInfo:
                if admin_data and 'Email' in admin_data and admin_data['Email'] == email and admin_data['password'] == password:
                    session['user_id'] = admin_data.get('id')
                    session['user_name'] = admin_data.get('name')
                    return redirect('/')

        return "Invalid credentials"

    return render_template('login.html')

@app.route('/mark_attendance')
def start_attendance():
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
    return render_template('dashboard.html')


@app.route('/', methods=['GET', 'POST'])
def view_dashboard():
    if 'user_id' in session:
        user_name = session.get('user_name')
        return render_template('dashboard.html', user_name=user_name)
    return redirect('/login')

def get_next_student_id():
    students_ref = db.reference('students')
    students = students_ref.get()

    if students:
        # If students exist, generate the next ID
        return str(len(students) )
    else:
        return "1"  # If no students exist, start with ID 1



@app.route('/add-student', methods=['GET', 'POST'])
def add_student():
    if 'user_id' in session:
        user_name = session.get('user_name')
        if request.method == 'POST':
            new_student_id = get_next_student_id()
            firstName = request.form['firstName']
            lastName = request.form['lastName']
            personalEmail = request.form['personalEmail']
            institutionEmail = request.form['institutionEmail']
            faculty = request.form['faculty']
            major = request.form['major']
            photo = request.files['profilePic']

            students_ref = db.reference('students')
            students_ref.child(new_student_id).set({
                'id': new_student_id,
                'firstName': firstName,
                'lastName': lastName,
                'personalEmail': personalEmail,
                'institutionEmail': institutionEmail,
                'faculty': faculty,
                'major': major,
                'total_attendance': 0,
                'last_attendance_time': '2023-11-06 13:04:20'
            })
            # place file in Images folder
            photo_extension = photo.filename.split('.')[-1]
            photo.save(os.path.join('Images', f"{new_student_id}.{photo_extension}"))



            # encode files and also upload files to a bucket
            folderPath = 'Images'
            pathList = os.listdir(folderPath)
            imgList = []
            studentIds = []

            for path in pathList:
                imgList.append(cv2.imread(os.path.join(folderPath, path)))
                studentIds.append(os.path.splitext(path)[0])

                fileName = f'{folderPath}/{path}'
                bucket = storage.bucket()
                blob = bucket.blob(fileName)
                blob.upload_from_filename(fileName)

                print(path)
                print(os.path.splitext(path)[0])
            print(studentIds)

            def findEncodings(imagesList):
                encodeList = []
                for img in imagesList:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    encode = face_recognition.face_encodings(img)[0]
                    encodeList.append(encode)
                return encodeList

            print("Encoding started... ")
            encodeListKnown = findEncodings(imgList)
            encodeListKnownWithIds = [encodeListKnown, studentIds]
            print("Encoding complete ")

            file = open("EncodeFile.p", 'wb')
            pickle.dump(encodeListKnownWithIds, file)
            file.close()
            print("File saved ")




            return render_template('add-student.html', mesg="Student added successfully", user_name=user_name)


        return render_template('add-student.html', user_name=user_name)
    return redirect('/login')

@app.route('/view-students', methods=['GET', 'POST'])
def view_students():
    students_ref = db.reference('students')
    students = students_ref.get()
    students_dict = {student.get('id'): student for student in students if student and 'id' in student}
    return render_template('view-students.html', students=students_dict)


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect('/')


if __name__ == '__main__':
    app.secret_key = 'super_secret_key'
    app.run(debug=True)
