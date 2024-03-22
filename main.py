import cvzone
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, session, url_for, make_response
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import pickle
import os
import cv2
import face_recognition
import pdfkit
from collections import defaultdict

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


@app.route('/generate-report', methods=['GET'])
def generate_report():
    students_ref = db.reference('students')
    students = students_ref.get()

    students_dict = {}

    if students and isinstance(students, list):
        students_dict = {student.get('id'): student for student in students if student and 'id' in student}

    attendance_per_day = defaultdict(int)

    for student_data in students_dict.values():
        if 'last_attendance_time' in student_data:
            date = student_data['last_attendance_time'].split()[0]
            attendance_per_day[date] += 1

    # Render HTML template with the attendance data
    rendered_html = render_template('view-records.html', attendance_per_day=attendance_per_day)

    # Generate PDF from HTML content
    pdfkit.from_string(rendered_html, 'report.pdf')

    # Serve the generated PDF
    response = make_response(open('report.pdf', 'rb').read())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'inline; filename=report.pdf'
    return response


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        userInfo = db.reference(f'users').get()

        if userInfo is not None:
            for user_data in userInfo:
                if user_data and 'Email' in user_data and user_data['Email'] == email and user_data['password'] == password:
                    session['user_id'] = user_data.get('id')
                    session['user_name'] = user_data.get('name')
                    session['email'] = user_data.get('Email')
                    session['role'] = user_data.get('role')
                    if 'major' in user_data:
                        session['major'] = user_data.get('major')
                    return redirect('/')

        return render_template('login.html', err="Invalid Credentials try again")

    return render_template('login.html')


@app.route('/mark_attendance')
def start_attendance():
    #Improvement: Display a html template showing capturing attendance with an option to stop if stop is clicked destroy opencv window and redirect to view-students
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
                    # bbox = 275 + x1, 10 + y1, x2 - x1, y2 - y1
                    # cvzone.cornerRect(imgBackground, bbox, rt=0)
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
                        # seconds elapsed should be greater than 24 hrs to record attendance again
                        if secondsElapsed > 86400:
                            ref = db.reference(f'students/{id}')
                            studentInfo['total_attendance'] += 1
                            ref.child('total_attendance').set(studentInfo['total_attendance'])
                            ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            insert_attendance_record(id)
                        else:
                            modeType = 3
                            counter = 0
                            imgBackground[208:208 + imgModeList[modeType].shape[0],
                            920:920 + imgModeList[modeType].shape[1]] = \
                                imgModeList[modeType]
                    if modeType != 3:
                        if 10 < counter < 20:
                            # switch mode type to marked
                            modeType = 2

                        imgBackground[208:208 + imgModeList[modeType].shape[0],
                        920:920 + imgModeList[modeType].shape[1]] = \
                            imgModeList[modeType]
                        if counter <= 10:
                            cv2.putText(imgBackground, str("ID: " + studentInfo['id']), (921, 440),
                                        cv2.FONT_HERSHEY_COMPLEX, 1,
                                        (0, 0, 0), 1)
                            cv2.putText(imgBackground, str(studentInfo['firstName'] + " " + studentInfo['lastName']), (921, 480),
                                        cv2.FONT_HERSHEY_COMPLEX, 1,
                                        (0, 0, 0), 1)
                            cv2.putText(imgBackground, str(studentInfo['major']), (921, 530),
                                        cv2.FONT_HERSHEY_COMPLEX, 1,
                                        (0, 0, 0), 1)


                            imgBackground[208:208 + imgStudent.shape[0], 920:920 + imgStudent.shape[1]] = imgStudent

                        counter += 1

                        if counter >= 20:
                            counter = 0
                            modeType = 0
                            studentInfo = []
                            imgStudent = []
                            imgBackground[208:208 + imgModeList[modeType].shape[0],
                            920:920 + imgModeList[modeType].shape[1]] = \
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
    return redirect('view-students')

def insert_attendance_record(student_id):
    # Insert the student ID and current date into another database table
    today_date = datetime.now().strftime("%Y-%m-%d")
    attendance_ref = db.reference(f'attendance/{today_date}')
    attendance_ref.child(student_id).set(True)
    print(f"Attendance recorded for student {student_id} on {today_date}")

@app.route('/', methods=['GET', 'POST'])
def view_dashboard():
    if 'user_id' in session:
        user_name = session.get('user_name')
        email = session.get('email')
        role = session.get('role')
        students_ref = db.reference('students')
        students_data = students_ref.get()
        total_students = len(students_data) - 1 if students_data else 0

        students_dict = {}
        if students_data and isinstance(students_data, list):
            students_dict = {student.get('id'): student for student in students_data if student and 'id' in student}

        students_per_major = {}
        if students_dict:
            for student_id, student_data in students_dict.items():
                major = student_data.get('major')
                if major in students_per_major:
                    students_per_major[major] += 1
                else:
                    students_per_major[major] = 1

        return render_template('dashboard.html', user_name=user_name, email=email, role=role, total_students=total_students, students_per_major=students_per_major)
    return redirect('/login')


def get_next_student_id():
    students_ref = db.reference('students')
    students = students_ref.get()

    if students:
        return str(len(students) )
    else:
        return "1"  # If no students exist, start with ID 1

def get_next_user_id():
    users_ref = db.reference('users')
    users = users_ref.get()

    if users:
        # If users exist, generate the next ID
        return str(len(users) )
    else:
        return "1"  # If no users exist, start with ID 1



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
            major = request.form['major']
            photo = request.files['profilePic']

            students_ref = db.reference('students')
            students_ref.child(new_student_id).set({
                'id': new_student_id,
                'firstName': firstName,
                'lastName': lastName,
                'personalEmail': personalEmail,
                'institutionEmail': institutionEmail,
                'major': major,
                'total_attendance': 0,
                'last_attendance_time': '2000-01-01 13:00:00'
            })
            date_today = datetime.now().strftime("%Y-%m-%d")
            attendance_per_day_ref = db.reference('attendance_per_day')
            attendance_per_day_ref.child(new_student_id).set({
                date_today: False
            }
            )

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

@app.route('/add-instructor', methods=['GET', 'POST'])
def add_instructor():
    if 'user_id' in session:
        user_name = session.get('user_name')
        if request.method == 'POST':
            new_user_id = get_next_user_id()
            Email = request.form['email']
            name = request.form['fullName']
            password = request.form['password']
            major = request.form['major']


            users_ref = db.reference('users')
            users_ref.child(new_user_id).set({
                'Email': Email,
                'name': name,
                'password': password,
                'role': 'instructor',
                'major': major

            })
            return render_template('add-instructor.html', mesg="Instructor added successfully", user_name=user_name)


        return render_template('add-instructor.html', user_name=user_name)
    return redirect('/login')


@app.route('/view-students', methods=['GET', 'POST'])
def view_students():
    students_ref = db.reference('students')
    students = students_ref.get()
    students_dict = {}
    if students and isinstance(students, list):
        students_dict = {student.get('id'): student for student in students if student and 'id' in student}
    return render_template('view-students.html', students=students_dict)

def get_student_name(student_id):
    students_ref = db.reference('students')
    student_info = students_ref.child(student_id).get()
    if student_info:
        return f"{student_info.get('firstName', '')} {student_info.get('lastName', '')}"
    else:
        return "Unknown Student"


@app.route('/view-records-per-student', methods=['GET', 'POST'])
def view_records_per_student():
    students_ref = db.reference('students')
    students = students_ref.get()
    students_dict = {}
    if students and isinstance(students, list):
        students_dict = {student.get('id'): student for student in students if student and 'id' in student}
    return render_template('view-records-per-student.html', students=students_dict)





@app.route('/view-instructors', methods=['GET', 'POST'])
def view_instructors():
    users_ref = db.reference('users')
    users = users_ref.get()
    users_dict = {}
    if users and isinstance(users, list):
        users_dict = {user.get('Email'): user for user in users if user and 'Email' in user}
    return render_template('view-instructors.html', users=users_dict)


@app.route('/total-attendance-per-day')
def total_attendance_per_day():
    # Fetch total attendance data for all students from the database
    total_attendance_data = get_total_attendance_data()

    return render_template('total_attendance_per_day.html', total_attendance_data=total_attendance_data)


def get_total_attendance_data():
    total_attendance_data = []

    # Fetch dates from the last 30 days (adjust as needed)
    today = datetime.now().date()
    date_range = [today - timedelta(days=x) for x in range(30)]

    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        attendance_ref = db.reference(f'attendance/{date_str}')

        # Check if attendance data exists for the current date
        attendance_data = attendance_ref.get()
        if attendance_data:
            # Count the number of present students for the current date
            present_students_count = sum(1 for student_id in attendance_data.values() if student_id)
        else:
            present_students_count = 0

        # Append total attendance record for the current date
        total_attendance_data.append({'date': date_str, 'present_students_count': present_students_count})

    return total_attendance_data


@app.route('/delete_student/<string:student_id>', methods=['GET', 'POST'])
def delete_student(student_id):
    if request.method == 'POST':
        # Delete the student from the database
        db.reference(f'students/{student_id}').delete()
        return redirect('view_students')
    student_info = db.reference(f'students/{student_id}').get()
    return render_template('confirm_delete.html', student_info=student_info)

@app.route('/edit-student/<int:student_id>', methods=['GET', 'POST'])
def edit_student(student_id):

    student_ref = db.reference(f'students/{student_id}')
    student = student_ref.get()

    if request.method == 'POST':

        new_first_name = request.form.get('new_first_name')
        student_ref.child('firstName').set(new_first_name)

        new_last_name = request.form.get('new_last_name')
        student_ref.child('lastName').set(new_last_name)

        new_major = request.form.get('new_major')
        student_ref.child('major').set(new_major)

        new_personal_email = request.form.get('new_personal_email')
        student_ref.child('personalEmail').set(new_personal_email)

        new_institution_email = request.form.get('new_institution_email')
        student_ref.child('institutionEmail').set(new_institution_email)

        return redirect(url_for('view_students'))

    return render_template('edit_student.html', student=student)


@app.route('/view-student-attendance-per-day/<student_id>')
def view_student_attendance_per_day(student_id):
    student_name = get_student_name(student_id)
    attendance_data = get_attendance_data(student_id)
    return render_template('view-student-attendance-per-day.html', student_name=student_name, student_id=student_id,
                           attendance_data=attendance_data)


def get_attendance_data(student_id):
    attendance_data = []
    today = datetime.now().date()
    date_range = [today - timedelta(days=x) for x in range(30)]

    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        attendance_ref = db.reference(f'attendance/{date_str}')
        is_present = attendance_ref.child(student_id).get()
        attendance_data.append({'date': date_str, 'present': is_present})

    return attendance_data


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')


if __name__ == '__main__':
    app.secret_key = 'super_secret_key'
    app.run(debug=True)
