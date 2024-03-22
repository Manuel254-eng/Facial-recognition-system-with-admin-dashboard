from flask import Flask, render_template, Response
import cv2
import face_recognition
import os

app = Flask(__name__)

# Function to load known faces
def load_known_faces(folder_path):
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(folder_path):
        img = face_recognition.load_image_file(os.path.join(folder_path, filename))
        face_encoding = face_recognition.face_encodings(img)[0]  # Assuming one face in each image
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(filename)[0])  # Extract the name from the file
    return known_face_encodings, known_face_names

# Load known faces from the 'Images' folder
known_encodings, known_names = load_known_faces('Images')

video_capture = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            # Convert the frame from BGR to RGB for face recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Detect face locations in the frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Compare detected faces with known faces
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]

                # Draw rectangles and labels around the faces
                for (top, right, bottom, left), name in zip(face_locations, name):
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
