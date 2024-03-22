import cv2
from flask import Flask, render_template, Response, request, redirect, url_for

app = Flask(__name__)

camera = None  # Initialized as None to be set on successful login

def generate_frames():
    global camera
    while True:
        if camera is not None:
            success, frame = camera.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    global camera
    username = request.form['username']
    password = request.form['password']

    # Simple username and password check
    if username == 'user' and password == 'password':
        camera = cv2.VideoCapture(0)
        return redirect(url_for('webcam'))
    else:
        return "Invalid login credentials"
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', username='user', email='test@test.com')
@app.route('/webcam')
def webcam():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if camera is None:
        return "Please log in to access the video feed."

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout')
def logout():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
