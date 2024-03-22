import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL' : "https://faceattendance-3feef-default-rtdb.firebaseio.com/"
})

ref = db.reference('records')
data = {
    "1":
        {"attendance_records": "0",
         "date": "2023-11-06 13:04:20",
         "major": ""
         }


}

for key, value in data.items():
    ref.child(key).set(value)