from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import sqlite3
import os
import base64
import cv2
import numpy as np
from datetime import datetime


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

# Secret key for login session
app.secret_key = "face_attendance_admin_secret_2026"


# =========================================================
# ADMIN LOGIN
# =========================================================

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE = os.path.join(BASE_DIR, "attendance.db")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_FILE = os.path.join(BASE_DIR, "face_model.yml")
LABEL_FILE = os.path.join(BASE_DIR, "labels.npy")


# =========================================================
# FACE DETECTION
# =========================================================

CASCADE_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(CASCADE_FILE)

os.makedirs(DATASET_DIR, exist_ok=True)


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_db():

    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row

    return conn


# =========================================================
# CREATE DATABASE TABLES
# =========================================================

def create_tables():

    conn = get_db()
    cursor = conn.cursor()

    # STUDENTS TABLE
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            registered_date TEXT
        )
    """)

    # ATTENDANCE TABLE
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            name TEXT,
            date TEXT,
            time TEXT,
            status TEXT
        )
    """)

    # REMOVE OLD DUPLICATES
    cursor.execute("""
        DELETE FROM attendance
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM attendance
            GROUP BY student_id, date
        )
    """)

    # PREVENT FUTURE DUPLICATES
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS
        unique_student_daily_attendance
        ON attendance(student_id, date)
    """)

    conn.commit()
    conn.close()


create_tables()


# =========================================================
# ADMIN LOGIN PAGE
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    # If already logged in
    if session.get("admin_logged_in"):
        return redirect(url_for("students"))

    error = None

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:

            session["admin_logged_in"] = True

            return redirect(url_for("students"))

        else:

            error = "Invalid username or password"

    return render_template(
        "login.html",
        error=error
    )


# =========================================================
# ADMIN LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.pop("admin_logged_in", None)

    return redirect(url_for("login"))


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/")
def dashboard():

    conn = get_db()
    cursor = conn.cursor()

    # Total students
    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM students
    """)

    total_students = cursor.fetchone()["total"]

    # Current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Present today
    cursor.execute("""
        SELECT COUNT(DISTINCT student_id) AS total
        FROM attendance
        WHERE date = ?
    """, (current_date,))

    present_today = cursor.fetchone()["total"]

    # Total attendance
    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM attendance
    """)

    total_attendance = cursor.fetchone()["total"]

    # Recent attendance
    cursor.execute("""
        SELECT *
        FROM attendance
        ORDER BY date DESC, time DESC, id DESC
        LIMIT 50
    """)

    records = cursor.fetchall()

    conn.close()

    return render_template(
        "dashboard.html",
        total_students=total_students,
        present_today=present_today,
        total_attendance=total_attendance,
        current_date=current_date,
        records=records
    )


# =========================================================
# STUDENTS PAGE - ADMIN ONLY
# =========================================================

@app.route("/students")
def students():

    # Check login
    if not session.get("admin_logged_in"):

        return redirect(
            url_for("login")
        )

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM students
        ORDER BY id DESC
    """)

    student_records = cursor.fetchall()

    conn.close()

    return render_template(
        "students.html",
        students=student_records
    )


# =========================================================
# SAVE FACE - ADMIN ONLY
# =========================================================

@app.route("/save_face", methods=["POST"])
def save_face():

    # Security check
    if not session.get("admin_logged_in"):

        return jsonify({
            "success": False,
            "message": "Unauthorized. Admin login required."
        }), 401

    try:

        data = request.get_json()

        student_id = data.get("student_id", "").strip()
        student_name = data.get("student_name", "").strip()
        image_data = data.get("image", "")

        # VALIDATION
        if not student_id:

            return jsonify({
                "success": False,
                "message": "Student ID is required"
            })

        if not student_name:

            return jsonify({
                "success": False,
                "message": "Student name is required"
            })

        if not image_data:

            return jsonify({
                "success": False,
                "message": "Image is required"
            })

        # REMOVE BASE64 HEADER
        if "," in image_data:

            image_data = image_data.split(",", 1)[1]

        # DECODE IMAGE
        image_bytes = base64.b64decode(image_data)

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            return jsonify({
                "success": False,
                "message": "Invalid image"
            })

        # GRAYSCALE
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # DETECT FACE
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:

            return jsonify({
                "success": False,
                "message": "No face detected"
            })

        # LARGEST FACE
        x, y, w, h = max(
            faces,
            key=lambda rect: rect[2] * rect[3]
        )

        # PADDING
        padding = 20

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)

        x2 = min(
            gray.shape[1],
            x + w + padding
        )

        y2 = min(
            gray.shape[0],
            y + h + padding
        )

        face = gray[y1:y2, x1:x2]

        # RESIZE
        face = cv2.resize(
            face,
            (200, 200)
        )

        # STUDENT FOLDER
        student_folder = os.path.join(
            DATASET_DIR,
            student_id
        )

        os.makedirs(
            student_folder,
            exist_ok=True
        )

        # NEXT IMAGE NUMBER
        existing_images = [
            file
            for file in os.listdir(student_folder)
            if file.lower().endswith(".jpg")
        ]

        image_number = len(existing_images) + 1

        image_path = os.path.join(
            student_folder,
            f"{image_number}.jpg"
        )

        # SAVE IMAGE
        cv2.imwrite(
            image_path,
            face
        )

        # SAVE STUDENT
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO students
            (student_id, name, registered_date)
            VALUES (?, ?, ?)
            ON CONFLICT(student_id)
            DO UPDATE SET
                name = excluded.name
        """, (
            student_id,
            student_name,
            datetime.now().strftime("%Y-%m-%d")
        ))

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": f"Face {image_number} saved successfully",
            "image_number": image_number
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        })


# =========================================================
# TRAIN FACE MODEL - ADMIN ONLY
# =========================================================

@app.route("/train_model", methods=["POST"])
def train_model():

    if not session.get("admin_logged_in"):

        return jsonify({
            "success": False,
            "message": "Unauthorized. Admin login required."
        }), 401

    try:

        if not hasattr(cv2, "face"):

            return jsonify({
                "success": False,
                "message": "OpenCV face module is not available"
            })

        recognizer = cv2.face.LBPHFaceRecognizer_create()

        faces = []
        labels = []

        label_names = {}

        label_id = 0

        for student_id in os.listdir(DATASET_DIR):

            student_folder = os.path.join(
                DATASET_DIR,
                student_id
            )

            if not os.path.isdir(student_folder):
                continue

            conn = get_db()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT *
                FROM students
                WHERE student_id = ?
            """, (student_id,))

            student = cursor.fetchone()

            conn.close()

            if student is None:
                continue

            label_names[label_id] = {
                "student_id": student_id,
                "name": student["name"]
            }

            for image_file in os.listdir(student_folder):

                if not image_file.lower().endswith(".jpg"):
                    continue

                image_path = os.path.join(
                    student_folder,
                    image_file
                )

                image = cv2.imread(
                    image_path,
                    cv2.IMREAD_GRAYSCALE
                )

                if image is None:
                    continue

                image = cv2.resize(
                    image,
                    (200, 200)
                )

                faces.append(image)
                labels.append(label_id)

            label_id += 1

        if len(faces) == 0:

            return jsonify({
                "success": False,
                "message": "No training images found in dataset"
            })

        recognizer.train(
            faces,
            np.array(labels)
        )

        recognizer.write(
            MODEL_FILE
        )

        np.save(
            LABEL_FILE,
            label_names,
            allow_pickle=True
        )

        return jsonify({
            "success": True,
            "message": "Face model trained successfully",
            "images": len(faces),
            "students": len(label_names)
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        })


# =========================================================
# START ATTENDANCE PAGE
# =========================================================

@app.route("/start_attendance")
def start_attendance():

    return render_template(
        "start_attendance.html"
    )


# =========================================================
# MARK ATTENDANCE
# =========================================================

@app.route("/mark_attendance", methods=["POST"])
def mark_attendance():

    conn = None

    try:

        if not os.path.exists(MODEL_FILE):

            return jsonify({
                "success": False,
                "message": "Face model not found. Train the model first."
            })

        if not os.path.exists(LABEL_FILE):

            return jsonify({
                "success": False,
                "message": "Labels file not found. Train the model first."
            })

        if not hasattr(cv2, "face"):

            return jsonify({
                "success": False,
                "message": "OpenCV face module is not available"
            })

        data = request.get_json()

        image_data = data.get("image", "")

        if not image_data:

            return jsonify({
                "success": False,
                "message": "Image is required"
            })

        if "," in image_data:

            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            return jsonify({
                "success": False,
                "message": "Invalid image"
            })

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:

            return jsonify({
                "success": False,
                "message": "No face detected"
            })

        x, y, w, h = max(
            faces,
            key=lambda rect: rect[2] * rect[3]
        )

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(
            face,
            (200, 200)
        )

        recognizer = cv2.face.LBPHFaceRecognizer_create()

        recognizer.read(
            MODEL_FILE
        )

        loaded_labels = np.load(
            LABEL_FILE,
            allow_pickle=True
        ).item()

        predicted_label, confidence = recognizer.predict(
            face
        )

        # Lower LBPH confidence/distance = better
        if confidence > 75:

            return jsonify({
                "success": False,
                "message": "Face not recognized",
                "confidence": round(float(confidence), 2)
            })

        label_info = loaded_labels.get(
            int(predicted_label)
        )

        if label_info is None:

            return jsonify({
                "success": False,
                "message": "Student label not found"
            })

        if isinstance(label_info, dict):

            student_id = label_info["student_id"]
            student_name = label_info["name"]

        else:

            student_id = str(label_info)
            student_name = str(label_info)

        now = datetime.now()

        current_date = now.strftime(
            "%Y-%m-%d"
        )

        current_time = now.strftime(
            "%H:%M:%S"
        )

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT *
            FROM attendance
            WHERE student_id = ?
            AND date = ?
        """, (
            student_id,
            current_date
        ))

        existing = cursor.fetchone()

        if existing:

            conn.close()

            return jsonify({
                "success": True,
                "already_present": True,
                "student_id": student_id,
                "name": student_name,
                "date": current_date,
                "time": existing["time"],
                "status": existing["status"],
                "confidence": round(float(confidence), 2),
                "message": "Attendance already marked today"
            })

        try:

            cursor.execute("""
                INSERT INTO attendance
                (student_id, name, date, time, status)
                VALUES (?, ?, ?, ?, ?)
            """, (
                student_id,
                student_name,
                current_date,
                current_time,
                "Present"
            ))

            conn.commit()

        except sqlite3.IntegrityError:

            conn.rollback()

            cursor.execute("""
                SELECT *
                FROM attendance
                WHERE student_id = ?
                AND date = ?
            """, (
                student_id,
                current_date
            ))

            existing = cursor.fetchone()

            conn.close()

            return jsonify({
                "success": True,
                "already_present": True,
                "student_id": student_id,
                "name": student_name,
                "date": current_date,
                "time": existing["time"] if existing else current_time,
                "status": existing["status"] if existing else "Present",
                "confidence": round(float(confidence), 2),
                "message": "Attendance already marked today"
            })

        conn.close()

        return jsonify({
            "success": True,
            "already_present": False,
            "student_id": student_id,
            "name": student_name,
            "date": current_date,
            "time": current_time,
            "status": "Present",
            "confidence": round(float(confidence), 2),
            "message": "Attendance marked successfully"
        })

    except Exception as e:

        if conn:

            try:
                conn.close()
            except:
                pass

        return jsonify({
            "success": False,
            "message": str(e)
        })


# =========================================================
# ATTENDANCE DATA
# =========================================================

@app.route("/attendance")
def attendance():

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM attendance
        ORDER BY date DESC, time DESC, id DESC
    """)

    records = cursor.fetchall()

    conn.close()

    return jsonify([
        dict(record)
        for record in records
    ])


# =========================================================
# DELETE ATTENDANCE - ADMIN ONLY
# =========================================================

@app.route("/delete_attendance/<int:record_id>", methods=["DELETE"])
def delete_attendance(record_id):

    if not session.get("admin_logged_in"):

        return jsonify({
            "success": False,
            "message": "Unauthorized. Admin login required."
        }), 401

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM attendance
        WHERE id = ?
    """, (record_id,))

    conn.commit()

    deleted = cursor.rowcount

    conn.close()

    if deleted == 0:

        return jsonify({
            "success": False,
            "message": "Attendance record not found"
        })

    return jsonify({
        "success": True,
        "message": "Attendance deleted successfully"
    })


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print(" Face Recognition Attendance System")
    print("==========================================")
    print()
    print("Dashboard       : http://127.0.0.1:5000/")
    print("Login           : http://127.0.0.1:5000/login")
    print("Students        : http://127.0.0.1:5000/students")
    print("Start Attendance: http://127.0.0.1:5000/start_attendance")
    print()

    app.run(
        debug=True
    )