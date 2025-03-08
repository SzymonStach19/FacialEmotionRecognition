from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import os
import numpy as np
import random
from fer import FER
import base64
from werkzeug.utils import secure_filename
import uuid
import mysql.connector
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'/Users/szymon/Hackathon2/HackathonGit/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Database initialization
def connect_to_db():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",  # domyślnie użytkownik to "root"
        password="password",  # w XAMPP hasło jest domyślnie puste
        database="face_analysis"
    )
    return connection

def create_table():
    connection = connect_to_db()
    cursor = connection.cursor()
    
    # Create photo table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS photo (
        id INT AUTO_INCREMENT PRIMARY KEY,
        path VARCHAR(255) NOT NULL
    )
    """)
    
    # Create face table to store individual faces
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS face (
        id INT AUTO_INCREMENT PRIMARY KEY,
        photo_id INT,
        face_number INT,
        FOREIGN KEY (photo_id) REFERENCES photo(id)
    )
    """)
    
    # Create emotions table with foreign key to face
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS emotions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        face_id INT,
        emotion_name VARCHAR(255),
        emotion_value FLOAT,
        FOREIGN KEY (face_id) REFERENCES face(id)
    )
    """)
    
    connection.commit()
    connection.close()


def normalize_emotions(emotions):
    total = sum(emotions.values())
    return {k: (v/total) * 100 for k, v in emotions.items()}

def apply_random_transformations(face_img):
    # Shift
    if random.choice([True, False]):
        max_shift = 20
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        face_img = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]))
    
    # Brightness
    if random.choice([True, False]):
        brightness_factor = random.uniform(0.5, 1.5)
        face_img = cv2.convertScaleAbs(face_img, alpha=brightness_factor, beta=0)
    
    # Zoom
    if random.choice([True, False]):
        zoom_factor = random.uniform(1.1, 1.5)
        width, height = int(face_img.shape[1] * zoom_factor), int(face_img.shape[0] * zoom_factor)
        face_img = cv2.resize(face_img, (width, height))
        face_img = face_img[:face_img.shape[0], :face_img.shape[1]]
    
    return face_img

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, "Failed to load image"
    
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    faces = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )
    
    faces_data = []
    for i, (x, y, w, h) in enumerate(faces):
        face_img = img[y:y + h, x:x + w]
        _, buffer = cv2.imencode('.jpg', face_img)
        face_base64 = base64.b64encode(buffer).decode('utf-8')
        
        faces_data.append({
            'id': i,
            'x': int(x),
            'y': int(y),
            'w': int(w),
            'h': int(h),
            'image': face_base64
        })
    
    return faces_data, None

def analyze_face(face_img, num_transformations=150):
    emotion_detector = FER()
    transformed_emotions = []
    
    for _ in range(num_transformations):
        transformed_face = apply_random_transformations(face_img)
        face_img_rgb = cv2.cvtColor(transformed_face, cv2.COLOR_BGR2RGB)
        detected_emotions = emotion_detector.detect_emotions(face_img_rgb)
        
        if detected_emotions:
            emotion_scores = detected_emotions[0]["emotions"]
            transformed_emotions.append(emotion_scores)
        else:
            transformed_emotions.append({"neutral": 1.0})
    
    avg_emotions = {
        emotion: 0 for emotion in [
            "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
        ]
    }
    
    for emotions in transformed_emotions:
        for emotion, score in emotions.items():
            avg_emotions[emotion] += score
    
    num_transforms = len(transformed_emotions)
    for emotion in avg_emotions:
        avg_emotions[emotion] /= num_transforms
    
    # Normalize emotions to sum to 100%
    return normalize_emotions(avg_emotions)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        faces_data, error = process_image(filepath)
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify({'faces': faces_data, 'filepath': filepath})

@app.route('/analyze_faces', methods=['POST'])
def analyze_faces():
    data = request.json
    filepath = data.get('filepath')
    accepted_faces = data.get('accepted_faces', [])

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400

    img = cv2.imread(filepath)
    results = []

    for face_data in accepted_faces:
        x, y, w, h = face_data['x'], face_data['y'], face_data['w'], face_data['h']
        face_img = img[y:y + h, x:x + w]
        emotions = analyze_face(face_img)
        results.append({
            'face_id': face_data['id'],
            'emotions': emotions
        })

    # Zmieniona część kodu - używamy face_id zamiast i
    for result in results:
        face_data = next(f for f in accepted_faces if f['id'] == result['face_id'])
        x, y = face_data['x'], face_data['y']
        # Używamy face_id zamiast iteratora
        cv2.putText(img, f"Face {result['face_id']}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.rectangle(img, (x, y), (x + face_data['w'], y + face_data['h']), (0, 255, 0), 2)

        y_offset = y + face_data['h'] + 20
        for emotion, score in result['emotions'].items():
            text = f"{emotion.capitalize()}: {score:.1f}%"
            cv2.putText(img, text, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_offset += 20

    output_filename = f"result_{uuid.uuid4()}.jpg"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    cv2.imwrite(output_path, img)

    return jsonify({
        'results': results,
        'output_image': output_filename
    })

@app.route('/save_results', methods=['POST'])
def save_results():
    data = request.get_json()

    if not data or 'image_path' not in data or 'emotions' not in data:
        return jsonify({'success': False, 'error': 'Invalid data'}), 400

    image_path = data['image_path']
    emotions = data['emotions']

    connection = connect_to_db()
    cursor = connection.cursor()

    try:
        # First, insert the photo record
        cursor.execute("""
        INSERT INTO photo (path)
        VALUES (%s)
        """, (image_path,))
        
        # Get the ID of the inserted photo
        photo_id = cursor.lastrowid

        # Insert face and emotions for each detected face
        for emotion in emotions:
            # Insert face record
            cursor.execute("""
            INSERT INTO face (photo_id, face_number)
            VALUES (%s, %s)
            """, (photo_id, emotion['face_id']))
            
            # Get the ID of the inserted face
            face_id = cursor.lastrowid
            
            # Insert emotions for this face
            for name, value in emotion['emotions'].items():
                rounded_value = round(float(value), 1)
                cursor.execute("""
                INSERT INTO emotions (face_id, emotion_name, emotion_value)
                VALUES (%s, %s, %s)
                """, (face_id, name, rounded_value))

        connection.commit()
        return jsonify({'success': True}), 200
    except Exception as e:
        print(f"Error: {e}")
        connection.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        connection.close()

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
@app.route('/get_archive', methods=['GET'])
def get_archive():
    connection = connect_to_db()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Get all photos with their faces and emotions
        cursor.execute("""
        SELECT 
            p.id as photo_id, 
            p.path, 
            f.id as face_id,
            f.face_number,
            GROUP_CONCAT(
                DISTINCT CONCAT(e.emotion_name, ':', e.emotion_value)
            ) as emotions
        FROM photo p
        LEFT JOIN face f ON p.id = f.photo_id
        LEFT JOIN emotions e ON f.id = e.face_id
        GROUP BY p.id, p.path, f.id, f.face_number
        ORDER BY p.id DESC, f.face_number
        """)
        
        archive_entries = cursor.fetchall()
        
        # Process the results
        result = []
        current_photo = None
        
        for entry in archive_entries:
            if current_photo is None or current_photo['image_path'] != entry['path']:
                if current_photo is not None:
                    result.append(current_photo)
                current_photo = {
                    'image_path': entry['path'],
                    'faces': {}
                }
            
            if entry['emotions']:
                emotions_data = {}
                emotions_list = entry['emotions'].split(',')
                for emotion_data in emotions_list:
                    name, value = emotion_data.split(':')
                    emotions_data[name] = float(value)
                
                current_photo['faces'][entry['face_number']] = emotions_data
        
        if current_photo is not None:
            result.append(current_photo)
        
        return jsonify({'success': True, 'archive': result})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        connection.close()

@app.route('/load_archive_image', methods=['POST'])
def load_archive_image():
    data = request.get_json()
    image_path = data.get('image_path')
    
    if not image_path or not os.path.exists(image_path):
        return jsonify({'error': 'Invalid image path'}), 400
        
    faces_data, error = process_image(image_path)
    if error:
        return jsonify({'error': error}), 400
    
    connection = connect_to_db()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Get emotions for each face in the photo
        cursor.execute("""
        SELECT 
            f.face_number,
            e.emotion_name,
            e.emotion_value
        FROM photo p
        JOIN face f ON p.id = f.photo_id
        JOIN emotions e ON f.id = e.face_id
        WHERE p.path = %s
        ORDER BY f.face_number
        """, (image_path,))
        
        emotions_data = cursor.fetchall()
        
        # Group emotions by face
        faces_emotions = {}
        for emotion in emotions_data:
            face_number = emotion['face_number']
            if face_number not in faces_emotions:
                faces_emotions[face_number] = {}
            faces_emotions[face_number][emotion['emotion_name']] = emotion['emotion_value']
        
        return jsonify({
            'faces': faces_data,
            'filepath': image_path,
            'emotions': faces_emotions
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        connection.close()


if __name__ == '__main__':
    connect_to_db()
    create_table()
    app.run(debug=True)