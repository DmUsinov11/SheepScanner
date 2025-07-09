from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import sqlite3
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

try:
    model = YOLO('weights/best.pt')
except:
    model = YOLO('yolov8n.pt')

def init_db():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            sheep_count INTEGER,
            confidence_avg REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format'}), 400
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Cannot read image'}), 400
        
        results = model(img)
        
        sheep_detections = []
        confidence_scores = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    sheep_detections.append(box)
                    confidence_scores.append(float(box.conf))
        
        sheep_count = len(sheep_detections)
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        output_img = img.copy()
        
        for box in sheep_detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf)
            
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f'Sheep: {confidence:.2f}'
            cv2.putText(output_img, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(output_img, f'Total: {sheep_count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        cv2.imwrite(result_path, output_img)
        
        conn = sqlite3.connect('history.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO detections (timestamp, filename, sheep_count, confidence_avg)
            VALUES (?, ?, ?, ?)
        ''', (datetime.now().isoformat(), filename, sheep_count, avg_confidence))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'sheep_count': sheep_count,
            'avg_confidence': round(avg_confidence, 3),
            'result_image': f'results/{result_filename}',
            'original_image': f'uploads/{filename}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def get_history():
    try:
        conn = sqlite3.connect('history.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, filename, sheep_count, confidence_avg
            FROM detections
            ORDER BY timestamp DESC
            LIMIT 10
        ''')
        history = cursor.fetchall()
        conn.close()
        
        return jsonify([{
            'timestamp': row[0],
            'filename': row[1], 
            'sheep_count': row[2],
            'confidence_avg': row[3]
        } for row in history])
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 