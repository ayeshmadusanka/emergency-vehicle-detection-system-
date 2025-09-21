from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import os
import io
import torch
import random
from PIL import Image
from emergency_detector import EmergencyVehiclePredictor
from traffic_system import CNNTrafficSystem

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize the emergency vehicle detector
detector = EmergencyVehiclePredictor()

# Initialize the traffic management system
try:
    traffic_system = CNNTrafficSystem(
        num_lanes=4,
        emergency_threshold=0.5,  # Lower threshold for better emergency detection
        device='cpu'
    )
    print("✓ Traffic management system initialized")
except Exception as e:
    print(f"Warning: Could not initialize full traffic system: {e}")
    traffic_system = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    with open('index.html', 'r') as f:
        return f.read()

@app.route('/traffic')
def traffic_system():
    with open('traffic_interface.html', 'r') as f:
        return f.read()

@app.route('/traffic_system.js')
def traffic_js():
    with open('traffic_system.js', 'r') as f:
        return f.read(), 200, {'Content-Type': 'application/javascript'}

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']

        # Check if file was selected
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400

        # Check file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400

        # Read and process the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Make prediction
        result = detector.predict_from_pil(image)

        if 'error' in result:
            return jsonify({'error': result['error']}), 500

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/traffic/analyze', methods=['POST'])
def analyze_traffic_image():
    """Analyze image using full traffic system"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Read and process the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Emergency vehicle detection
        emergency_result = detector.predict_from_pil(image)

        # Simulate traffic analysis (since we don't have full intersection images)
        traffic_result = {
            'emergency_detected': emergency_result['is_emergency'],
            'emergency_confidence': emergency_result['confidence'],
            'vehicle_type': emergency_result['prediction'],
            'lane_counts': [random.randint(2, 12) for _ in range(4)],  # Simulate 4 lanes
            'traffic_decision': 'EMERGENCY_OVERRIDE' if emergency_result['is_emergency'] else 'CONTINUE_NORMAL',
            'recommended_action': 'Clear all lanes for emergency vehicle' if emergency_result['is_emergency'] else 'Continue normal traffic flow'
        }

        # Merge results
        result = {
            **emergency_result,
            'traffic_analysis': traffic_result
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Error analyzing traffic: {str(e)}'}), 500

@app.route('/traffic/status')
def traffic_status():
    """Get current traffic system status"""
    return jsonify({
        'status': 'active',
        'mode': 'auto',
        'emergency_override': False,
        'current_phase': 'north-south-green',
        'phase_timer': random.randint(10, 30),
        'lane_counts': {
            'north': random.randint(3, 15),
            'south': random.randint(2, 12),
            'east': random.randint(1, 8),
            'west': random.randint(4, 11)
        },
        'total_vehicles_processed': random.randint(100, 500),
        'emergency_vehicles_detected': random.randint(1, 5)
    })

@app.route('/traffic/control', methods=['POST'])
def traffic_control():
    """Control traffic lights manually"""
    data = request.get_json()
    action = data.get('action')

    if action == 'emergency_override':
        return jsonify({
            'status': 'success',
            'message': 'Emergency override activated',
            'all_lights': 'red'
        })
    elif action == 'auto_mode':
        return jsonify({
            'status': 'success',
            'message': 'Auto mode activated',
            'mode': 'auto'
        })
    elif action == 'manual_mode':
        return jsonify({
            'status': 'success',
            'message': 'Manual mode activated',
            'mode': 'manual'
        })
    else:
        return jsonify({'error': 'Invalid action'}), 400

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'emergency_detector': True,
        'traffic_system': traffic_system is not None
    })

if __name__ == '__main__':
    print("Starting Smart Traffic Management System...")
    print(f"Emergency detector device: {detector.device}")
    print("Available endpoints:")
    print("  - Emergency Detection: http://localhost:8003/")
    print("  - Traffic Management: http://localhost:8003/traffic")
    print("  - Health Check: http://localhost:8003/health")
    app.run(debug=True, host='0.0.0.0', port=8003)