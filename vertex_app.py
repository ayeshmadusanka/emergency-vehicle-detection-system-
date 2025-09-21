from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import os
import io
from PIL import Image
from vertex_emergency_detector import VertexEmergencyDetector
from vertex_direct_detector import VertexDirectDetector
from gemini_emergency_detector import GeminiEmergencyDetector
from simple_ai_detector import SimpleAIDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize the AI emergency vehicle detector (try Vertex AI first, fallback to Gemini)
detector = None
detector_status = "not_initialized"
detector_type = "unknown"

# Try Vertex AI Direct first (most reliable with Gemini 2.5 Flash)
try:
    detector = VertexDirectDetector()
    detector_status = "vertex_direct_initialized"
    detector_type = "vertex_direct"
    print("✓ Using Vertex AI Direct (Gemini 2.5 Flash) for emergency detection")
except Exception as vertex_direct_error:
    print(f"Warning: Could not initialize Vertex AI Direct detector: {vertex_direct_error}")

    # Try Gemini AI as second option
    try:
        detector = GeminiEmergencyDetector()
        detector_status = "gemini_ai_initialized"
        detector_type = "gemini_ai"
        print("✓ Using Gemini AI for emergency detection")
    except Exception as gemini_error:
        print(f"Warning: Could not initialize Gemini AI detector: {gemini_error}")

        # Try original Vertex AI as third option
        try:
            detector = VertexEmergencyDetector()
            detector_status = "vertex_ai_initialized"
            detector_type = "vertex_ai"
            print("✓ Using Vertex AI for emergency detection")
        except Exception as vertex_error:
            print(f"Warning: Could not initialize Vertex AI detector: {vertex_error}")

            # Use Simple AI as final fallback
            try:
                detector = SimpleAIDetector()
                detector_status = "simple_ai_initialized"
                detector_type = "simple_ai"
                print("✓ Using Simple AI (Computer Vision) for emergency detection")
            except Exception as simple_error:
                print(f"Error: Could not initialize any detector: {simple_error}")
                detector = None
                detector_status = f"error: All detectors failed"
                detector_type = "none"

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vertex AI Emergency Vehicle Detection</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        'vertex': '#4285F4',
                        'emergency': '#ef4444',
                        'regular': '#10b981',
                    }
                }
            }
        }
    </script>
</head>
<body class="bg-gradient-to-br from-blue-50 to-purple-50 min-h-screen">
    <!-- Header -->
    <div class="bg-white shadow-lg border-b border-gray-200">
        <div class="container mx-auto px-4 py-6">
            <div class="flex items-center justify-between">
                <div>
                    <h1 class="text-3xl font-bold text-gray-900">🤖 Vertex AI Emergency Vehicle Detection</h1>
                    <p class="text-gray-600 mt-1">Powered by Google Cloud Gemini Vision AI</p>
                </div>
                <div class="flex items-center space-x-2">
                    <div id="status-indicator" class="w-3 h-3 bg-green-500 rounded-full"></div>
                    <span class="text-sm text-gray-600">Vertex AI Active</span>
                </div>
            </div>
        </div>
    </div>

    <div class="container mx-auto px-4 py-8">
        <div class="max-w-4xl mx-auto">
            <!-- Upload Section -->
            <div class="bg-white rounded-xl shadow-xl p-8 mb-8">
                <h2 class="text-2xl font-bold text-gray-800 mb-6 text-center">🚨 Upload Vehicle Image for AI Analysis</h2>

                <!-- Upload Area -->
                <div class="mb-6">
                    <div class="border-2 border-dashed border-vertex rounded-xl p-8 text-center hover:border-blue-500 transition-all duration-300 bg-blue-50 hover:bg-blue-100">
                        <input type="file" id="imageInput" accept="image/*" class="hidden">
                        <div id="uploadArea" class="cursor-pointer" onclick="document.getElementById('imageInput').click()">
                            <div class="mx-auto w-16 h-16 bg-vertex rounded-full flex items-center justify-center mb-4">
                                <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                                </svg>
                            </div>
                            <p class="text-xl text-gray-700 mb-2 font-semibold">Drop an image here or click to upload</p>
                            <p class="text-sm text-gray-500">PNG, JPG, GIF up to 16MB • Powered by Gemini Vision</p>
                        </div>
                    </div>
                </div>

                <!-- Image Preview -->
                <div id="imagePreview" class="hidden mb-6">
                    <div class="bg-gray-50 rounded-lg p-4">
                        <h3 class="text-lg font-semibold text-gray-800 mb-3">📷 Image Preview</h3>
                        <img id="previewImg" class="max-w-full h-auto rounded-lg mx-auto max-h-96 shadow-md" alt="Uploaded image">
                    </div>
                </div>

                <!-- Analyze Button -->
                <div class="text-center mb-6">
                    <button id="analyzeBtn" class="hidden bg-vertex hover:bg-blue-600 text-white font-bold py-4 px-8 rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg">
                        <span id="analyzeText">🔍 Analyze with Vertex AI</span>
                        <div id="loadingSpinner" class="hidden inline-block ml-3">
                            <svg class="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                        </div>
                    </button>
                </div>

                <!-- Results Section -->
                <div id="results" class="hidden">
                    <div class="bg-gradient-to-r from-gray-50 to-gray-100 rounded-xl p-6 border">
                        <h3 class="text-xl font-bold text-gray-800 mb-4">🤖 Vertex AI Analysis Results</h3>
                        <div id="resultsContent"></div>
                    </div>
                </div>

                <!-- Error Section -->
                <div id="errorSection" class="hidden bg-red-50 border border-red-200 rounded-xl p-4">
                    <div class="flex items-start">
                        <svg class="h-6 w-6 text-red-400 mr-3 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
                        </svg>
                        <div>
                            <h3 class="text-red-800 font-bold">Analysis Error</h3>
                            <p id="errorMessage" class="text-red-700 text-sm mt-1"></p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- AI Info Section -->
            <div class="bg-white rounded-xl shadow-lg p-6">
                <h3 class="text-lg font-bold text-gray-800 mb-4">🧠 About this AI System</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-600">
                    <div class="bg-blue-50 p-4 rounded-lg">
                        <h4 class="font-semibold text-blue-800 mb-2">🔍 Detection Method</h4>
                        <p>Google Vertex AI Gemini Vision model analyzes images to identify emergency vehicles with high accuracy.</p>
                    </div>
                    <div class="bg-green-50 p-4 rounded-lg">
                        <h4 class="font-semibold text-green-800 mb-2">🚨 Vehicle Types</h4>
                        <p>Detects ambulances, fire trucks, police cars, and other emergency service vehicles.</p>
                    </div>
                    <div class="bg-purple-50 p-4 rounded-lg">
                        <h4 class="font-semibold text-purple-800 mb-2">⚡ Real-time Analysis</h4>
                        <p>Cloud-based processing provides instant results with detailed reasoning and confidence scores.</p>
                    </div>
                    <div class="bg-orange-50 p-4 rounded-lg">
                        <h4 class="font-semibold text-orange-800 mb-2">🛡️ Privacy & Security</h4>
                        <p>Images are processed securely through Google Cloud with enterprise-grade security.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const imageInput = document.getElementById('imageInput');
        const imagePreview = document.getElementById('imagePreview');
        const previewImg = document.getElementById('previewImg');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const analyzeText = document.getElementById('analyzeText');
        const loadingSpinner = document.getElementById('loadingSpinner');
        const results = document.getElementById('results');
        const errorSection = document.getElementById('errorSection');

        imageInput.addEventListener('change', handleImageUpload);
        analyzeBtn.addEventListener('click', analyzeImage);

        function handleImageUpload(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImg.src = e.target.result;
                    imagePreview.classList.remove('hidden');
                    analyzeBtn.classList.remove('hidden');
                    results.classList.add('hidden');
                    errorSection.classList.add('hidden');
                };
                reader.readAsDataURL(file);
            }
        }

        async function analyzeImage() {
            const file = imageInput.files[0];
            if (!file) return;

            // Show loading state
            analyzeText.textContent = '🤖 Analyzing with Vertex AI...';
            loadingSpinner.classList.remove('hidden');
            analyzeBtn.disabled = true;
            results.classList.add('hidden');
            errorSection.classList.add('hidden');

            const formData = new FormData();
            formData.append('image', file);

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.error) {
                    showError(result.error);
                } else {
                    showResults(result);
                }
            } catch (error) {
                showError('Failed to analyze image. Please check your internet connection and try again.');
            } finally {
                // Reset button state
                analyzeText.textContent = '🔍 Analyze with Vertex AI';
                loadingSpinner.classList.add('hidden');
                analyzeBtn.disabled = false;
            }
        }

        function showResults(result) {
            const isEmergency = result.is_emergency;
            const confidence = Math.round(result.confidence * 100);

            const resultsContent = document.getElementById('resultsContent');

            const emergencyIcon = isEmergency ? '🚨' : '🚗';
            const emergencyClass = isEmergency ? 'text-red-700 bg-red-100' : 'text-green-700 bg-green-100';
            const emergencyText = isEmergency ? 'EMERGENCY VEHICLE DETECTED' : 'REGULAR VEHICLE';

            resultsContent.innerHTML = `
                <div class="space-y-4">
                    <!-- Main Result -->
                    <div class="${emergencyClass} p-4 rounded-lg border-2 ${isEmergency ? 'border-red-200' : 'border-green-200'}">
                        <div class="flex items-center justify-center text-center">
                            <span class="text-3xl mr-3">${emergencyIcon}</span>
                            <div>
                                <h4 class="text-xl font-bold">${emergencyText}</h4>
                                <p class="text-sm">Confidence: ${confidence}%</p>
                            </div>
                        </div>
                    </div>

                    <!-- Details -->
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="bg-white p-4 rounded-lg border">
                            <h5 class="font-semibold text-gray-800 mb-2">🎯 Detection Details</h5>
                            <p class="text-sm text-gray-600 mb-1"><strong>Vehicle Type:</strong> ${result.vehicle_type || 'Unknown'}</p>
                            <p class="text-sm text-gray-600 mb-1"><strong>Method:</strong> ${result.method || 'vertex_ai_gemini'}</p>
                            <p class="text-sm text-gray-600"><strong>Emergency:</strong> ${isEmergency ? 'Yes' : 'No'}</p>
                        </div>

                        <div class="bg-white p-4 rounded-lg border">
                            <h5 class="font-semibold text-gray-800 mb-2">📊 Confidence Scores</h5>
                            <div class="space-y-2">
                                <div class="flex justify-between items-center">
                                    <span class="text-sm text-gray-600">Regular:</span>
                                    <div class="flex items-center">
                                        <div class="w-20 bg-gray-200 rounded-full h-2 mr-2">
                                            <div class="bg-gray-500 h-2 rounded-full" style="width: ${Math.round(result.probabilities.regular * 100)}%"></div>
                                        </div>
                                        <span class="text-sm font-medium">${Math.round(result.probabilities.regular * 100)}%</span>
                                    </div>
                                </div>
                                <div class="flex justify-between items-center">
                                    <span class="text-sm text-gray-600">Emergency:</span>
                                    <div class="flex items-center">
                                        <div class="w-20 bg-gray-200 rounded-full h-2 mr-2">
                                            <div class="bg-red-500 h-2 rounded-full" style="width: ${Math.round(result.probabilities.emergency * 100)}%"></div>
                                        </div>
                                        <span class="text-sm font-medium">${Math.round(result.probabilities.emergency * 100)}%</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- AI Reasoning -->
                    ${result.reasoning ? `
                    <div class="bg-blue-50 p-4 rounded-lg border border-blue-200">
                        <h5 class="font-semibold text-blue-800 mb-2">🤖 AI Reasoning</h5>
                        <p class="text-sm text-blue-700">${result.reasoning}</p>
                    </div>
                    ` : ''}
                </div>
            `;

            results.classList.remove('hidden');
        }

        function showError(message) {
            document.getElementById('errorMessage').textContent = message;
            errorSection.classList.remove('hidden');
        }
    </script>
</body>
</html>
    ''')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        # Check if AI detector is available
        if detector is None:
            return jsonify({'error': f'AI detector not available: {detector_status}'}), 500

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

        # Make prediction using AI detector
        result = detector.predict_from_pil(image, filename=file.filename)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'detector_status': detector_status,
        'detector_type': detector_type,
        'ai_available': detector is not None,
        'project_id': os.getenv('GAPI_PROJECT_ID', 'Not configured') if detector_type == 'vertex_ai' else 'N/A',
        'detection_method': detector_type
    })

if __name__ == '__main__':
    print("🚀 Starting AI Emergency Vehicle Detection Server...")
    print(f"📡 Detector Status: {detector_status}")
    print(f"🤖 Detector Type: {detector_type}")
    if detector and detector_type == 'vertex_ai' and hasattr(detector, 'project_id'):
        print(f"🌟 Project ID: {detector.project_id}")
        print(f"🌍 Location: {detector.location}")
    elif detector and detector_type == 'simple_ai':
        print(f"🔧 Method: Computer Vision + Color Analysis")
    print("\n🌐 Available endpoints:")
    print("  - Main Interface: http://localhost:8005/")
    print("  - Health Check: http://localhost:8005/health")
    print("\n⚡ Features:")
    print(f"  - Google {detector_type.replace('_', ' ').title()} Gemini Vision")
    print("  - Real-time emergency vehicle detection")
    print("  - Detailed AI reasoning and confidence scores")
    print("  - Enterprise-grade cloud processing")
    app.run(debug=True, host='0.0.0.0', port=8005)