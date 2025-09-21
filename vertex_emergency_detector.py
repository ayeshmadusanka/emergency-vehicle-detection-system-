import os
import base64
import json
from dotenv import load_dotenv
from google.cloud import aiplatform
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from PIL import Image
import io

# Load environment variables
load_dotenv()

class VertexEmergencyDetector:
    """
    Emergency vehicle detector using Google Vertex AI Gemini Vision
    """

    def __init__(self):
        self.project_id = os.getenv('GAPI_PROJECT_ID')
        self.location = 'us-central1'  # Default Vertex AI location

        # Set up service account authentication
        self._setup_authentication()

        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.location)

        # Initialize the Gemini model (using 2.5-flash for better availability)
        self.model = GenerativeModel("gemini-2.5-flash")

        print(f"✓ Vertex AI Emergency Detector initialized")
        print(f"  Project: {self.project_id}")
        print(f"  Location: {self.location}")

        # Initialize credentials file path
        self._credentials_file = None

    def _setup_authentication(self):
        """Set up Google Cloud authentication using service account from .env"""
        try:
            # Validate required environment variables
            required_vars = ['TYPE', 'GAPI_PROJECT_ID', 'PRIVATE_KEY_ID', 'GAPI_PRIVATE_KEY', 'GAPI_CLIENT_EMAIL', 'GAPI_TOKEN_URI']
            missing_vars = [var for var in required_vars if not os.getenv(var)]

            if missing_vars:
                raise ValueError(f"Missing required environment variables: {missing_vars}")

            # Create service account info from environment variables
            service_account_info = {
                "type": os.getenv('TYPE'),
                "project_id": os.getenv('GAPI_PROJECT_ID'),
                "private_key_id": os.getenv('PRIVATE_KEY_ID'),
                "private_key": os.getenv('GAPI_PRIVATE_KEY').replace('\\n', '\n'),
                "client_email": os.getenv('GAPI_CLIENT_EMAIL'),
                "client_id": "",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": os.getenv('GAPI_TOKEN_URI'),
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{os.getenv('GAPI_CLIENT_EMAIL')}"
            }

            # Create credentials
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )

            # Set up authentication for Vertex AI using temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(service_account_info, f)
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f.name
                self._credentials_file = f.name

            print(f"✓ Authentication configured for project: {self.project_id}")

        except Exception as e:
            print(f"Warning: Could not set up authentication: {e}")
            print("Make sure your .env file has the correct Google Cloud credentials")
            raise

    def predict_from_pil(self, pil_image, filename=None):
        """
        Predict if PIL image contains emergency vehicle using Vertex AI
        """
        try:
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            # Create image part for Gemini
            image_part = Part.from_data(
                mime_type="image/jpeg",
                data=img_byte_arr
            )

            # Create prompt for emergency vehicle detection
            prompt = """
            Analyze this image and determine if it contains an emergency vehicle (ambulance, fire truck, police car, or rescue vehicle).

            Please respond with a JSON object in this exact format:
            {
                "is_emergency": true/false,
                "vehicle_type": "ambulance/fire_truck/police_car/regular_vehicle",
                "confidence": 0.0-1.0,
                "reasoning": "brief explanation of why this is/isn't an emergency vehicle"
            }

            Look for:
            - Red and white ambulance with medical symbols
            - Red fire trucks with ladders/hoses
            - Police cars with distinctive markings
            - Emergency lights, sirens, or official markings
            - Emergency service text/logos

            Be conservative - only classify as emergency if you're confident it's an official emergency vehicle.
            """

            # Generate content using Gemini
            response = self.model.generate_content([prompt, image_part])

            # Parse the response
            response_text = response.text.strip()

            # Try to extract JSON from response
            try:
                # Look for JSON in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1

                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")

                # Validate and format the result
                is_emergency = result.get('is_emergency', False)
                confidence = float(result.get('confidence', 0.5))
                vehicle_type = result.get('vehicle_type', 'regular_vehicle')
                reasoning = result.get('reasoning', 'AI analysis')

                # Map to our expected format
                prediction = "Emergency Vehicle (Ambulance/Fire Truck)" if is_emergency else "Regular Vehicle"

                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'is_emergency': is_emergency,
                    'vehicle_type': vehicle_type,
                    'reasoning': reasoning,
                    'probabilities': {
                        'regular': 1 - confidence if is_emergency else confidence,
                        'emergency': confidence if is_emergency else 1 - confidence
                    },
                    'method': 'vertex_ai_gemini'
                }

            except (json.JSONDecodeError, ValueError) as e:
                # Fallback: analyze response text for keywords
                response_lower = response_text.lower()
                is_emergency = any(word in response_lower for word in ['ambulance', 'fire truck', 'emergency', 'yes', 'true'])
                confidence = 0.7 if is_emergency else 0.8

                return {
                    'prediction': "Emergency Vehicle (Ambulance/Fire Truck)" if is_emergency else "Regular Vehicle",
                    'confidence': confidence,
                    'is_emergency': is_emergency,
                    'vehicle_type': 'unknown',
                    'reasoning': f'Fallback analysis: {response_text[:100]}...',
                    'probabilities': {
                        'regular': 1 - confidence if is_emergency else confidence,
                        'emergency': confidence if is_emergency else 1 - confidence
                    },
                    'method': 'vertex_ai_gemini_fallback'
                }

        except Exception as e:
            print(f"Vertex AI error: {e}")
            # Return fallback result
            return {
                'prediction': 'Regular Vehicle',
                'confidence': 0.5,
                'is_emergency': False,
                'vehicle_type': 'unknown',
                'reasoning': f'Error: {str(e)}',
                'probabilities': {
                    'regular': 0.5,
                    'emergency': 0.5
                },
                'method': 'error_fallback',
                'error': str(e)
            }

    def predict(self, image_path):
        """
        Predict if image file contains emergency vehicle
        """
        try:
            image = Image.open(image_path).convert('RGB')
            return self.predict_from_pil(image, filename=os.path.basename(image_path))
        except Exception as e:
            return {
                'error': str(e),
                'prediction': 'Error',
                'confidence': 0.0,
                'is_emergency': False
            }

    def cleanup(self):
        """Clean up temporary credentials file"""
        if hasattr(self, '_credentials_file') and self._credentials_file and os.path.exists(self._credentials_file):
            try:
                os.unlink(self._credentials_file)
                print("✓ Cleaned up temporary credentials file")
            except Exception as e:
                print(f"Warning: Could not clean up credentials file: {e}")

if __name__ == "__main__":
    # Test the Vertex AI detector
    try:
        detector = VertexEmergencyDetector()

        # Create a test image
        test_img = Image.new('RGB', (300, 200), color='red')

        print("\nTesting Vertex AI Emergency Vehicle Detection...")
        result = detector.predict_from_pil(test_img)

        print(f"Result: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Emergency: {result['is_emergency']}")
        print(f"Method: {result['method']}")
        if 'reasoning' in result:
            print(f"Reasoning: {result['reasoning']}")

    except Exception as e:
        print(f"Error testing Vertex AI detector: {e}")
        print("Make sure your .env file has valid Google Cloud credentials")