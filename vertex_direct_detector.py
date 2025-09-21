import os
import base64
import json
import time
import requests
import jwt
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables
load_dotenv()

class VertexDirectDetector:
    """
    Emergency vehicle detector using Google Vertex AI Gemini 2.5 Flash via direct API calls
    Similar to the PHP implementation you provided
    """

    def __init__(self):
        self.project_id = os.getenv('GAPI_PROJECT_ID')
        self.location = 'us-central1'  # Default Vertex AI location

        # Set up authentication
        self._setup_authentication()

        # Get access token
        self.access_token = self._get_access_token()

        if not self.access_token:
            raise Exception("Failed to get access token")

        print(f"✓ Vertex AI Direct Emergency Detector initialized")
        print(f"  Project: {self.project_id}")
        print(f"  Location: {self.location}")
        print(f"  Model: gemini-2.5-flash")

    def _setup_authentication(self):
        """Set up Google Cloud authentication using service account from .env"""
        try:
            # Store service account details for JWT creation
            self.client_email = os.getenv('GAPI_CLIENT_EMAIL')
            self.private_key = os.getenv('GAPI_PRIVATE_KEY').replace('\\n', '\n')
            self.token_uri = os.getenv('GAPI_TOKEN_URI')

            if not all([self.client_email, self.private_key, self.token_uri]):
                raise Exception("Missing required environment variables")

            print(f"✓ Authentication configured for project: {self.project_id}")

        except Exception as e:
            print(f"Error: Could not set up authentication: {e}")
            raise

    def _get_access_token(self):
        """Get access token using JWT similar to PHP implementation"""
        try:
            # Create JWT payload
            now = int(time.time())
            payload = {
                'iss': self.client_email,
                'scope': 'https://www.googleapis.com/auth/cloud-platform',
                'aud': self.token_uri,
                'exp': now + 3600,
                'iat': now
            }

            # Create JWT
            token = jwt.encode(payload, self.private_key, algorithm='RS256')

            # Exchange JWT for access token
            data = {
                'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
                'assertion': token
            }

            response = requests.post(self.token_uri, data=data)

            if response.status_code == 200:
                token_data = response.json()
                return token_data.get('access_token')
            else:
                print(f"Token request failed: {response.status_code} {response.text}")
                return None

        except Exception as e:
            print(f"Error getting access token: {e}")
            return None

    def _image_to_base64(self, pil_image):
        """Convert PIL image to base64 string"""
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str

    def predict_from_pil(self, pil_image, filename=None):
        """
        Predict if PIL image contains emergency vehicle using Vertex AI Gemini 2.5 Flash
        """
        try:
            # Convert image to base64
            image_base64 = self._image_to_base64(pil_image)

            # Vertex AI endpoint for Gemini 2.5 Flash
            vertex_url = f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/publishers/google/models/gemini-2.5-flash:generateContent"

            # Create the payload similar to PHP version
            payload = {
                "system_instruction": {
                    "parts": [{
                        "text": "You are an expert AI system specialized in emergency vehicle detection. Analyze images to identify emergency vehicles with high accuracy."
                    }]
                },
                "contents": [{
                    "role": "user",
                    "parts": [
                        {
                            "text": """Analyze this image and determine if it contains an emergency vehicle (ambulance, fire truck, police car, or rescue vehicle).

Please respond with a JSON object in this exact format:
{
    "is_emergency": true/false,
    "vehicle_type": "ambulance/fire_truck/police_car/regular_vehicle",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of why this is/isn't an emergency vehicle"
}

Look for:
- Red and white ambulance with medical symbols or cross
- Red fire trucks with ladders, hoses, or fire department markings
- Police cars with distinctive markings, light bars, or official emblems
- Emergency lights, sirens, or official emergency service markings
- Emergency service text/logos (FIRE, POLICE, AMBULANCE, RESCUE, EMS)

Be conservative - only classify as emergency if you're confident it's an official emergency vehicle."""
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }]
            }

            # Make the API request
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }

            response = requests.post(vertex_url, json=payload, headers=headers)

            if response.status_code == 200:
                data = response.json()

                # Extract the text response
                if 'candidates' in data and data['candidates'] and 'content' in data['candidates'][0]:
                    response_text = data['candidates'][0]['content']['parts'][0]['text']

                    # Try to parse JSON from response
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
                        reasoning = result.get('reasoning', 'Gemini AI analysis')

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
                            'method': 'vertex_ai_direct_gemini_2.5'
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
                            'method': 'vertex_ai_direct_fallback'
                        }
                else:
                    raise Exception("Invalid response format from Gemini")

            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                print(f"Vertex AI API error: {error_msg}")
                raise Exception(error_msg)

        except Exception as e:
            print(f"Vertex AI Direct error: {e}")
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

if __name__ == "__main__":
    # Test the Vertex AI Direct detector
    try:
        detector = VertexDirectDetector()

        # Create a test image
        test_img = Image.new('RGB', (300, 200), color='red')

        print("\nTesting Vertex AI Direct Emergency Vehicle Detection...")
        result = detector.predict_from_pil(test_img)

        print(f"Result: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Emergency: {result['is_emergency']}")
        print(f"Method: {result['method']}")
        if 'reasoning' in result:
            print(f"Reasoning: {result['reasoning']}")

    except Exception as e:
        print(f"Error testing Vertex AI Direct detector: {e}")
        print("Make sure your .env file has valid Google Cloud credentials")