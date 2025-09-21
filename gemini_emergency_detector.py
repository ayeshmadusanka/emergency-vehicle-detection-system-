import os
import base64
import json
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import io

# Load environment variables
load_dotenv()

class GeminiEmergencyDetector:
    """
    Emergency vehicle detector using Google Gemini AI
    """

    def __init__(self):
        # For Google AI API (Gemini), we need an API key, not service account
        # You can get this from https://makersuite.google.com/app/apikey
        api_key = os.getenv('GOOGLE_AI_API_KEY')

        if not api_key:
            print("Warning: GOOGLE_AI_API_KEY not found in .env file")
            print("You can get an API key from: https://makersuite.google.com/app/apikey")
            print("Add it to your .env file as: GOOGLE_AI_API_KEY=your_api_key_here")
            raise ValueError("GOOGLE_AI_API_KEY is required")

        # Configure the API
        genai.configure(api_key=api_key)

        # Initialize the Gemini model
        self.model = genai.GenerativeModel('gemini-1.5-flash')

        print(f"✓ Gemini AI Emergency Detector initialized")
        print(f"  Model: gemini-1.5-flash")

    def predict_from_pil(self, pil_image, filename=None):
        """
        Predict if PIL image contains emergency vehicle using Gemini AI
        """
        try:
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
            response = self.model.generate_content([prompt, pil_image])

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
                    'method': 'gemini_ai'
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
                    'method': 'gemini_ai_fallback'
                }

        except Exception as e:
            print(f"Gemini AI error: {e}")
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
    # Test the Gemini AI detector
    try:
        detector = GeminiEmergencyDetector()

        # Create a test image
        test_img = Image.new('RGB', (300, 200), color='red')

        print("\nTesting Gemini AI Emergency Vehicle Detection...")
        result = detector.predict_from_pil(test_img)

        print(f"Result: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Emergency: {result['is_emergency']}")
        print(f"Method: {result['method']}")
        if 'reasoning' in result:
            print(f"Reasoning: {result['reasoning']}")

    except Exception as e:
        print(f"Error testing Gemini AI detector: {e}")
        print("Make sure your .env file has a valid GOOGLE_AI_API_KEY")