import os
import cv2
import numpy as np
from PIL import Image
import io

class SimpleAIDetector:
    """
    Simple emergency vehicle detector using computer vision techniques
    """

    def __init__(self):
        print("✓ Simple AI Emergency Detector initialized")
        print("  Method: Computer Vision + Color Analysis")

    def predict_from_pil(self, pil_image, filename=None):
        """
        Predict if PIL image contains emergency vehicle using simple CV techniques
        """
        try:
            # Convert PIL to opencv format
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Analyze colors for emergency vehicle characteristics
            emergency_score = self._analyze_emergency_colors(opencv_image)

            # Analyze shapes for vehicle characteristics
            vehicle_score = self._analyze_vehicle_shapes(opencv_image)

            # Combine scores
            confidence = (emergency_score + vehicle_score) / 2
            is_emergency = confidence > 0.6

            # Determine vehicle type based on dominant colors
            vehicle_type = self._classify_vehicle_type(opencv_image, emergency_score)

            # Generate reasoning
            reasoning = self._generate_reasoning(emergency_score, vehicle_score, confidence)

            prediction = "Emergency Vehicle (Ambulance/Fire Truck)" if is_emergency else "Regular Vehicle"

            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'is_emergency': is_emergency,
                'vehicle_type': vehicle_type,
                'reasoning': reasoning,
                'probabilities': {
                    'regular': 1 - confidence if is_emergency else confidence,
                    'emergency': confidence if is_emergency else 1 - confidence
                },
                'method': 'simple_ai_cv'
            }

        except Exception as e:
            return {
                'prediction': 'Regular Vehicle',
                'confidence': 0.5,
                'is_emergency': False,
                'vehicle_type': 'unknown',
                'reasoning': f'Error in analysis: {str(e)}',
                'probabilities': {
                    'regular': 0.5,
                    'emergency': 0.5
                },
                'method': 'error_fallback',
                'error': str(e)
            }

    def _analyze_emergency_colors(self, image):
        """Analyze image for emergency vehicle colors (red, blue, white)"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Define color ranges for emergency vehicles
            # Red range (for fire trucks, ambulances)
            red_lower1 = np.array([0, 50, 50])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([170, 50, 50])
            red_upper2 = np.array([180, 255, 255])

            # Blue range (for police cars)
            blue_lower = np.array([100, 50, 50])
            blue_upper = np.array([130, 255, 255])

            # White range (common on emergency vehicles)
            white_lower = np.array([0, 0, 200])
            white_upper = np.array([180, 30, 255])

            # Create masks
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = red_mask1 + red_mask2

            blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
            white_mask = cv2.inRange(hsv, white_lower, white_upper)

            # Calculate color percentages
            total_pixels = image.shape[0] * image.shape[1]
            red_percent = np.sum(red_mask > 0) / total_pixels
            blue_percent = np.sum(blue_mask > 0) / total_pixels
            white_percent = np.sum(white_mask > 0) / total_pixels

            # Score based on emergency color presence
            emergency_color_score = 0
            if red_percent > 0.05:  # Significant red presence
                emergency_color_score += 0.4
            if blue_percent > 0.03:  # Blue lights/markings
                emergency_color_score += 0.3
            if white_percent > 0.2:  # White background
                emergency_color_score += 0.2
            if red_percent > 0.15:  # Dominant red (fire truck)
                emergency_color_score += 0.3

            return min(emergency_color_score, 1.0)

        except:
            return 0.3  # Default moderate score

    def _analyze_vehicle_shapes(self, image):
        """Analyze image for vehicle-like shapes"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            vehicle_score = 0

            # Look for rectangular shapes (typical of vehicles)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Significant size
                    # Approximate the contour
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    # Check if it's roughly rectangular
                    if len(approx) >= 4:
                        vehicle_score += 0.1

                    # Check aspect ratio
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    if 1.5 <= aspect_ratio <= 4:  # Typical vehicle proportions
                        vehicle_score += 0.2

            return min(vehicle_score, 1.0)

        except:
            return 0.5  # Default moderate score

    def _classify_vehicle_type(self, image, emergency_score):
        """Classify the type of vehicle based on color analysis"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Red analysis for fire trucks
            red_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
            red_mask += cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
            red_percent = np.sum(red_mask > 0) / (image.shape[0] * image.shape[1])

            # Blue analysis for police
            blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
            blue_percent = np.sum(blue_mask > 0) / (image.shape[0] * image.shape[1])

            if emergency_score > 0.6:
                if red_percent > 0.15:
                    return "fire_truck"
                elif blue_percent > 0.05:
                    return "police_car"
                elif red_percent > 0.05:
                    return "ambulance"
                else:
                    return "emergency_vehicle"
            else:
                return "regular_vehicle"

        except:
            return "regular_vehicle"

    def _generate_reasoning(self, emergency_score, vehicle_score, confidence):
        """Generate human-readable reasoning for the prediction"""
        try:
            reasoning_parts = []

            if emergency_score > 0.4:
                reasoning_parts.append("detected emergency colors (red/blue/white)")
            if vehicle_score > 0.5:
                reasoning_parts.append("identified vehicle-like shapes")
            if confidence > 0.7:
                reasoning_parts.append("high confidence in emergency classification")
            elif confidence > 0.4:
                reasoning_parts.append("moderate confidence in classification")
            else:
                reasoning_parts.append("low confidence, likely regular vehicle")

            if not reasoning_parts:
                reasoning_parts.append("standard vehicle appearance detected")

            return "Computer vision analysis " + ", ".join(reasoning_parts) + f" (confidence: {confidence:.1%})"

        except:
            return "Basic computer vision analysis completed"

    def predict(self, image_path):
        """Predict if image file contains emergency vehicle"""
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
    # Test the simple AI detector
    try:
        detector = SimpleAIDetector()

        # Create a test image with red color (simulating fire truck)
        test_img = Image.new('RGB', (300, 200), color=(200, 0, 0))  # Red image

        print("\nTesting Simple AI Emergency Vehicle Detection...")
        result = detector.predict_from_pil(test_img)

        print(f"Result: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Emergency: {result['is_emergency']}")
        print(f"Vehicle Type: {result['vehicle_type']}")
        print(f"Method: {result['method']}")
        print(f"Reasoning: {result['reasoning']}")

    except Exception as e:
        print(f"Error testing Simple AI detector: {e}")