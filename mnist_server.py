from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import cv2
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the advanced model
MODEL_PATH = 'models/advanced_digit_recognizer.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None

def preprocess_image(image_data):
    """Advanced preprocessing for hand-drawn digits"""
    try:
        # Decode base64
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('L')
        
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Resize to 28x28
        img_resized = cv2.resize(img_gray, (28, 28))
        
        # Advanced preprocessing
        # 1. Gaussian blur to reduce noise
        img_blur = cv2.GaussianBlur(img_resized, (3, 3), 0)
        
        # 2. Adaptive thresholding
        img_thresh = cv2.adaptiveThreshold(
            img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 3. Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_clean = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        
        # 4. Find contours and center the digit
        contours, _ = cv2.findContours(img_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(contours[0])
            
            # Extract the digit
            digit = img_clean[y:y+h, x:x+w]
            
            # Create a centered 28x28 image
            centered = np.zeros((28, 28), dtype=np.uint8)
            
            # Calculate position to center the digit
            if w > 0 and h > 0:
                scale = min(20 / w, 20 / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Resize digit
                digit_resized = cv2.resize(digit, (new_w, new_h))
                
                # Center it
                x_offset = (28 - new_w) // 2
                y_offset = (28 - new_h) // 2
                
                centered[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized
        
        # Normalize to [0, 1]
        img_final = centered.astype(np.float32) / 255.0
        
        # Add channel dimension
        img_final = img_final.reshape(1, 28, 28, 1)
        
        return img_final
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess the image
        processed_image = preprocess_image(data['image'])
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                'digit': int(idx),
                'confidence': float(predictions[0][idx])
            }
            for idx in top_3_indices
        ]
        
        logger.info(f"Prediction: {predicted_digit} (confidence: {confidence:.4f})")
        
        return jsonify({
            'prediction': predicted_digit,
            'confidence': confidence,
            'top_3': top_3_predictions,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("🚀 Starting Advanced Digit Recognition Server...")
    print(f"📁 Model path: {MODEL_PATH}")
    print("🌐 Server will be available at http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)