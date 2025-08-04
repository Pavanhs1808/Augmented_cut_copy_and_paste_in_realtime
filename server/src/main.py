import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageFilter
import io
import base64
import argparse
from ultralytics import YOLO
import torch
from rembg import remove
import requests
from scipy import ndimage
from skimage import morphology, measure
import json

app = Flask(__name__)
CORS(app)

# Global variables
model = None
basnet_service_ip = None
basnet_service_host = None
photoshop_password = None

# Load YOLOv8 model
def load_yolo_model():
    global model
    try:
        # Try to load custom model first, fallback to pretrained
        model_path = "yolov8n.pt"
        if os.path.exists(model_path):
            model = YOLO(model_path)
        else:
            # Download and use pretrained model
            model = YOLO('yolov8n.pt')
        print("YOLOv8 model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        model = None

# Enhanced object detection with better filtering
def detect_objects_yolo(image_array):
    """
    Detect objects using YOLOv8 with improved filtering for better accuracy
    """
    if model is None:
        return []
    
    try:
        # Run inference
        results = model(image_array, conf=0.25, iou=0.45)
        
        detected_objects = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    # Filter out low-quality detections and unwanted classes
                    if confidence > 0.3 and is_valid_object(class_name, confidence):
                        # Calculate area to filter out very small objects
                        area = (x2 - x1) * (y2 - y1)
                        image_area = image_array.shape[0] * image_array.shape[1]
                        relative_area = area / image_area
                        
                        # Only include objects that are reasonably sized
                        if relative_area > 0.01:  # At least 1% of image area
                            detected_objects.append({
                                'class': class_name,
                                'confidence': float(confidence),
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'area': float(area),
                                'relative_area': float(relative_area)
                            })
        
        # Sort by confidence and area
        detected_objects.sort(key=lambda x: (x['confidence'] * x['relative_area']), reverse=True)
        
        # Limit to top 10 objects
        return detected_objects[:10]
        
    except Exception as e:
        print(f"Error in object detection: {e}")
        return []

def is_valid_object(class_name, confidence):
    """
    Filter out unwanted object classes
    """
    excluded_classes = {
        'person', 'dining table', 'chair', 'couch', 'bed', 'toilet', 
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone'
    }
    
    preferred_classes = {
        'bottle', 'cup', 'bowl', 'banana', 'apple', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
        'wine glass', 'fork', 'knife', 'spoon'
    }
    
    if class_name in preferred_classes:
        return confidence > 0.25
    
    if class_name in excluded_classes:
        return confidence > 0.7
    
    return confidence > 0.4

def extract_object_advanced(image_array, bbox, class_name):
    """
    Advanced object extraction with multiple techniques
    """
    try:
        x1, y1, x2, y2 = bbox
        
        # Crop the object region with padding
        padding = 20
        h, w = image_array.shape[:2]
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)
        
        cropped = image_array[y1_pad:y2_pad, x1_pad:x2_pad]
        
        if cropped.size == 0:
            return None
        
        # Convert to PIL Image
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        
        # Try rembg for automatic background removal
        try:
            extracted = remove(cropped_pil)
            if extracted is not None:
                return extracted
        except Exception as e:
            print(f"Rembg failed: {e}")
        
        # Fallback: Use GrabCut algorithm
        try:
            extracted = extract_with_grabcut(cropped, bbox, x1_pad, y1_pad)
            if extracted is not None:
                return extracted
        except Exception as e:
            print(f"GrabCut failed: {e}")
        
        # Final fallback: Return cropped image with transparent background
        return add_transparent_background(cropped_pil)
        
    except Exception as e:
        print(f"Error in object extraction: {e}")
        return None

def extract_with_grabcut(image, original_bbox, offset_x, offset_y):
    """
    Extract object using OpenCV's GrabCut algorithm
    """
    try:
        x1, y1, x2, y2 = original_bbox
        rect_x = max(0, x1 - offset_x)
        rect_y = max(0, y1 - offset_y)
        rect_w = min(image.shape[1] - rect_x, x2 - x1)
        rect_h = min(image.shape[0] - rect_y, y2 - y1)
        
        if rect_w <= 0 or rect_h <= 0:
            return None
        
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        rect = (rect_x, rect_y, rect_w, rect_h)
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
        
        result = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        result[:, :, 3] = mask2 * 255
        
        return Image.fromarray(result)
        
    except Exception as e:
        print(f"GrabCut extraction error: {e}")
        return None

def add_transparent_background(image_pil):
    """
    Add transparent background to image (fallback method)
    """
    try:
        if image_pil.mode != 'RGBA':
            image_pil = image_pil.convert('RGBA')
        
        data = np.array(image_pil)
        
        # Simple background removal based on corner colors
        corners = [
            data[0, 0],  # top-left
            data[0, -1],  # top-right
            data[-1, 0],  # bottom-left
            data[-1, -1]  # bottom-right
        ]
        
        corner_color = corners[0][:3]
        
        # Create mask for similar colors
        tolerance = 30
        mask = np.all(np.abs(data[:, :, :3] - corner_color) < tolerance, axis=2)
        
        # Apply mask to alpha channel
        data[mask, 3] = 0
        
        return Image.fromarray(data)
        
    except Exception as e:
        print(f"Transparent background error: {e}")
        return image_pil

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        img_bytes = file.read()
        npimg = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Run YOLO detection
        results = model(image)
        objects = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                objects.append({
                    "bbox": [x1, y1, x2, y2],
                    "class": model.names[cls],
                    "confidence": conf
                })

        return jsonify({"objects": objects})
    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({"error": str(e)}), 500
@app.route('/extract', methods=['POST'])
def extract_object():
    """
    Endpoint for object extraction
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get bounding box and other parameters
        bbox_str = request.form.get('bbox')
        class_name = request.form.get('class', 'object')
        confidence = float(request.form.get('confidence', 0.5))
        
        if not bbox_str:
            return jsonify({'error': 'No bounding box provided'}), 400
        
        try:
            bbox_data = json.loads(bbox_str)
            bbox = [bbox_data['x1'], bbox_data['y1'], bbox_data['x2'], bbox_data['y2']]
        except:
            return jsonify({'error': 'Invalid bounding box format'}), 400
        
        # Read and process the image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Extract object
        extracted_object = extract_object_advanced(image, bbox, class_name)
        
        if extracted_object is None:
            return jsonify({'error': 'Object extraction failed'}), 500
        
        # Convert result to byte array for sending in response
        img_byte_arr = io.BytesIO()
        extracted_object.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return send_file(img_byte_arr, mimetype='image/png')
        
    except Exception as e:
        print(f"Extraction endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect_and_extract', methods=['POST'])
def detect_and_extract():
    """
    Endpoint for detecting and extracting objects in one go
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read and process the image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Detect objects
        detected_objects = detect_objects_yolo(image)
        
        extracted_objects = []
        
        # Extract each detected object
        for obj in detected_objects:
            bbox = obj['bbox']
            class_name = obj['class']
            
            extracted_object = extract_object_advanced(image, bbox, class_name)
            
            if extracted_object is not None:
                # Convert result to byte array
                img_byte_arr = io.BytesIO()
                extracted_object.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                extracted_objects.append({
                    'class': class_name,
                    'confidence': obj['confidence'],
                    'image': base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                })
        
        return jsonify({
            'success': True,
            'objects': detected_objects,
            'extracted': extracted_objects,
            'count': len(detected_objects)
        })
        
    except Exception as e:
        print(f"Detection and extraction endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect', methods=['POST'])
def api_detect():
    """
    API endpoint for object detection (v2)
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read and process the image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Detect objects
        detected_objects = detect_objects_yolo(image)
        
        # Prepare response data
        response_data = []
        for obj in detected_objects:
            response_data.append({
                'class': obj['class'],
                'confidence': obj['confidence'],
                'bbox': obj['bbox']
            })
        
        return jsonify({
            'success': True,
            'objects': response_data,
            'count': len(response_data)
        })
        
    except Exception as e:
        print(f"API detection error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/extract', methods=['POST'])
def api_extract():
    """
    API endpoint for object extraction (v2)
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get bounding box and other parameters
        bbox_str = request.form.get('bbox')
        class_name = request.form.get('class', 'object')
        confidence = float(request.form.get('confidence', 0.5))
        
        if not bbox_str:
            return jsonify({'error': 'No bounding box provided'}), 400
        
        try:
            bbox_data = json.loads(bbox_str)
            bbox = [bbox_data['x1'], bbox_data['y1'], bbox_data['x2'], bbox_data['y2']]
        except:
            return jsonify({'error': 'Invalid bounding box format'}), 400
        
        # Read and process the image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Extract object
        extracted_object = extract_object_advanced(image, bbox, class_name)
        
        if extracted_object is None:
            return jsonify({'error': 'Object extraction failed'}), 500
        
        # Convert result to byte array for sending in response
        img_byte_arr = io.BytesIO()
        extracted_object.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return send_file(img_byte_arr, mimetype='image/png')
        
    except Exception as e:
        print(f"Extraction API error: {e}")
        return jsonify({'error': str(e)}), 500

def register_basnet_service(host, port, password):
    """
    Register the BASNet service for background removal
    """
    global basnet_service_host, basnet_service_ip, photoshop_password
    
    basnet_service_host = host
    basnet_service_ip = f"http://{host}:{port}"
    photoshop_password = password
    
    print(f"BASNet service registered: {basnet_service_ip}")

def remove_background_basnet(image_pil):
    """
    Remove background using BASNet service
    """
    global basnet_service_ip
    
    try:
        if image_pil.mode != 'RGBA':
            image_pil = image_pil.convert('RGBA')
        
        data = np.array(image_pil)
        
        # Prepare image for BASNet (resize, normalize, etc.)
        # Note: BASNet may have specific requirements for input image size and format
        image_bgr = cv2.cvtColor(data, cv2.COLOR_RGBA2BGRA)
        image_bgr = cv2.resize(image_bgr, (320, 320))  # Resize to 320x320 for BASNet
        
        # Convert to byte array
        _, img_encoded = cv2.imencode('.png', image_bgr)
        img_bytes = img_encoded.tobytes()
        
        # Call BASNet service
        response = requests.post(
            f"{basnet_service_ip}/remove_background",
            files={"image": img_bytes},
            data={"password": photoshop_password}
        )
        
        if response.status_code == 200:
            # Process BASNet response
            result = response.json()
            if 'image' in result:
                # Decode and return the image
                img_data = base64.b64decode(result['image'])
                img_array = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA))
        
        return None
    
    except Exception as e:
        print(f"BASNet background removal error: {e}")
        return None

@app.route('/remove_background', methods=['POST'])
def api_remove_background():
    """
    API endpoint for background removal using BASNet
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read and process the image
        image_bytes = file.read()
        image_pil = Image.open(io.BytesIO(image_bytes))
        
        # Remove background using BASNet
        result_image = remove_background_basnet(image_pil)
        
        if result_image is None:
            return jsonify({'error': 'Background removal failed'}), 500
        
        # Convert result to byte array for sending in response
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return send_file(img_byte_arr, mimetype='image/png')
        
    except Exception as e:
        print(f"Background removal API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/remove_background', methods=['POST'])
def api_remove_background_v2():
    """
    API endpoint for background removal using BASNet (v2)
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read and process the image
        image_bytes = file.read()
        image_pil = Image.open(io.BytesIO(image_bytes))
        
        # Remove background using BASNet
        result_image = remove_background_basnet(image_pil)
        
        if result_image is None:
            return jsonify({'error': 'Background removal failed'}), 500
        
        # Convert result to byte array for sending in response
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return send_file(img_byte_arr, mimetype='image/png')
        
    except Exception as e:
        print(f"Background removal API v2 error: {e}")
        return jsonify({'error': str(e)}), 500

def register_photoshop_service(host, port, password):
    """
    Register the Photoshop service for advanced editing
    """
    global photoshop_service_host, photoshop_service_ip, photoshop_password
    
    photoshop_service_host = host
    photoshop_service_ip = f"http://{host}:{port}"
    photoshop_password = password
    
    print(f"Photoshop service registered: {photoshop_service_ip}")

def edit_with_photoshop(image_pil, actions):
    """
    Edit image using Photoshop service
    """
    global photoshop_service_ip, photoshop_password
    
    try:
        # Convert image to byte array
        img_byte_arr = io.BytesIO()
        image_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Prepare payload for Photoshop actions
        files = {
            'image': img_byte_arr,
            'password': (None, photoshop_password)
        }
        
        # Add actions to payload
        for i, action in enumerate(actions):
            files[f'action_{i}'] = (None, action)
        
        # Call Photoshop service
        response = requests.post(
            f"{photoshop_service_ip}/edit",
            files=files
        )
        
        if response.status_code == 200:
            # Process Photoshop response
            result = response.json()
            if 'image' in result:
                # Decode and return the image
                img_data = base64.b64decode(result['image'])
                img_array = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA))
        
        return None
    
    except Exception as e:
        print(f"Photoshop editing error: {e}")
        return None

@app.route('/edit', methods=['POST'])
def api_edit():
    """
    API endpoint for image editing using Photoshop
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get actions from request
        actions = request.form.getlist('action')
        
        # Read and process the image
        image_bytes = file.read()
        image_pil = Image.open(io.BytesIO(image_bytes))
        
        # Edit image using Photoshop
        result_image = edit_with_photoshop(image_pil, actions)
        
        if result_image is None:
            return jsonify({'error': 'Image editing failed'}), 500
        
        # Convert result to byte array for sending in response
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return send_file(img_byte_arr, mimetype='image/png')
        
    except Exception as e:
        print(f"Image editing API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/edit', methods=['POST'])
def api_edit_v2():
    """
    API endpoint for image editing using Photoshop (v2)
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get actions from request
        actions = request.form.getlist('action')
        
        # Read and process the image
        image_bytes = file.read()
        image_pil = Image.open(io.BytesIO(image_bytes))
        
        # Edit image using Photoshop
        result_image = edit_with_photoshop(image_pil, actions)
        
        if result_image is None:
            return jsonify({'error': 'Image editing failed'}), 500
        
        # Convert result to byte array for sending in response
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return send_file(img_byte_arr, mimetype='image/png')
        
    except Exception as e:
        print(f"Image editing API v2 error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({'status': 'ok'}), 200

@app.route('/')
def index():
    return "AR CutPaste Server is running!"

# Load model at startup
load_yolo_model()

if __name__ == "__main__":
    load_yolo_model()
    app.run(host="0.0.0.0", port=5000)