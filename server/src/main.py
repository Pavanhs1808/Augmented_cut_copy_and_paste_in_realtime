import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory
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
import easyocr
import os
import win32clipboard
import pymeshlab
import requests
import sys
import shutil
import tempfile
import subprocess
import time
# remove: "import onnxruntime as ort" and the immediate InferenceSession creation
# keep a lazy loader so ONNX can be enabled later if you download the model
session = None

def load_onnx_model(path=None):
    """
    Lazy-load an ONNX model. Call load_onnx_model(path) when you have the model file.
    """
    try:
        import onnxruntime as ort
    except Exception:
        print("onnxruntime not installed; ONNX inference disabled.")
        return None

    model_path = path or os.environ.get("MODEL_ONNX", os.path.join(os.getcwd(), "model.onnx"))
    if os.path.exists(model_path):
        try:
            sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            print(f"ONNX model loaded: {model_path}")
            return sess
        except Exception as e:
            print(f"Failed to create ONNX InferenceSession: {e}")
            return None
    else:
        print(f"ONNX model not found at {model_path}; inference disabled.")
        return None

# trimesh is optional; if it's not installed we keep trimesh as None
# to allow the code to fall back to the Blender CLI conversion path.
try:
    import trimesh
    print("trimesh available")
except Exception:
    trimesh = None
    print("trimesh not available; install with: pip install trimesh pygltflib")

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

os.environ['EASYOCR_MODEL_STORAGE'] = os.path.join(os.getcwd(), 'easyocr_models')
reader = easyocr.Reader(['en'])

@app.route('/detect_and_ocr', methods=['POST'])
def detect_and_ocr():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        file = request.files['image']
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        pil_image = Image.open(io.BytesIO(image_bytes))

        # YOLO object detection
        detected_objects = detect_objects_yolo(image)

        # EasyOCR text detection
        results = reader.readtext(np.array(pil_image))
        detected_texts = [res[1] for res in results]
        detected_text = ' '.join(detected_texts) if detected_texts else None

        return jsonify({
            'objects': detected_objects,
            'text': detected_text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/paste_text', methods=['POST'])
def paste_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardText(text)
        win32clipboard.CloseClipboard()

        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/extract_frames', methods=['POST'])
def extract_frames():
    """Extract frames from video file, run COLMAP, convert to .glb, and upload to Lovable"""
    try:
        # Create unique timestamp-based directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        work_dir = f"workspace_{timestamp}"
        os.makedirs(work_dir, exist_ok=True)
        
        # Create subdirectories
        frames_dir = os.path.join(work_dir, "frames")
        sparse_dir = os.path.join(work_dir, "sparse")
        dense_dir = os.path.join(work_dir, "dense")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(sparse_dir, exist_ok=True)
        os.makedirs(dense_dir, exist_ok=True)

        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
            
        # Save video with timestamp
        file = request.files['video']
        video_path = os.path.join(work_dir, f'video_{timestamp}.mp4')
        with open(video_path, 'wb') as f:
            f.write(file.read())

        # Extract frames
        ffmpeg_cmd = f'ffmpeg -i {video_path} -vf "fps=2" {os.path.join(frames_dir, "frame_%04d.jpg")}'
        os.system(ffmpeg_cmd)

        # Run COLMAP pipeline with unique paths
        db_path = os.path.join(work_dir, f'database_{timestamp}.db')
        os.system(f'colmap feature_extractor --database_path {db_path} --image_path {frames_dir}')
        os.system(f'colmap exhaustive_matcher --database_path {db_path}')
        os.system(f'colmap mapper --database_path {db_path} --image_path {frames_dir} --output_path {sparse_dir}')
        os.system(f'colmap image_undistorter --image_path {frames_dir} --input_path {os.path.join(sparse_dir, "0")} --output_path {dense_dir} --output_type COLMAP')
        os.system(f'colmap patch_match_stereo --workspace_path {dense_dir} --workspace_format COLMAP --PatchMatchStereo.geom_consistency true')
        
        # Generate unique PLY and GLB paths
        ply_path = os.path.join(dense_dir, f'model_{timestamp}.ply')
        glb_path = os.path.join(dense_dir, f'model_{timestamp}.glb')
        
        os.system(f'colmap stereo_fusion --workspace_path {dense_dir} --workspace_format COLMAP --input_type geometric --output_path {ply_path}')
        
        frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]

        if os.path.exists(ply_path):
            convert_ply_to_glb(ply_path, glb_path)
        else:
            return jsonify({'error': '3D model (.ply) not generated'}), 500

        if os.path.exists(glb_path):
            response = send_model_to_lovable(glb_path, f"scan_{timestamp}")
            upload_status = response.status_code
            upload_resp = response.text
        else:
            return jsonify({'error': '3D model (.glb) not generated'}), 500

        return jsonify({
            'success': True,
            'workspace': work_dir,
            'frames': frame_files,
            'count': len(frame_files),
            'upload_status': upload_status,
            'upload_response': upload_resp,
            'model_paths': {
                'ply': ply_path,
                'glb': glb_path
            }
        })

    except Exception as e:
        print(f"Frame extraction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    """
    Receive a frame image from mobile and save to frames/ directory
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        filename = file.filename or 'frame.jpg'
        os.makedirs('frames', exist_ok=True)
        save_path = os.path.join('frames', filename)
        file.save(save_path)
        return jsonify({'success': True, 'filename': filename}), 200
    except Exception as e:
        print(f"Frame upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    save_path = os.path.join('frames', file.filename)
    file.save(save_path)
    return 'OK', 200

@app.route('/download_model', methods=['GET'])
def download_model():
    model_path = 'dense/fused.ply'  # or .obj if you prefer
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return jsonify({'error': 'Model not found'}), 404

@app.route('/models/<filename>')
def serve_model(filename):
    return send_from_directory('static/models', filename)

def convert_ply_to_glb(ply_path, glb_path):
    """Convert .ply -> .glb with detailed logging"""
    print(f"Starting PLY->GLB conversion:")
    print(f"Input PLY: {ply_path} ({os.path.exists(ply_path)})")
    print(f"Output GLB: {glb_path}")
    
    if os.path.exists(glb_path):
        print(f"Removing existing GLB file")
        try:
            os.remove(glb_path)
        except Exception as e:
            print(f"Failed to remove existing GLB: {e}")

    # Try trimesh
    if trimesh is not None:
        try:
            mesh = trimesh.load(ply_path, process=False)
            mesh.export(glb_path)
            if os.path.exists(glb_path):
                print(f"Successfully converted with trimesh: {os.path.getsize(glb_path)} bytes")
                return True
            else:
                print("trimesh export produced no file")
        except Exception as e:
            print(f"trimesh conversion failed: {e}")

    print("Conversion failed - no methods succeeded")
    return False

def send_model_to_lovable(model_path, name):
    """Upload model to Lovable with detailed logging and retries"""
    url = "https://hybrid-image-lab.lovable.app/api/upload-model/psmxw6kryh"
    
    # Verify file exists and is readable
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    file_size = os.path.getsize(model_path)
    print(f"Uploading {model_path} ({file_size} bytes) to Lovable...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(model_path, "rb") as fh:
                files = {"model": fh}
                data = {"name": name}
                response = requests.post(url, files=files, data=data, timeout=30)
                
                print(f"Lovable upload attempt {attempt + 1}:")
                print(f"Status: {response.status_code}")
                print(f"Response: {response.text[:500]}...")  # First 500 chars
                
                if response.ok:
                    return response
                else:
                    print(f"Upload failed with status {response.status_code}")
                    
        except Exception as e:
            print(f"Upload attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(2)  # Wait before retry
            continue
    
    raise Exception(f"Failed to upload after {max_retries} attempts")

@app.route('/upload_model', methods=['POST'])
def upload_model():
    """
    Accept .glb upload, save to static/models, analyze (trimesh) and optionally forward to Lovable.
    """
    try:
        if 'model' not in request.files:
            return jsonify({'error': 'No model file provided'}), 400

        file = request.files['model']
        filename = file.filename or f"model_{int(time.time())}.glb"
        os.makedirs('static/models', exist_ok=True)
        save_path = os.path.join('static/models', filename)
        file.save(save_path)

        info = {'filename': filename, 'size_bytes': os.path.getsize(save_path)}
        analysis = {}

        # quick trimesh analysis if available
        if trimesh is not None:
            try:
                mesh = trimesh.load(save_path, force='mesh')
                analysis['is_mesh'] = True
                analysis['vertices'] = int(mesh.vertices.shape[0]) if hasattr(mesh, 'vertices') else None
                analysis['faces'] = int(mesh.faces.shape[0]) if hasattr(mesh, 'faces') else None
                try:
                    analysis['bounds'] = mesh.bounds.tolist()
                except Exception:
                    analysis['bounds'] = None
            except Exception as e:
                analysis['error'] = str(e)
        else:
            analysis['warning'] = 'trimesh not installed'

        info['analysis'] = analysis

        # optional: forward to Lovable if client requests it
        forward = request.form.get('forward', 'false').lower() in ('1', 'true', 'yes')
        if forward:
            try:
                resp = send_model_to_lovable(save_path, filename)
                info['forward_status'] = resp.status_code
                info['forward_response'] = resp.text[:1000]
            except Exception as e:
                info['forward_error'] = str(e)

        return jsonify(info), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    load_yolo_model()
    app.run(host="0.0.0.0", port=5000)