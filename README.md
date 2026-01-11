# AR Cut Paste Server

A backend server for the AR Cut Paste mobile app, enabling real-time object detection, OCR, background removal, 3D reconstruction, and cross-device clipboard synchronization.

## Features

- **Object Detection**: Uses YOLOv8 to detect objects in images with bounding boxes and confidence scores.
- **OCR**: Extracts text from images using EasyOCR.
- **Background Removal**: Removes backgrounds from selected objects using rembg, producing transparent PNGs.
- **3D Reconstruction**: Processes videos into 3D models using COLMAP and pymeshlab, generating GLB files.
- **Clipboard Sync**: Integrates with a separate clipboard server to sync text/images to PC clipboard.
- **API-Driven**: RESTful Flask APIs for seamless mobile app integration.

## Tech Stack

- **Backend**: Python, Flask, Flask-CORS
- **AI/ML**: Ultralytics YOLOv8, rembg, EasyOCR, OpenCV
- **3D Processing**: COLMAP, ffmpeg, pymeshlab
- **Database**: SQLite (for logs/models)
- **Other**: Pillow, requests, win32clipboard (for clipboard server)

## Project Structure

- `src/main.py`: Main Flask server with AI endpoints.
- `clipboard_server.py`: Separate server for PC clipboard integration.
- `requirements.txt`: Python dependencies.
- `src/database.db`: SQLite database.
- `src/yolov8n.pt`: YOLO model file.
- `bin/`: Binaries (ffmpeg, colmap).
- Temporary folders: `frames/`, `dense/`, `sparse/`, `workspace_*/` for 3D outputs.

## Setup

### Prerequisites
- Python 3.7+
- Node.js (for Expo mobile app)
- ffmpeg and COLMAP installed (place in `bin/` or system PATH)

### Installation
1. Clone the repository.
2. Create virtual environments:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```
3. For clipboard server:
   ```bash
   python -m venv clipboard-venv
   clipboard-venv\Scripts\activate
   pip install flask pillow pywin32  # Install deps if not in requirements
   ```

### Running the Servers
1. Main server:
   ```bash
   cd src
   python main.py
   ```
   Runs on `http://0.0.0.0:5000`.

2. Clipboard server:
   ```bash
   python clipboard_server.py
   ```
   Runs on `http://0.0.0.0:8000`.

## API Endpoints

### Main Server (Port 5000)
- `POST /detect_and_ocr`: Detect objects and extract text from image.
- `POST /extract`: Remove background from selected object.
- `POST /extract_frames`: Process video for 3D reconstruction.
- `GET /download_model`: Download generated 3D model.
- `POST /paste_text`: Send text to clipboard (deprecated; use clipboard server).

### Clipboard Server (Port 8000)
- `POST /paste`: Send image to PC clipboard.
- `POST /paste_text`: Send text to PC clipboard.

## Usage

1. Start servers as above.
2. Run the mobile app (see `cutpaste-app/` README).
3. Capture/pick media in app; sends to server for processing.
4. View results, extract objects, or sync to clipboard.

## Requirements

See `requirements.txt`. Key packages:
- Flask==2.3.0
- ultralytics
- rembg
- easyocr
- opencv-python
- pymeshlab

## Troubleshooting

- **ffmpeg/COLMAP not found**: Ensure binaries are in `bin/` or PATH. Download from official sites.
- **Permission errors**: Run as admin for clipboard access.
- **Model loading fails**: Check `yolov8n.pt` exists.
- **3D reconstruction errors**: Verify video format; check COLMAP logs.

## Contributing

1. Fork and clone.
2. Create feature branch.
3. Test changes.
4. Submit PR.

## License

[Add license if applicable]
