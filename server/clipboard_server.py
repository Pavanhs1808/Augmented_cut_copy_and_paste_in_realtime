from flask import Flask, request
import base64
from io import BytesIO
from PIL import Image
import win32clipboard
from win32con import CF_DIB, CF_UNICODETEXT

app = Flask(__name__)

def send_to_clipboard(clip_type, data):
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(clip_type, data)
    win32clipboard.CloseClipboard()

def image_to_clipboard(image):
    # Convert PIL Image to DIB format for clipboard
    output = BytesIO()
    image.convert('RGB').save(output, 'BMP')
    data = output.getvalue()[14:]  # BMP file header is 14 bytes, skip it
    output.close()
    send_to_clipboard(CF_DIB, data)

@app.route('/paste', methods=['POST'])
def paste_image():
    data = request.json
    if 'image' in data:
        img_data = base64.b64decode(data['image'])
        img = Image.open(BytesIO(img_data))
        image_to_clipboard(img)
        return 'Image pasted to clipboard'
    return 'No valid image data', 400

@app.route('/paste_text', methods=['POST'])
def paste_text():
    data = request.json
    if 'text' in data:
        send_to_clipboard(CF_UNICODETEXT, data['text'])
        return 'Text pasted to clipboard'
    return 'No valid text data', 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
