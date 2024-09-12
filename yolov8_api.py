from flask import Flask, request, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

# Load mô hình YOLOv8n
model = YOLO('yolov8n.pt')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return {'error': 'No image provided'}, 400

    file = request.files['image']
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Chạy mô hình YOLO
    results = model(image)

    # Vẽ bounding boxes lên ảnh
    boxes = results[0].boxes
    names = results[0].names
    
    color_map = {}

    for i in range(len(boxes.xyxy)):
        box = boxes.xyxy[i].numpy()
        x1,y1,x2,y2 = box.astype(int)
        id = boxes.cls[i].item()
        if id not in color_map:
            color = tuple(int(x) for x in np.random.randint(0,255,3))
            color_map[id] = color
        cv2.rectangle(image,(x1,y1),(x2,y2),color_map[id],2)
        conf = boxes.conf[i].item()
        cv2.putText(image,f'{conf:.2f}',(x1,y1+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,color_map[id],1)
        cv2.putText(image,f'{names[id]}',(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color_map[id],1)

    # Chuyển ảnh thành định dạng byte để gửi lại cho Flutter
    _, buffer = cv2.imencode('.png', image)
    io_buf = io.BytesIO(buffer)

    return send_file(io_buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
