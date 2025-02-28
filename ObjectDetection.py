
import subprocess
import numpy as np
import tensorflow as tf
from flask import Flask, Response, render_template
import cv2
import time

app = Flask(__name__)

# check libcamera
def check_camera():
    try:
        result = subprocess.run(["libcamera-hello", "-t", "2000"], capture_output=True, text=True)
        if result.returncode == 0:
            print("Camera detected.")
            return True
        else:
            print("Camera not detected")
            return False
    except FileNotFoundError:
        print("libcamera-hello not found!")
        return False

if not check_camera():
    exit()

# Load labels
def load_labels(label_path):
    with open(label_path, "r") as f:
        return [line.strip() for line in f.readlines()]

class_names = load_labels("labels.txt")

# Load TFLite model
model_path = "detect.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# libcamera
def capture_frame():
    frame_path = "/dev/shm/frame.jpg"
    subprocess.run(["libcamera-still", "-o", frame_path, "-t", "1", "--width", "640", "--height", "480", "-n"], check=True)
    frame = cv2.imread(frame_path)
    return frame

# TensorFlow Lite model
def preprocess_frame(frame, input_shape):
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_uint8 = np.array(frame_rgb, dtype=np.uint8)
    return np.expand_dims(frame_uint8, axis=0)

# Stream
def generate():
    while True:
        frame = capture_frame()
        if frame is None:
            print("Failed to capture frame")
            continue

        input_shape = input_details[0]['shape']
        input_data = preprocess_frame(frame, input_shape)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        for i in range(len(scores)):
            if scores[i] > 0.5:
                box = boxes[i]
                class_id = int(classes[i])
                confidence = scores[i]

                if class_id >= len(class_names):
                    print(f"Class ID {class_id} is out of range (Max: {len(class_names)-1})")
                    continue

                y_min, x_min, y_max, x_max = (box * np.array([frame.shape[0], frame.shape[1], frame.shape[0], frame.shape[1]])).astype(int)

                label = f"{class_names[class_id]}: {confidence:.2f}"
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

        time.sleep(0.1)  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start Flask app
if __name__ == "__main__":
    print("Started, Open http://<your-pi-ip>:5000")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
