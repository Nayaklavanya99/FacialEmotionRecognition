from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np

from tensorflow.keras.models import load_model
import threading

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model.h5')

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Initialize the camera
camera = cv2.VideoCapture(0)

# Add a flag to control the video stream
stream_active = True

def preprocess_face(face):
    # Convert to grayscale and resize to 48x48
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (48, 48))
    # Normalize
    face_normalized = face_resized / 255.0
    # Reshape for model input
    face_reshaped = face_normalized.reshape(1, 48, 48, 1)
    return face_reshaped

def generate_frames():
    global stream_active
    while stream_active:
        success, frame = camera.read()
        if not success:
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Process each face
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            try:
                # Preprocess face for emotion detection
                face_processed = preprocess_face(face_roi)
                
                # Predict emotion
                prediction = model.predict(face_processed)
                emotion_idx = np.argmax(prediction[0])
                emotion = EMOTIONS[emotion_idx]
                confidence = float(prediction[0][emotion_idx]) * 100

                # Draw rectangle around face
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                # Add emotion label
                label = f"{emotion} ({confidence:.1f}%)"
                cv2.putText(frame, label, (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            except Exception as e:
                print(f"Error processing face: {str(e)}")
                continue

        # Convert frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop')
def stop():
    global stream_active
    stream_active = False
    cleanup()
    return jsonify({"message": "Camera stopped. Closing application..."})

def cleanup():
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        cleanup()
