# Real-time Facial Emotion Recognition

A Flask-based web application that performs real-time facial emotion detection using a deep learning model.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Supported Emotions](#supported-emotions)
- [Error Handling](#error-handling)
- [Contributing](#contributing)

## Features

- Real-time face detection using OpenCV  
- Emotion classification into 7 categories  
- Live video streaming with emotion predictions  
- Web interface for easy interaction  
- Real-time confidence scores  

## Project Structure

```
livefacial/
│
├── app.py                      # Main application file
├── model.h5                     # Pre-trained emotion detection model
├── requirements.txt             # Project dependencies
│
├── static/                      # Static files directory
│   ├── css/                     # CSS stylesheets
│   │   └── style.css
│   └── js/                      # JavaScript files
│       └── main.js
│
├── templates/                   # HTML templates
│   └── index.html               # Main web interface
│
├── models/                      # Model-related files
│   └── haarcascade_frontalface_default.xml
│
└── README.md                    # Project documentation
```

## Prerequisites

- Python 3.x  
- Required Python packages:  
  - Flask  
  - OpenCV (cv2)  
  - TensorFlow  
  - NumPy  

## Installation

1. **Clone the repository**  
```bash
git clone <repository-url>
cd livefacial
```

2. **Create a virtual environment (recommended)**  
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**  
```bash
pip install -r requirements.txt
```

## Usage

1. **Start the application**  
```bash
python app.py
```

2. **Access the application**  
   - Open your web browser  
   - Navigate to `http://localhost:5000`  
   - Grant camera permissions  

3. **Stop the application**  
   - Click the stop button  
   - Or close the browser  

## Technical Details

### Model Architecture  
- **Input**: 48x48 grayscale images  
- **Output**: 7 emotion classes  
- **Framework**: TensorFlow/Keras  

### Processing Pipeline  
1. **Video Capture**  
   - Real-time webcam feed  
   - Frame-by-frame processing  

2. **Face Detection**  
   - Using Haar Cascade Classifier  
   - **Parameters:**  
     - Scale Factor: `1.1`  
     - Minimum Neighbors: `5`  
     - Minimum Size: `30x30`  

3. **Emotion Detection**  
   - **Preprocessing:**  
     - Grayscale conversion  
     - Resizing to `48x48`  
     - Pixel normalization  
   - **Real-time prediction**  
   - **Confidence score calculation**  

## Supported Emotions  

- 😠 **Angry**  
- 🤢 **Disgust**  
- 😨 **Fear**  
- 😃 **Happy**  
- 😐 **Neutral**  
- 😢 **Sad**  
- 😲 **Surprise**  

## Error Handling

- **Face detection errors** (no face detected)  
- **Model prediction errors** (invalid input)  
- **Camera resource management** (camera in use by another app)  
- **Graceful application shutdown** (handle user exit)  

## Contributing

If you'd like to contribute, please fork the repository and submit a pull request.  

---

### **📌 Next Steps**
1. **Save this as `README.md`** in your project folder.  
2. **Commit and push to GitHub** using:
   ```bash
   git add README.md
   git commit -m "Added README file"
   git push origin main
   ```

