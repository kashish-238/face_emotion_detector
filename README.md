# 🎭 Real-Time Facial Emotion Detection (Local AI)

A real-time facial emotion recognition application built using **OpenCV** and a **locally deployed Convolutional Neural Network (CNN)**.  
The system detects faces from a webcam feed and predicts emotions such as *Happy, Sad, Angry, Neutral,* etc., entirely **offline**.

This project focuses on building an efficient **computer vision pipeline**, rather than relying on cloud APIs or large language models.

---

## ✨ Features
- Real-time face detection using OpenCV (Haar Cascade)
- Emotion classification using a pretrained CNN (FER-2013)
- Fully **local inference** (no internet or API keys required)
- Prediction smoothing to reduce flickering
- Emotion timeline visualization (recent emotional trends)
- Automatic CSV logging of detected emotions with timestamps
- Privacy-friendly design (no screenshots included)

---

## 🧠 AI / ML Concepts Used
- Convolutional Neural Networks (CNNs)
- Image preprocessing (grayscale conversion, normalization)
- Real-time inference
- Majority-vote smoothing over prediction windows
- Computer vision pipelines

---

## 🛠️ Tech Stack
- **Python**
- **OpenCV**
- **TensorFlow / Keras**
- **NumPy**

---

## 📁 Project Structure
emotion-app/
│
├── app.py # Main application
├── emotion_model.h5 # Pretrained CNN model (FER-2013)
├── emotion_log.csv # Auto-generated emotion log
└── README.md


---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd emotion-app

2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate

3. Install dependencies
pip install tensorflow==2.13.0 opencv-python numpy

4. Run the application
python app.py

Press q to quit.

📊 Output

Live webcam feed with detected face and predicted emotion

On-screen emotion timeline (recent predictions)

emotion_log.csv containing:

timestamp

predicted emotion

confidence score