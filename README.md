# Real-Time Sentiment Analysis using OpenCV & Facial Emotion Recognition

A real-time sentiment analysis system that detects human emotions from live webcam video using **OpenCV**, **Deep Learning**, and **Facial Emotion Recognition** techniques.  
The project identifies emotions such as **Happy, Sad, Angry, Neutral, Surprise, Fear, and Disgust** in real time.

---

## Features

- Real-time webcam emotion detection
- Face detection using OpenCV
- Facial emotion recognition using trained deep learning model
- Live prediction display on video stream
- Simple and easy-to-run project structure
- Pre-trained model included

---

## Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- CNN (Convolutional Neural Network)

---

## Project Structure

```bash
sentimental-analysis/
│
├── src/
│   ├── video_stream.py
│   ├── emotion_model.py
│   ├── sentiment_mapping.py
│   ├── face_detection.py
│   ├── train_model.py
│   └── utils.py
│
├── models/
│   └── emotion_model.h5
│
├── dataset/fer2013
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/sentimental-analysis.git
```

Move into the project folder:

```bash
cd sentimental-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

To get the final output, run the following command inside the sentimental analysis folder:

```bash
python -m src.video_stream
```

Just copy and paste this command because everything has already been trained and configured. You do not need to run each file separately.

---

## Model Information

The model is trained using facial expression datasets and uses a CNN-based architecture for emotion classification.

### Detected Emotions

- Happy
- Sad
- Angry
- Neutral
- Surprise
- Fear
- Disgust

---

## Output

The system opens the webcam and displays:

- Face detection box
- Predicted emotion label
- Real-time emotion updates

---

## Future Improvements

- Improve model accuracy
- Add voice sentiment analysis
- Deploy as web application
- Support multiple face detection
- Add emotion statistics dashboard

---

## Author

Developed by **Chandhiramouli K**

B.E. Computer Science and Engineering (AI & ML)  
Annamalai University
