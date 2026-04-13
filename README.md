# 🎭 Face Emotion Detection using Deep Learning

## 📌 Problem Statement

The objective of this project is to develop a system that can automatically detect human emotions from facial expressions using deep learning techniques. The goal is to enable machines to understand and classify emotions such as Angry, Happy, Sad, Surprise, and Neutral from visual input.

---

## 🧠 Explanation

This project implements a real-time Face Emotion Detection system using a Convolutional Neural Network (CNN). The model is trained on facial image data to learn patterns and features associated with different emotions.

During execution, the system captures video input through a webcam using OpenCV. It detects the face region, preprocesses the image, and passes it to the trained CNN model. The model then predicts the emotion and displays the result on the screen in real time.

The project demonstrates the application of deep learning and computer vision techniques for emotion recognition in practical scenarios.

---

## 📂 Dataset

FER-2013 Facial Expression Dataset (Kaggle)

🔗 https://www.kaggle.com/datasets/pankaj4321/fer-2013-facial-expression-dataset

---

## ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib

---

## 🚀 How to Run

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Train the model

```bash
python src/train.py
```

3. Run live detection

```bash
python src/live_detection.py
```

---

## 📊 Results

* Validation Accuracy: ~67%
* Performs well under normal lighting conditions

---

## 🧠 Conclusion

This project successfully demonstrates how deep learning can be used to detect human emotions from facial expressions in real time. It provides a practical implementation of CNN for image classification and real-time computer vision.
