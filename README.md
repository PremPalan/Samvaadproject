# Samvaad: Gujarati Sign Language Gesture Recognition

## Overview
Samvaad is a Python-based computer vision application designed to recognize and classify gestures from Gujarati Sign Language using a Random Forest classifier. It assists communication for the specially-abled community by translating hand gestures into machine-readable commands.

## Features
- Data collection scripts (`collect_imgs.py`) for building a custom gesture dataset.
- Data processing and feature extraction (`process_data.py`).
- Model training (`train_classifier.py`) and persistence (`model.p`).
- Inference module (`inference_classifier.py`) for real-time gesture recognition.

## Requirements
- Python 3.8+
- OpenCV
- scikit-learn
- numpy
- pandas

Install dependencies:
```bash
pip install -r requirements.txt

