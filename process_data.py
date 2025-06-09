import cv2
import mediapipe as mp
import os
import numpy as np
from tqdm import tqdm

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Directory containing your dataset
DATA_DIR = './data'
PROCESSED_DIR = './processed_data'

if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

def preprocess_image(image):
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    # Get hand bbox
    h, w, _ = image.shape
    landmarks = results.multi_hand_landmarks[0].landmark
    x_min = min(landmark.x for landmark in landmarks)
    y_min = min(landmark.y for landmark in landmarks)
    x_max = max(landmark.x for landmark in landmarks)
    y_max = max(landmark.y for landmark in landmarks)
    
    # Add padding
    padding = 0.1
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(1, x_max + padding)
    y_max = min(1, y_max + padding)
    
    # Crop image
    x_min, y_min, x_max, y_max = [int(coord * dim) for coord, dim in zip([x_min, y_min, x_max, y_max], [w, h, w, h])]
    cropped_image = image[y_min:y_max, x_min:x_max]
    
    # Resize to a standard size
    resized_image = cv2.resize(cropped_image, (224, 224))
    
    # Enhance contrast
    lab = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_image = cv2.merge((cl,a,b))
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)
    
    return enhanced_image

# Process each image in the dataset
for folder in tqdm(os.listdir(DATA_DIR)):
    folder_path = os.path.join(DATA_DIR, folder)
    processed_folder_path = os.path.join(PROCESSED_DIR, folder)
    
    if not os.path.isdir(folder_path):
        continue
    
    if not os.path.exists(processed_folder_path):
        os.makedirs(processed_folder_path)
    
    print(f"Processing folder: {folder}")
    print(f"Number of images in {folder}: {len(os.listdir(folder_path))}")
    
    processed_count = 0
    
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        processed_image = preprocess_image(image)
        
        if processed_image is not None:
            cv2.imwrite(os.path.join(processed_folder_path, image_file), processed_image)
            processed_count += 1
        else:
            print(f"No hand detected in image: {image_path}")
    
    print(f"Processed {processed_count} images for folder {folder}")

print("Preprocessing completed.")