import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load the model
try:
    with open('./model.p', 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: model.p file not found. Make sure it's in the correct directory.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'ક', 1: 'ખ', 2: 'ગ', 3: 'ઘ', 4: 'ચ', 5: 'છ', 6: 'જ', 7: 'ઝ', 8: 'ટ', 9: 'ઠ',
    10: 'ડ', 11: 'ઢ', 12: 'ણ', 13: 'ત', 14: 'થ', 15: 'દ', 16: 'ધ', 17: 'ન', 18: 'પ', 19: 'ફ',
    20: 'બ', 21: 'ભ', 22: 'મ', 23: 'ય', 24: 'ર', 25: 'લ', 26: 'વ', 27: 'સ', 28: 'શ', 29: 'ષ',
    30: 'હ', 31: 'ળ', 32: 'ક્ષ', 33: 'જ્ઞ'
}


# Load your custom font (ensure you have a Gujarati font installed)
font_path = r'C:\SIH Hackathon\code\NotoSansGujarati-VariableFont_wdth,wght.ttf'  # Update this path to your font file
  # Update this path to your font file
font = ImageFont.truetype(font_path, 100)

# Function to put Gujarati text on the frame
def put_gujarati_text(img, text, position, font, color=(0, 0, 0)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

print("Starting video capture. Press 'q' to quit.")

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        try:
            # Ensure the data_aux has 84 features
            if len(data_aux) < 84:
                data_aux.extend([0] * (84 - len(data_aux)))
            elif len(data_aux) > 84:
                data_aux = data_aux[:84]

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Use the custom function to display Gujarati text
            frame = put_gujarati_text(frame, predicted_character, (x1, y1 - 10), font)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        except Exception as e:
            print(f"Error making prediction: {e}")

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Video capture ended.")
