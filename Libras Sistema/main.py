import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np

# Inicialização
cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'O', 'Q', 'R', 'S', 'T', 'U', 'W', 'X', 'Y', 'Z']

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

while True:
    success, img = cap.read()
    if not success:
        break
    
    frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    handsPoints = results.multi_hand_landmarks
    h, w, _ = img.shape

    if handsPoints:
        for hand in handsPoints:
            x_max, y_max = 0, 0
            x_min, y_min = w, h
            for lm in hand.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_max, y_max = max(x, x_max), max(y, y_max)
                x_min, y_min = min(x, x_min), min(y, y_min)
            
            x_min, y_min = max(0, x_min - 50), max(0, y_min - 50)
            x_max, y_max = min(w, x_max + 50), min(h, y_max + 50)
            
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            try:
                imgCrop = img[y_min:y_max, x_min:x_max]
                imgCrop = cv2.resize(imgCrop, (224, 224))
                imgArray = np.asarray(imgCrop)
                normalized_image_array = (imgArray.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array
                prediction = model.predict(data)
                indexVal = np.argmax(prediction)
                cv2.putText(img, classes[indexVal], (x_min, y_min - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    cv2.imshow('Imagem', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
