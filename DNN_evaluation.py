import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import mediapipe as mp
import os

# 모델 불러오기
model = tf.keras.models.load_model("gesture_dnn_model.h5")

# MediaPipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# 클래스 정보
gesture_data_path = "gesture_images"
num_classes = 11
y_true, y_pred = [], []

# 예측 및 실제값 수집
for label in sorted(os.listdir(gesture_data_path), key=lambda x: int(x)):
    label_dir = os.path.join(gesture_data_path, label)
    for file_name in os.listdir(label_dir):
        file_path = os.path.join(label_dir, file_name)
        image = cv2.imread(file_path)
        if image is None:
            continue

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :]
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :]
                v = v2 - v1
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], :],
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :]
                ))
                angle = np.degrees(angle)
                data = np.array([angle], dtype=np.float32)
                pred = model.predict(data, verbose=0)
                pred_label = np.argmax(pred)

                y_true.append(int(label))
                y_pred.append(pred_label)
                break

# 혼동 행렬 출력
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(num_classes)])
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# 정확도 계산
accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
print(f"Accuracy: {accuracy:.4f}")
