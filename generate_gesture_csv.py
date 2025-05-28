import cv2
import mediapipe as mp
import numpy as np
import csv

# MediaPipe 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

file = open('gesture_train.csv', mode='a', newline='')
writer = csv.writer(file)

label = 7  # 저장할 제스처 번호 (예: 0 = fist)

print("데이터 수집 시작: 제스처", label)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :]))
            angle = np.degrees(angle)

            # Save to CSV
            data = angle.tolist()
            data.append(label)
            writer.writerow(data)

    cv2.imshow('Train', img)
    if cv2.waitKey(1) == ord('q'):
        break

file.close()
cap.release()
cv2.destroyAllWindows()
