import cv2
import numpy as np
import mediapipe as mp
import glob
import os
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter

# gesture definition
my_gesture = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'k-heart', 7: 'fig', 8: 'racist', 9: 'heart', 10: 'fy'
}

# mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

data = np.load('knn_train_data.npz')
angle = data['angle']
label = data['label']

knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

y_true = []
y_pred = []

for label_num in range(11):
    image_paths = glob.glob(f'gesture_images/{label_num}/*.jpg')

    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            print(f"cannot find: {image_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            preds = []
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
                ret, results, neighbours, dist = knn.findNearest(data, 3)
                idx = int(results[0][0])
                preds.append(idx)
        

            pred_majority = Counter(preds).most_common(1)[0][0]
        
            predicted = my_gesture.get(pred_majority, 'Unknown')
            actual = my_gesture.get(label_num, str(label_num))
        
            print(f"[{os.path.basename(image_path)}] actual: {actual:<8} | pred: {predicted}")
        
            y_true.append(label_num)
            y_pred.append(pred_majority)
        
            res = result.multi_hand_landmarks[0]
            org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
            cv2.putText(img, predicted.upper(), (org[0], org[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Result', img)
            key = cv2.waitKey(100)
            if key == ord('q'):
                break
        else:
            print(f"[{os.path.basename(image_path)}] cannot detect hand.")


cv2.destroyAllWindows()

acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.4f}")

#confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[my_gesture[i] for i in range(11)])
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45)
plt.show()
