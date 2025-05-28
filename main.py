import cv2
import depthai as dai
import numpy as np
import mediapipe as mp

# gesture definition
my_gesture = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'k-heart', 7: 'fig', 8: 'racist', 9: 'heart', 10: 'fy'
}

# mediapipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# train gesture detection model
file = np.genfromtxt('gesture_train.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# DepthAI pipeline definition
pipeline = dai.Pipeline()
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(30)
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# device connection and image processing
with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name='rgb', maxSize=4, blocking=False)

    while True:
        in_rgb = q_rgb.get()
        img = in_rgb.getCvFrame()

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            my_result = []
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

                if idx in my_gesture:
                    # Mosaic
                    if idx in [7, 8, 10]:
                        x1, y1 = tuple((joint.min(axis=0)[:2] * [img.shape[1], img.shape[0]] * 0.95).astype(int))
                        x2, y2 = tuple((joint.max(axis=0)[:2] * [img.shape[1], img.shape[0]] * 1.05).astype(int))
                        x1, y1 = max(x1, 0), max(y1, 0)
                        x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])

                        if x2 > x1 and y2 > y1:
                            region = img[y1:y2, x1:x2].copy()
                            region = cv2.resize(region, dsize=None, fx=0.05, fy=0.05, interpolation=cv2.INTER_NEAREST)
                            region = cv2.resize(region, dsize=(x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                            img[y1:y2, x1:x2] = region
                    
                    # Print text with label
                    org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                    cv2.putText(img, text=my_gesture[idx].upper(), org=(org[0], org[1] + 20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                    my_result.append({'rps': my_gesture[idx], 'org': org})

        cv2.imshow('Game', img)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
