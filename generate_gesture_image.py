import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
import depthai as dai

# MediaPipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# DepthAI pipeline definition
pipeline = dai.Pipeline()
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(30)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# set image saving path
label = 10
save_dir = os.path.join('gesture_images', str(label))
os.makedirs(save_dir, exist_ok=True)

img_count = 0

print("Saving image started: gesture", label)

last_save_time = time.time()
save_interval = 1.0 

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        in_rgb = q_rgb.get()
        img = in_rgb.getCvFrame()

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            current_time = time.time()
            if current_time - last_save_time >= save_interval:
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

                    # Save image
                    image_name = f"{label}_{img_count:04d}.jpg"
                    image_path = os.path.join(save_dir, image_name)
                    cv2.imwrite(image_path, img)

                    img_count += 1
                    print(f"{img_count}/10 saved")

                    last_save_time = current_time

                    if img_count >= 10:
                        print("Saved 10 images")
                        file.close()
                        cv2.destroyAllWindows()
                        exit()

        cv2.imshow('Train', img)
        if cv2.waitKey(1) == ord('q'):
            break

file.close()
cv2.destroyAllWindows()
