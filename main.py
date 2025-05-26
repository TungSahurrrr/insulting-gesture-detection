# 출처 : https://velog.io/@dlth508/Toy-Project-%EA%B0%80%EC%9A%B4%EB%8D%B0-%EC%86%90%EA%B0%80%EB%9D%BD-%EB%AA%A8%EC%9E%90%EC%9D%B4%ED%81%AC-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EB%A7%8C%EB%93%A4%EA%B8%B0

import cv2
import mediapipe as mp
import numpy as np

# 1. 손 인식 개수 및 제스처 정의
max_num_hands = 1 # 최대 몇 개의 손을 인식할 건지 정의
my_gesture = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'six', 7: 'pig', 8: 'spiderman', 9: 'yeah', 10: 'ok', 11: 'fy'
}

# 2. Mediapipe hands model (모델 1 - 손 감지 모델)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils # 손 위에 그림을 그릴 수 있는 메소드
# hands 손가락 인식모듈의 작동 option을 지정한다. 
hands = mp_hands.Hands(
    max_num_hands=max_num_hands, # 인식할 손모양의 갯수, 생략하면 2가 지정된다.
    min_detection_confidence=0.5, # 성공적인 것으로 간주되는 최소 신뢰도 값. 0.0 ~1.0사이로서 기본값은 0.5이다.
    min_tracking_confidence=0.5) # 손 랜드마크가 성공적으로 추적된 것으로 간주되는 최소 신뢰도 값. 0.0 ~1.0 사이로서 기본값은 0.5이다. 이 값을 높이면 시간이 더 소요되지만 좀 더 정확한 작동이 보장된다. 
 
# 3. Gesture recognition model (모델 2 - 제스처 인식 모델)
file = np.genfromtxt('gesture_train.csv', delimiter=',') # data file(손가락의 각도들과 label 값 저장 파일)
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create() # K-Nearest Neighbors 알고리즘(K-최근접 알고리즘)
knn.train(angle, cv2.ml.ROW_SAMPLE, label) # 학습 시키기

# OpenCV로 웹캠을 읽어 입력데이터 소스로 지정한다. 
cap = cv2.VideoCapture(0) # USB 캠을 활용하고 싶다면 0 대신 1

while cap.isOpened():
    ret, img = cap.read() 	# 한 프레임씩 이미지 읽어옴
    if not ret:
        continue
    # 전처리 (opencv: BGR, mediapipe: RGB)
    img = cv2.flip(img, 1) 	# 이미지 좌우 반전
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 전처리 된 이미지
    result = hands.process(img) # 전처리 및 모델 추론을 함께 실행함
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # 이미지 출력을 위해 다시 바꿔줌
    
    if result.multi_hand_landmarks is not None: # 만약 손을 인식했다면
        my_result = []
        for res in result.multi_hand_landmarks: # 여러 개의 손을 인식할 수 있기 때문에 for문 사용
            joint = np.zeros((21, 3)) # joint -> 빨간점, 21개의 joint / 빨간점의 x, y, z 3개의 좌표이므로 3
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z] # 각 joint 마다 landmark 저장 (landmark의 x, y, z 좌표 저장)

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]  # 길이로 나눠줌 (크키 1짜리 vector 나오게 됨(unit vector))

            # Get angle using arcos of dot product (15개의 각도 구하기)
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
            
            # Convert radian to degree
            angle = np.degrees(angle) # angle이 radian 값으로 나오기 때문에 degree 값으로 바꿔줌

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            #---------------------------------------------------------------------------------
            # Draw gesture result
            if idx in my_gesture.keys():
                org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                cv2.putText(img, text=my_gesture[idx].upper(), org=(org[0], org[1] + 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            
                my_result.append({
                    'rps': my_gesture[idx],
                    'org': org
                })
            
                # 추가: 제스처가 7(pig) 또는 11(fy)일 때만 이미지 왜곡 효과 적용
                if idx == 7 or idx == 11:
                    x1, y1 = tuple((joint.min(axis=0)[:2] * [img.shape[1], img.shape[0]] * 0.95).astype(int))
                    x2, y2 = tuple((joint.max(axis=0)[:2] * [img.shape[1], img.shape[0]] * 1.05).astype(int))
            
                    # x1, y1, x2, y2 값이 이미지 범위를 벗어나지 않도록 제한
                    x1 = max(x1, 0)
                    y1 = max(y1, 0)
                    x2 = min(x2, img.shape[1])
                    y2 = min(y2, img.shape[0])
            
                    if x2 > x1 and y2 > y1:
                        mosaic_scale = 0.01
                        fy_img = img[y1:y2, x1:x2].copy()
                        fy_img = cv2.resize(fy_img, dsize=None, fx=0.05, fy=0.05, interpolation=cv2.INTER_NEAREST)
                        fy_img = cv2.resize(fy_img, dsize=(x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            
                        img[y1:y2, x1:x2] = fy_img


            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)


    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break