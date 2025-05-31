import cv2
import depthai as dai
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

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

# 데이터 불러오기
data = np.genfromtxt('gesture_train.csv', delimiter=',')
x_data = data[:, :-1]
y_data = data[:, -1].astype(int)

# 원-핫 인코딩
num_classes = len(np.unique(y_data))
y_data = to_categorical(y_data, num_classes)

# 학습/검증 분리
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42)

# 모델 구성
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(15,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 컴파일 및 학습
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

# 모델 저장
model.save('gesture_dnn_model.h5')