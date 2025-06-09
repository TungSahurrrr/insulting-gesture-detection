# ✋ Insulting Gesture Detection – 실시간 폭력적 손동작 인식 및 모자이크 처리 시스템
 <img src="https://github.com/user-attachments/assets/e8ca636e-c5a6-44cb-8c7b-bfb9bb03ebe5" width="400" height="400"/>
 
## 📌 프로젝트 개요

이 프로젝트는 **실시간 영상 송출 방송** 또는 **CCTV 시스템** 등에서 사람의 손 제스처를 인식하여, **모욕적인 손동작(insulting gesture)** 을 자동 감지하고 **모자이크 처리**하는 시스템입니다.

이를 통해 **방송 사고나 부적절한 콘텐츠 노출을 사전에 방지**할 수 있습니다.

- 🎥 **실시간 웹캠 영상 처리**
- ✋ MediaPipe 기반 **손 랜드마크 추출**
- 🧠 **KNN, DNN 모델 기반 손동작 분류기**
- 🚫 **폭력적 제스처(예: middle fingers up)** 에 대해 자동 **모자이크 처리**
- ⚡ 개발 기간: **약 14일**

---

## 💡 기대 효과

- **방송 콘텐츠의 품질 향상 및 사고 예방**
- **자동화된 감시 시스템에 적용 가능**
- 다양한 **폭력/비폭력 제스처 확장 가능성**

---

## 🛠️ 주요 기술 스택

| 영역            | 사용 기술                     |
|-----------------|-------------------------------|
| 손 인식         | MediaPipe (Hands)             |
| 실시간 영상     | OpenCV                        |
| 제스처 분류     | CV2 (KNN, DNN)                |
| 전처리 특징     | 손가락 관절 간 각도 15개      |
| 모델 학습 파일  | CSV 기반 학습 데이터          |

---

## 🚀 실행 방법

### 1. 프로젝트 클론
```bash
git clone https://github.com/your-username/insulting-gesture-blur.git
cd insulting-gesture-blur
```

### 2. 필수 라이브러리 설치
```bash
pip install opencv-python mediapipe tensorflow numpy
```

### 3. 실행
```bash
python DNN_main.py
```
---

## 📈 향후 개선 방향
- 다양한 손동작 제스처 클래스 확대
- CNN/LSTM 기반 분류기로 정확도 향상
- 비디오 파일이나 스트리밍 플랫폼에 적용
- 음성 감정 분석 등 멀티모달 감지 연계

## 👨‍💻 개발자
- 이름: [KingHamster](https://github.com/KingHasmter), [Sirius912](https://github.com/Sirius912), 
- 개발 기간: 2025년 5월, 약 7일 소요
