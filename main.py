import cv2
from ultralytics import YOLO

# 1. 모델 로드 (가장 가벼운 n 모델)
model = YOLO('yolov8n.pt') 

# 2. 웹캠 연결 (0번은 기본 내장 카메라)
cap = cv2.VideoCapture(0)

# 카메라가 정상적으로 열렸는지 확인
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("실시간 탐지를 시작합니다. 'q'를 누르면 종료합니다.")

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 3. YOLOv8 추론 (사람 클래스인 0번만 탐지)
    # 딸기 질병 모델을 쓰실 땐 classes=[0]을 지우고 model = YOLO('best.pt')로 바꾸세요.
    results = model.predict(frame, classes=[0], conf=0.5, verbose=False)

    # 4. 결과 시각화 (프레임에 박스 그리기)
    annotated_frame = results[0].plot()

    # 5. 화면에 출력
    cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()