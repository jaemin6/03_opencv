import cv2

# 0번 웹캠 열기 (내장 카메라일 경우)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    cv2.imshow('Webcam View', frame)

    key = cv2.waitKey(1)
    if key == ord('s'):  # s 키 누르면 저장
        cv2.imwrite('../img/captured_line.png', frame)
        print("사진 저장 완료!")
    elif key == ord('q'):  # q 키 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
