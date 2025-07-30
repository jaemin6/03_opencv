import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Webcam', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):   # 스페이스바를 누르면 저장
        cv2.imwrite('line_tape.jpg', frame)
        print('사진 저장됨: line_tape.jpg')
    elif key == ord('q'): # q 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
