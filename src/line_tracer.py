import cv2
import numpy as np

# 웹캠 열기 (기본 카메라: 0번)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame not captured")
        break

    # 1단계: 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2단계: 적응형 이진화
    binary_adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)

    # 컬러 프레임 복사 (중심점 표시용)
    img_color = frame.copy()

    # 3단계: 외곽선 찾기
    contours, _ = cv2.findContours(binary_adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tape_center_x, tape_center_y = -1, -1

    if contours:
        # 가장 큰 외곽선 선택
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M["m00"] != 0:
            tape_center_x = int(M["m10"] / M["m00"])
            tape_center_y = int(M["m01"] / M["m00"])

            # 중심점 표시
            cv2.circle(img_color, (tape_center_x, tape_center_y), 5, (0, 0, 255), -1)
            center_text = f"({tape_center_x}, {tape_center_y})"
            cv2.putText(img_color, center_text, (tape_center_x + 10, tape_center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ROI 사각형 표시
        x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(largest_contour)
        cv2.rectangle(img_color, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (0, 255, 0), 2)

        # 마스크 생성
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)

        # 마스크 적용 결과
        masked_tape = cv2.bitwise_and(gray, gray, mask=mask)
    else:
        masked_tape = gray.copy()

    # 이진화 AND 연산 결과
    bitwise_and_result = cv2.bitwise_and(gray, binary_adapt)

    # 출력 윈도우
    cv2.imshow("1. Original + Center", img_color)
    cv2.imshow("2. Adaptive Binary", binary_adapt)
    cv2.imshow("3. Masked Tape Area", masked_tape)
    cv2.imshow("4. Bitwise AND", bitwise_and_result)

    # 종료 키: ESC
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
