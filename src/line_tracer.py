import cv2
import numpy as np

# 이미지 불러오기
img = cv2.imread('../img/line_tape.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: Image not found. Please check the path.")
    exit()

# 복사본
gray = img.copy()

# 바이너리로 이진화, 테이프 영역 분리
binary_adapt = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV, 11, 2)

# 컬러 입히기 + 중심점 표시
img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
# 외곽선 추출 (테이프 인식)
contours, _ = cv2.findContours(binary_adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 처음 중심 좌표표 -1로 시작
tape_center_x, tape_center_y = -1, -1 

if contours:
    # 가장 큰 외곽선 선택
    largest_contour = max(contours, key=cv2.contourArea)
    # 중심점 찾기 모멘트 함수로 중심점 계산
    M = cv2.moments(largest_contour)

    if M["m00"] != 0:
        tape_center_x = int(M["m10"] / M["m00"])
        tape_center_y = int(M["m01"] / M["m00"])
        # 중심점 빨간색 포인터로 표시
        cv2.circle(img_color, (tape_center_x, tape_center_y), 5, (0, 0, 255), -1)
        # 중심점 좌표 표기
        center_text = f"Center: ({tape_center_x}, {tape_center_y})"
        cv2.putText(img_color, center_text, (tape_center_x + 10, tape_center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # 관심영역 roi 설정, 추출
    x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(largest_contour)
    padding_roi = 20
    x1_padded = max(0, x_roi - padding_roi)
    y1_padded = max(0, y_roi - padding_roi)
    x2_padded = min(gray.shape[1], x_roi + w_roi + padding_roi)
    y2_padded = min(gray.shape[0], y_roi + h_roi + padding_roi)
    roi_extracted = gray[y1_padded:y2_padded, x1_padded:x2_padded]
    # roi 표기를 위한 포인터 그리기(사각형)
    cv2.rectangle(img_color, (x1_padded, y1_padded), (x2_padded, y2_padded), (0, 255, 0), 2)

else:
    # 못찾았을 경우
    print("No contours found. Cannot calculate center point or define ROI dynamically.")
    roi_extracted = gray.copy() 
# 원본과 이진화 영상 AND 연산, 공통 부분 찾아 추출 
bitwise_and_result = cv2.bitwise_and(gray, binary_adapt)
# 마스크 활용 관심영역 추출
mask = np.zeros_like(gray)
if contours:
    cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)

masked_tape = cv2.bitwise_and(gray, gray, mask=mask)

cv2.imshow('1. Original Gray', gray)
cv2.imshow('2. Adaptive Binary', binary_adapt)
cv2.imshow('3. Tape Center and ROI', img_color) 
cv2.imshow('4. Extracted ROI Area', roi_extracted)
cv2.imshow('5. Bitwise AND (Gray & Binary)', bitwise_and_result)
cv2.imshow('6. Masked Tape Area', masked_tape)

cv2.waitKey(0)
cv2.destroyAllWindows()