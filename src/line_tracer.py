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

# 컬러 입히기
img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

contours, _ = cv2.findContours(binary_adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

tape_center_x, tape_center_y = -1, -1 

if contours:
    largest_contour = max(contours, key=cv2.contourArea)

    M = cv2.moments(largest_contour)

    if M["m00"] != 0:
        tape_center_x = int(M["m10"] / M["m00"])
        tape_center_y = int(M["m01"] / M["m00"])

        cv2.circle(img_color, (tape_center_x, tape_center_y), 5, (0, 0, 255), -1)

        center_text = f"Center: ({tape_center_x}, {tape_center_y})"
        cv2.putText(img_color, center_text, (tape_center_x + 10, tape_center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(largest_contour)
    padding_roi = 20
    x1_padded = max(0, x_roi - padding_roi)
    y1_padded = max(0, y_roi - padding_roi)
    x2_padded = min(gray.shape[1], x_roi + w_roi + padding_roi)
    y2_padded = min(gray.shape[0], y_roi + h_roi + padding_roi)
    roi_extracted = gray[y1_padded:y2_padded, x1_padded:x2_padded]

    cv2.rectangle(img_color, (x1_padded, y1_padded), (x2_padded, y2_padded), (0, 255, 0), 2)

else:
    print("No contours found. Cannot calculate center point or define ROI dynamically.")
    roi_extracted = gray.copy() 

bitwise_and_result = cv2.bitwise_and(gray, binary_adapt)

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