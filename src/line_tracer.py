import cv2
import numpy as np

# 1. 사진 불러오기 (그레이스케일)
img = cv2.imread('../img/line_tape.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: Image not found. Please check the path.")
    exit()

gray = img.copy()

# 2. 적응적 이진화 (Contour finding often works well with a good binary image)
binary_adapt = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY_INV, 11, 2) # Use THRESH_BINARY_INV if the line is dark on a light background

# 3. 윤곽선 찾기
# findContours modifies the input image, so we'll work on a copy of binary_adapt
contours, hierarchy = cv2.findContours(binary_adapt.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. 가장 큰 윤곽선 찾기 (가장 큰 것이 라인일 확률 높음)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 5. 관심영역 설정 (찾은 윤곽선 주변으로 여유 공간 추가)
    # Adjust padding as needed
    padding = 20
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(gray.shape[1], x + w + padding)
    y2 = min(gray.shape[0], y + h + padding)

    roi = gray[y1:y2, x1:x2]

    # 사각형 시각화용 컬러 이미지
    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
else:
    print("No contours found. ROI could not be defined dynamically.")
    roi = gray # Fallback to full image if no contours are found
    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# 6. 이미지 창 띄우기
cv2.imshow('Gray', gray)
cv2.imshow('Binary Adapt for Contours', binary_adapt)
cv2.imshow('ROI (Contour Based)', roi)
cv2.imshow('Boxed (Contour Based)', img_color)

cv2.waitKey(0)
cv2.destroyAllWindows()