import cv2
import numpy as np

# 1. 사진 불러오기 (이미 그레이스케일로 불러옴)
img = cv2.imread('../img/line_tape.png', cv2.IMREAD_GRAYSCALE)

# 2. 고정 임계값 이진화
ret, binary_fixed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 3. 적응적 이진화
binary_adapt = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY, 11, 2)

# 4. 관심영역 ROI 설정 및 시각화
h, w = img.shape  # ✅ img는 이미 gray임

# 관심영역 추출
roi = img[h//2-100:h//2+100, w//2-30:w//2+30]

# 시각화용 이미지 (컬러로 변환 후 사각형 그리기)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.rectangle(img_color, (w//2 - 30, h//2 - 100), (w//2 + 30, h//2 + 100), (0, 255, 0), 2)

cv2.imshow('ROI', img_color)
cv2.imshow('Cut', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
