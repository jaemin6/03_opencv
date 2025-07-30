import cv2
import numpy as np

# 1. 사진 불러오기 (그레이스케일)
img = cv2.imread('../img/line_tape.png', cv2.IMREAD_GRAYSCALE)

# 2. 고정 임계값 이진화
ret, binary_fixed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 3. 적응적 이진화
binary_adapt = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY, 11, 2)

# 4. 관심영역 ROI 설정 및 마스크 적용
h, w = img.shape
x, y, w_roi, h_roi = w//4, h//4, w//2, h//2
mask = np.zeros_like(img)
mask[y:y+h_roi, x:x+w_roi] = 255
masked_img = cv2.bitwise_and(binary_fixed, binary_fixed, mask=mask)

img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.rectangle(img_color, (x,y), (x+w_roi, y+h_roi), (0,255,0), 2)

# 5. 결과 출력
cv2.imshow('Binary Fixed', binary_fixed)
cv2.imshow('Binary Adaptive', binary_adapt)
cv2.imshow('Masked Image', masked_img)
cv2.imshow('ROI', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
