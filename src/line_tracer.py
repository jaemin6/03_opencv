import cv2
import numpy as np

# 1. 사진 불러오기 (그레이스케일)
img = cv2.imread('../img/line_tape.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: Image not found. Please check the path.")
    exit()

gray = img.copy()

# 2. 적응적 이진화 
binary_adapt = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV, 11, 2)

# 시각화를 위한 컬러 이미지
img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

contours, _ = cv2.findContours(binary_adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)

else:
    print("No contours found.")
   
    roi_extracted = gray.copy() # 예시


cv2.imshow('1. Original Gray', gray)
cv2.imshow('2. Adaptive Binary', binary_adapt)

cv2.waitKey(0)
cv2.destroyAllWindows()