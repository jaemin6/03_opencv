import cv2
import numpy as np

img = cv2.imread('../img/bad_rect.png')
img2 = img.copy()

# 그레이스케일과 바이너리 이미지 변환
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, th = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)

# 컨투어 찾기 (OpenCV 4.x 기준: 2개 리턴)
contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 첫 번째 컨투어 가져오기
contour = contours[0]

# 둘레 길이의 5% 오차로 근사화
epsilon = 0.05 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)

# 원본 컨투어와 근사화된 컨투어 각각 그리기
cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)
cv2.drawContours(img2, [approx], -1, (0, 0, 255), 3)

cv2.imshow('Original Contour', img)
cv2.imshow('Approximated Contour', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
