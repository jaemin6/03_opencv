import cv2
import numpy as np

img = cv2.imread('../img/shapes.png')  # 이미지 경로 확인 필요
img2 = img.copy()

# 그레이 스케일 변환
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 이진화 (흑백 반전)
ret, imthres = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)

# 컨투어 찾기 (OpenCV 4.x 기준)
contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour2, hierarchy2 = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 컨투어 개수 출력
print('도형의 갯수: %d (%d)' % (len(contour), len(contour2)))

# 전체 좌표를 갖는 컨투어 그리기 (초록색)
cv2.drawContours(img, contour, -1, (0, 255, 0), 2)

# 꼭지점 좌표만 갖는 컨투어 그리기 (초록색)
cv2.drawContours(img2, contour2, -1, (0, 255, 0), 2)

# 모든 좌표 점 그리기 (진한 파란색)
for i in contour:
    for j in i:
        cv2.circle(img, tuple(j[0]), 1, (255, 0, 0), -1)

# 꼭지점 좌표 점 그리기 (진한 빨간색)
for i in contour2:
    for j in i:
        cv2.circle(img2, tuple(j[0]), 2, (0, 0, 255), -1)

# 결과 출력
cv2.imshow('CHAIN_APPROX_NONE', img)
cv2.imshow('CHAIN_APPROX_SIMPLE', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
