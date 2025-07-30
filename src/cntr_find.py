import cv2
import numpy as np

# 이미지 불러오기
img = cv2.imread('../img/shapes.png')
img2 = img.copy()

# 그레이 스케일
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 스레숄드 적용 (흑백 반전)
ret, imthres = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)

# 컨투어: 모든 좌표 유지 (CHAIN_APPROX_NONE)
contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 컨투어: 꼭지점만 유지 (CHAIN_APPROX_SIMPLE)
contour2, hierarchy2 = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 결과 출력
print('도형의 갯수: %d (%d)' % (len(contour), len(contour2)))

# 전체 좌표로 그리기 (초록색)
cv2.drawContours(img, contour, -1, (0, 255, 0), 4)

# 꼭지점 좌표만으로 그리기 (초록색)
cv2.drawContours(img2, contour2, -1, (0, 255, 0), 4)

# 전체 좌표를 파란 점으로
for i in contour:
    for j in i:
        cv2.circle(img, tuple(j[0]), 1, (255, 0, 0), -1)

# 꼭지점 좌표만 파란 점으로
for i in contour2:
    for j in i:
        cv2.circle(img2, tuple(j[0]), 1, (255, 0, 0), -1)

# 이미지 출력
cv2.imshow('CHAIN_APPROX_NONE', img)
cv2.imshow('CHAIN_APPROX_SIMPLE', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
