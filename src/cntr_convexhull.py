import cv2
import numpy as np

img = cv2.imread('../img/hand.jpg')
img2 = img.copy()

# 그레이 스케일 및 바이너리 스케일 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# 컨투어 찾기와 그리기 (OpenCV 4.x 기준: 반환값 2개)
contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntr = contours[0]
cv2.drawContours(img, [cntr], -1, (0, 255, 0), 1)

# 볼록 선체 찾기(좌표 기준)와 그리기
hull = cv2.convexHull(cntr)
cv2.drawContours(img2, [hull], -1, (0, 255, 0), 1)

# 볼록 선체 만족 여부 확인
print(cv2.isContourConvex(cntr), cv2.isContourConvex(hull))

# 볼록 선체 찾기(인덱스 기준)
hull2 = cv2.convexHull(cntr, returnPoints=False)

# 볼록 선체 결함 찾기
defects = cv2.convexityDefects(cntr, hull2)

# 볼록 선체 결함 순회
if defects is not None:
    for i in range(defects.shape[0]):
        startP, endP, farthestP, distance = defects[i, 0]
        farthest = tuple(cntr[farthestP][0])
        dist = distance / 256.0
        if dist > 1:
            cv2.circle(img2, farthest, 3, (0, 0, 255), -1)

cv2.imshow('contour', img)
cv2.imshow('convex hull', img2)
cv2.waitKey()
cv2.destroyAllWindows()
