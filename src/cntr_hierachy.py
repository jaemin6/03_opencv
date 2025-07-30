import cv2
import numpy as np

# 이미지 불러오기
img = cv2.imread('../img/shapes_donut.png')
img2 = img.copy()

# 바이너리 이미지로 변환
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, imthres = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)

# 가장 바깥쪽 컨투어만 찾기
contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(len(contour), hierarchy)

# 모든 컨투어를 계층 트리 구조로 찾기
contour2, hierarchy = cv2.findContours(imthres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contour2), hierarchy)

cv2.drawContours(img, contour, -1, (0,255,0), 3)

for idx, cont in enumerate(contour2):
    color = [int(i) for i in np.random.randint(0,255,3)]
    cv2.drawContours(img2, contour2, idx, color, 3)
    cv2.putText(img2, str(idx), tuple(cont[0][0]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))

cv2.imshow('RETR_EXTERNAL', img)
cv2.imshow('RETR_TREE', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
