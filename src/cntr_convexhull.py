import cv2
import numpy as np

img = cv2.imread('../img/hand.jpg')
img2 = img.copy()











cv2.imshow('contour', img)
cv2.imshow('convex hull', img2)
cv2.waitKey()
cv2.destroyAllWindows()