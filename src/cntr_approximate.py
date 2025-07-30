import cv2
import numpy as np

img = cv2.imread('../img/bad_rect.png')





cv2.imshow('contour', img)
cv2.imshow('approx', img2)
cv2.waitKey()
cv2.destroyAllWindows()