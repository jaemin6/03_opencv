import cv2
import matplotlib.pyplot as plt

# 저장된 사진 경로에 맞게 수정
img = cv2.imread('../img/line_tape.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("실패")
else:
    print("성공")


hist = cv2.calcHist([img], [0], None, [256], [0, 256])