import cv2
import matplotlib.pyplot as plt

# 저장된 사진 불러오기
img = cv2.imread('../img/line_tape.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("실패")
    exit()

hist = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.plot(hist, color='black')
plt.title('Grayscale Histogram')
plt.xlabel('Pixel value')
plt.ylabel('Pixel count')
plt.show()
