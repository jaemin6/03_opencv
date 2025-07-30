import cv2

# 저장된 사진 경로에 맞게 수정
img = cv2.imread('../img/line_tape.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("실패")
else:
    print("성공")
