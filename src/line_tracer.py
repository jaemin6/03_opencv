import cv2
import numpy as np

# 1. 이미지 불러오기 (grayscale)
img = cv2.imread('../img/line_tape.png', cv2.IMREAD_GRAYSCALE)

# 2. 곤심영역 마우스로 드래그
cv2.imshow('Select ROI', img)
roi_rect = cv2.selectROI('Select ROI', img, showCrossHair=True, fromCenter=False)
cv2.destroyWindow('Select ROI')  # ROI 선택 후 창 닫기

# 3. ROI 좌표 추출
x, y, w, h = roi_rect

# 선택이 유효할 때만 실행
if w > 0 and h > 0:
    roi = img[y:y+h, x:x+w]
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 4. 이진화 처리 (고정 + 적응)
    _, binary_fixed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    binary_adapt = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 11, 2)

    # 5. 결과 창 표시 (한글 없이)
    cv2.imshow('Gray', img)
    cv2.imshow('Fixed', binary_fixed)
    cv2.imshow('Adapt', binary_adapt)
    cv2.imshow('ROI', roi)
    cv2.imshow('Boxed', img_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("ROI 캔슬")
