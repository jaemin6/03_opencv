import numpy as np

# 이미지 크기 확인
h, w = img.shape

# 관심 영역 좌표 (예: A4 용지가 이미지 가운데 있다고 가정)
x, y, w_roi, h_roi = w//4, h//4, w//2, h//2

# 마스크 생성 (검은 바탕에 관심 영역만 흰색)
mask = np.zeros_like(img)
mask[y:y+h_roi, x:x+w_roi] = 255

# 마스크 적용
masked_img = cv2.bitwise_and(binary_fixed, binary_fixed, mask=mask)

# ROI 시각화 (원본 영상에 사각형 표시)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.rectangle(img_color, (x,y), (x+w_roi, y+h_roi), (0,255,0), 2)

cv2.imshow('Masked Image', masked_img)
cv2.imshow('ROI', img_color)
