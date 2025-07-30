import cv2                                        # opencv 라이브러리 불러오기
import numpy as np                                # numpy 라이브러리 불러오기

# 웹캠 열기 (기본 카메라: 0번)
cap = cv2.VideoCapture(0)
# 캠이 열렸는지 확인 하는 단계
if not cap.isOpened():
    print("Error: Camera not accessible")         # 에러메세지 출력
    exit()                                        # 프로그램 종료
# 실시간으로 프레임을 받는 루프
while True:
    ret, frame = cap.read()                       # 캠에서 한 프레임을 캡쳐
    if not ret:                                   # ret = 성공여부, frame = 이미지
        print("Error: Frame not captured")        # 프레임을 못 읽을 경우
        break                                     # 루프 종료

### 영상 처리 단계 ###

    # 1단계: 컬러 영상을 흑백(Grayscale)으로 변환 (색 정보 제거, 계산 단순화)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2단계: 이진화 처리 (흑백 라인 분류) 밝기 변화가 있는 환경에서도 동일
    binary_adapt = cv2.adaptiveThreshold(
        gray,                          # 입력: 그레이스케일 이미지
        255,                           # 최대값 (흰색)
        cv2.ADAPTIVE_THRESH_MEAN_C,    # 주변 픽셀 평균값 기반으로 임계값 계산
        cv2.THRESH_BINARY_INV,         # 흰색 → 검정, 검정 → 흰색 (반전)
        11,                            # 블록 크기 (11x11 영역 기준)
        2                              # 평균값에서 -2 한 값을 임계값으로 사용
    )
    # 컬러 프레임 복사 (중심점 표시용)
    img_color = frame.copy()

    # 3단계: 외곽선 찾기
    contours, _ = cv2.findContours(binary_adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 중심 좌표를 초기화 없을 경우 -1로표시시
    tape_center_x, tape_center_y = -1, -1

    if contours:
        # 가장 큰 외곽선 선택 (컨투어)
        largest_contour = max(contours, key=cv2.contourArea)
        # 무게 중심 계산 (컨투어 모멘트 사용)
        M = cv2.moments(largest_contour)
        
        if M["m00"] != 0:
            tape_center_x = int(M["m10"] / M["m00"])       # x 중심좌표표
            tape_center_y = int(M["m01"] / M["m00"])       # y 중심좌표

            # 중심점 빨간 동그라미로 표시
            cv2.circle(img_color, (tape_center_x, tape_center_y), 5, (0, 0, 255), -1)
            center_text = f"({tape_center_x}, {tape_center_y})"
            cv2.putText(img_color, center_text, (tape_center_x + 10, tape_center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ROI 사각형 표시 (초록색)
        x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(largest_contour)
        cv2.rectangle(img_color, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (0, 255, 0), 2)

        # 마스크 생성 (외곽선만 남기기)
        mask = np.zeros_like(gray)                                      # np.zeros 사용 비어있는 검정색 생성
        cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)  # 외곽선

            # 마스크 적용 결과 (흑백 이미지에서 해당 부분만 보이게 설정)
        masked_tape = cv2.bitwise_and(gray, gray, mask=mask)
    else:   # 외곽선이 없으면 원본 회색 이미지를 사용
        masked_tape = gray.copy()

    # 이진화 AND 연산 결과
    bitwise_and_result = cv2.bitwise_and(gray, binary_adapt)

    # 출력창 여러개 띄우기
    cv2.imshow("1. Original + Center", img_color)       # 1. 중심점 포함한 원본 영상
    cv2.imshow("2. Adaptive Binary", binary_adapt)      # 2. 이진화로 흑백 처리된 결과
    cv2.imshow("3. Masked Tape Area", masked_tape)      # 3. 검은색 띠 영역만 마스크로 추출한 영상
    cv2.imshow("4. Bitwise AND", bitwise_and_result)    # 4. 원본과 이진화가 결합된 영상

    # 종료 조건 ESC
    key = cv2.waitKey(1)
    if key == 27:                 # 27은 아스키 코드로 esc 
        break

cap.release()
cv2.destroyAllWindows()
