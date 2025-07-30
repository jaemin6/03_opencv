# 🌟 1단계: 기본 객체인식 (Basic Object Detection)

## 📖 학습 목표

* 웹캠을 통해 경영의 경계를 구현한 것과 같이 객체 보고 인식할 수 있도록 함
* 시간적 표시를 통해 결과 보여주기

---

## 🎯 미션 1-1: 웹캠 영상 획득 및 기본 처리

### ▶️ 준비 작업:

* A4용지에 경영보통의 경계 (전기 절염 테이프)를 세로로 길게 붙인다
* 웹캠을 이 용지를 지면적으로 보게 보설

### ▶️ 실습 과제:

#### 1. 웹캠 영상 읽기

```python
cap = cv2.VideoCapture(0) # 0는 보호 웹캠
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

* `cv2.VideoCapture()` : 현재 웹캠의 영상을 가져오기 위해 사용
* 일정 해상도로 만들어 보기 편한 그래픽 설정

#### 2. 상호 변환 (Color Conversion)

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
```

* `GRAY`: 메이터리 포맷을 제거해서 화상 메인로 써다
* `HSV`: 색상과 베객을 구분 하여 선명한 인식을 위해 사용

#### 3. 라이브 루프 구현:

```python
while True:
    ret, frame = cap.read()
    # ... 처리 ...
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

* `while True`: 보복적으로 영상을 가져오는 루프
* `cv2.waitKey()`: 키를 눌러 라인을 종료 할 수 있게 함

### ▶️ 결과:

* 웹캠에서 경영 서탄을 보여주며, 다양한 그래이스캔 인식 준비

---

## 🎯 미션 1-2: 히스토그러트(타겟) 배경과 구분

### ▶️ 실습 과제:

#### 1. 히스토그러트 계산:

```python
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
```

* 그레이스시케일을 대상으로 256가지 값에 대해 히스토그러트 계산

#### 2. 히스토그러트 그래프 표시:

```python
plt.plot(hist)
plt.title('Histogram')
plt.show()
```

#### 3. 최근 구분점 찾기:

* 테이프가 가장 지난 포건에 대해 어떻게 하여 값을 분리하는지 어고하게 구현

---

## 🎯 미션 1-3: 이진화 및 객체 검색

### ▶️ 이진화 (Thresholding)

```python
ret, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
```

* `gray` 이미지의 값이 100 이상이면 255(흔상), 그 이하는 0(검색)

### ▶️ 적응형 Threshold

```python
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
```

* 지역에 따라 자동적으로 값 설정

### ▶️ ROI(관심영역) 설정

```python
roi = frame[y:y+h, x:x+w]
```

* 일반적인 A4의 특정 범위만 해당
* 바가운드가 아니라 테이프가 보이는 경우 필요

---

## 🎯 미션 1-4: 기본 객체 정보 추출

### ▶️ 중심점(모명) 계산:

```python
M = cv2.moments(contour)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
```

* `cv2.moments()` : 객체의 모명을 계산하여 무게중점을 구해

### ▶️ 시간적 표시:

```python
cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
cv2.putText(frame, f"({cx},{cy})", (cx+10, cy), ...)
```

* 모델의 주식과 그래픽 창을 이용

### ▶️ 이미지 연산

```python
masked = cv2.bitwise_and(frame, frame, mask=binary)
```

* 바가운드 연산으로 관심영역과 관련된 모양을 찾음

---

## 🏆 1단계 완료 조건 체크리스트
| 항목                  | 확인 여부 |
| ------------------- | ----- |
| 웹캠에서 실시간으로 테이프 인식   | ✅     |
| 히스토그램 분석으로 픽셀 분포 확인 | ✅     |
| 이진화로 테이프 영역 분리 성공   | ✅     |
| 중심점 계산 및 시각화 구현     | ✅     |

## 🖼️ 예시 이미지 모음
### 📷 히스토그램 분석 화면
<img width="450" alt="histogram_example" src="https://upload.wikimedia.org/wikipedia/commons/5/56/Histogramme.png" />

### 🎯 중심점 표시 예시
<img width="450" alt="centroid_example" src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Image_moments.svg/512px-Image_moments.svg.png" />

## 🛠 사용한 라이브러리 요약

| 라이브러리          | 역할              |
| -------------- | --------------- |
| `cv2` (OpenCV) | 이미지/비디오 처리 전반   |
| `numpy`        | 수치 연산, 마스크 생성 등 |
| `matplotlib`   | 히스토그램 시각화용      |




# 📌 컨투어(Contour) 관련 개념 정리

## ✅ 컨투어란?
### 이미지에서 경계선을 찾아내는 기능. 외곽선을 따서 윤곽을 파악함.
### 예를 들어 테이프 라인의 경계나 장애물의 모양을 파악할 때 유용함.

## ✅ 컨투어 추출 함수
### cv2.findContours(image, mode, method)
### - image: 입력 이진 이미지 (흑백 or 바이너리 이미지만 가능)
### - mode: 윤곽선 추출 방식 (예: cv2.RETR_EXTERNAL → 외곽선만)
### - method: 윤곽선 근사화 방법 (예: cv2.CHAIN_APPROX_SIMPLE → 꼭지점만 저장)

## ➕ 사용 예
### contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

## ✅ 가장 큰 외곽선 찾기
### 가장 면적이 넓은 contour를 찾을 땐 max()와 cv2.contourArea() 사용
### largest_contour = max(contours, key=cv2.contourArea)

## ✅ 중심점 구하기 (무게중심, moment 이용)
### M = cv2.moments(contour)
### cx = int(M["m10"] / M["m00"])
### cy = int(M["m01"] / M["m00"])

## ✅ 외곽선 그리기
### cv2.drawContours(output_img, [contour], -1, (색상), thickness)

## ✅ 외곽선의 사각형 영역 (bounding box)
### x, y, w, h = cv2.boundingRect(contour)
### 이 값은 관심 영역(ROI)로도 활용 가능

## ✅ 외곽선 내부만 추출하고 싶다면?
### 1. 마스크 이미지 만들기 (mask = np.zeros_like(gray))
### 2. cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED) 로 흰색으로 채움
### 3. cv2.bitwise_and() 로 마스크 영역만 추출

## ✅ 컨투어는 이진 이미지에서만 동작!
### 그래서 꼭 이진화(스레숄드 or 적응형 이진화) 이후에 사용해야 함.

## ✅ Tip:
### - 작은 노이즈 제거: contour 크기 비교해서 작은 것 무시
### - 내부 컨투어도 찾고 싶다면 cv2.RETR_TREE 등도 사용 가능

