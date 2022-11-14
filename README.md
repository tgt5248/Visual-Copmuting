# Visual-Copmuting
파이썬을 이용한 Canny-edge, Image Pyramid, Histogram Equalization, Image Stitching, Calibration, Panda3D



## Canny-edge
<b>stage1.</b> RGB 채널의 이미지를 GRAYSCALE로 변환 후 노이즈 감소를 위한 스무딩 필터 적용

<b>stage2.</b> sobel 커널을 수평,수직 방향으로 적용 후 Edge_Gradient와 Angle계산
low_threshold value보다 낮은 값을 가지는 픽셀을 제거하여 Edge candidates를 구함
        
<b>stage3.</b> 픽셀의 방향과 연결된 엣지 픽셀들을 비교하여 Gradient 값이 최대인 픽셀만 남겨 Edge Thinning 수행 

<b>stage4.</b> 픽셀의 Gradient값이 high_threshold value보다 크면 확실한 엣지로 결정 (빨간색), 
low ~ high_threshold value 사이 값이라면 확실한 엣지와 연결되어 있을 경우 엣지로 선정 (파란색),
확실한한 엣지와 연결되어 있지 않아 선정되지 못한 픽셀 (노란색)

<b>result.</b> 노란색을 제외한 최종 엣지 검출 결과

<img width="452" alt="image" src="https://user-images.githubusercontent.com/63574571/169782753-349c2e7b-b346-4034-b6c1-37ee34fe3b6f.png">


## Image Pyramid
1. 눈, 손 이미지를 이용해 각각 가우시안 피라미드, 라플라시안 피라미드를 구함

2. 가우시안, 라플라시안 피라미드를 이용하여 손바닥 안에 눈 합성

3. 이미지에 라플라시안 피라미드를 더하는것을 반복하여 이미지가 선명해도록함

<img width="452" alt="image" src="https://user-images.githubusercontent.com/63574571/201585591-33121050-43a2-4b00-b63b-61ea237e8984.png">
<img width="317" alt="image" src="https://user-images.githubusercontent.com/63574571/201586195-447df82e-2f47-4f4f-96a2-1b07b73f8bc2.png">
단순히 두 이미지를 합친 것보다 경계가 희미해져 자연스럽게 합성되었음을 확인할 수 있다.



## Histogram Equalization
<b>Original.</b> HSV 채널의 이미지의 V값으로 cdf(누적분포함수)를 구함

<b>cv.HE.</b> V값에 관해서 Histogram Equalization을 진행하여 HE가 적용된 이미지 생성. => 이미지의 대비 상승

<b>AHE.</b> V값을 기준으로 Adaptive Histogram Equalization을 진행. => HE 과도/소 밝기 문제점 해결 

<b>CLAHE.</b> 과도하게 기울어진 cdf를 제한하기 위해 clipLimit을 설정하여 Contrast Limited AHE 진행. => 노이즈 감소 확인 

<img width="452" alt="image" src="https://user-images.githubusercontent.com/63574571/169783052-04d8fc9c-d77b-46c3-80d2-4a386ed664f7.png">



## Image Stitching
1. SIFT 알고리즘을 통해 이미지의 keypoint, description 계산

2. 기반 이미지를 선택하고 나머지 이미지들과의 description으로 KNN-Matching 진행하여 good correpondences 계산

3. good correpondences이 가장 높은 이미지와 stitching

4. 반복

 <img width="473" alt="image" src="https://user-images.githubusercontent.com/63574571/201588393-c104717d-d83f-4bbb-b6b4-dc63dccf5bbd.png">

<img width="452" alt="IMG_7050" src="https://user-images.githubusercontent.com/63574571/169783693-e3705470-94fd-43d7-9d95-657d3de8b615.PNG">

## Panda3D
아루코 마커 위에 객체를 그리는 AR 

1. 노트북 웹캠을 사용하여 Camera calibration 진행
<img width="468" alt="image" src="https://user-images.githubusercontent.com/63574571/201590607-3f8b1a3a-a361-414c-83a5-8b9863d6df50.png">

2. ArUco Marker 인식하고 축 표시
<img width="231" alt="image" src="https://user-images.githubusercontent.com/63574571/201590562-ebc4ae36-045f-436b-a666-ed702d95e458.png">

3. 객체를 marker에 배치 후 marker의 좌표계를 조정하여 축 표시
<img width="219" alt="image" src="https://user-images.githubusercontent.com/63574571/201590539-ea8f1ec0-c7b4-4ebf-bcde-fd86eb2f5a21.png">


4. 3개의 marker 생성 후 각각 다른 객체를 배치, 축 표시
<img width="452" alt="image" src="https://user-images.githubusercontent.com/63574571/169783863-b9c180c1-2a3b-4a48-ab02-97dac6c3d998.png">

5. 애니메이션 및 Rendering effets(point light, ambient light)추가
<img width="604" alt="image" src="https://user-images.githubusercontent.com/63574571/201590485-51223462-49f8-4997-9600-3755fc99aa0f.png">


