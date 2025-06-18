# Auto MPG (연비) 데이터셋

## 데이터셋 설명

1970년대 후반부터 1980년대 초반까지의 다양한 자동차 모델에 대한 연비(miles per gallon) 및 기타 특성을 담고 있는 데이터셋입니다. 주로 연비(mpg)를 예측하는 회귀 분석 모델을 훈련하고 테스트하는 데 사용됩니다.

## 컬럼 설명

- **mpg**: 연비 (Miles Per Gallon). 1갤런의 연료로 주행할 수 있는 마일 거리. (연속형, 예측 대상)
- **cylinders**: 엔진의 실린더 개수 (이산형: 3, 4, 5, 6, 8)
- **displacement**: 배기량 (Cubic Inches, 연속형)
- **horsepower**: 마력 (Horsepower, 연속형)
- **weight**: 차량 무게 (Pounds, 연속형)
- **acceleration**: 0에서 60mph까지 도달하는 데 걸리는 시간 (초, 연속형)
- **model_year**: 모델 연식 (이산형, 예: 70은 1970년을 의미)
- **origin**: 제조 국가 (범주형)
    - 1: 미국 (USA)
    - 2: 유럽 (Europe)
    - 3: 일본 (Japan)
- **car_name**: 자동차 모델명 (문자열, 고유 식별자) 