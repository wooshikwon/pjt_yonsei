# Linear Regression 분석 테스트 시나리오

## 데이터셋 정보
- **파일명**: `house_price_prediction.csv`
- **분석 대상**: 주택 특성이 시장 가치에 미치는 영향
- **독립변수**: property_size, bedroom_count, bathroom_count, garage_spaces, lot_area, year_built
- **종속변수**: market_value

## 테스트 명령어

### 기본 테스트
```bash
poetry run python -m src.main --file house_price_prediction.csv --request "부동산 크기와 방의 개수가 주택 시장가치에 어떤 영향을 주는지 예측 모델을 만들어주세요"
```

### 대안 표현 테스트 (강건성 평가)

#### 변형 1: 다른 용어 사용
```bash
poetry run python -m src.main --file house_price_prediction.csv --request "집의 평수와 침실 수가 부동산 가격을 어떻게 결정하는지 회귀분석으로 알아보세요"
```

#### 변형 2: 예측 모델 강조
```bash
poetry run python -m src.main --file house_price_prediction.csv --request "주택의 물리적 특성들을 바탕으로 시장가격을 예측할 수 있는 선형 모델을 구축해주세요"
```

#### 변형 3: 구체적 변수 언급
```bash
poetry run python -m src.main --file house_price_prediction.csv --request "property_size, bedroom_count, year_built 등의 요인들이 market_value에 미치는 영향을 정량적으로 분석해주세요"
```

#### 변형 4: 인과관계 표현
```bash
poetry run python -m src.main --file house_price_prediction.csv --request "건물 규모와 건축연도가 주택 가치에 미치는 선형적 관계를 파악하고 싶습니다"
```

## 예상 결과

### 분석 절차
1. 데이터 탐색 및 상관관계 분석
2. 다중공선성 확인 (VIF)
3. 다중 선형 회귀 모델 구축
4. 회귀 가정 검증 (잔차 분석)
5. 모델 유의성 및 적합도 평가
6. 회귀 계수 해석

### 기대 결과
- property_size가 가장 강한 양의 영향
- year_built도 유의한 양의 영향
- 높은 R² 값 (0.8 이상)
- 모든 가정 만족

## 평가 기준

### 성공 기준
- [x] 적절한 변수 선택
- [x] 다중 선형 회귀 수행
- [x] 회귀 가정 검증
- [x] 모델 성능 평가
- [x] 계수의 실용적 해석

### 강건성 평가
- [x] 용어 변화에 대한 적응
- [x] 예측 모델 구축 의도 파악
- [x] 적절한 독립변수 선택
- [x] 통계적 유의성 검증 