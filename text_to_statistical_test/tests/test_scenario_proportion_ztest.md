# 비율 검정 (Z-Test) 분석 테스트 시나리오

## 데이터셋 정보
- **파일명**: `manufacturing_quality_control.csv`
- **분석 대상**: 두 생산라인의 불량률 비교
- **그룹**: LineA vs LineB
- **비율 계산**: defective_units / total_units

## 테스트 명령어

### 기본 테스트
```bash
poetry run python -m src.main --file manufacturing_quality_control.csv --request "생산라인A와 생산라인B의 결함품 발생 비율에 통계적으로 의미 있는 차이가 존재하는지 검증해주세요"
```

### 대안 표현 테스트 (강건성 평가)

#### 변형 1: 다른 용어 사용
```bash
poetry run python -m src.main --file manufacturing_quality_control.csv --request "두 제조 라인의 불량률을 비교하여 품질 차이가 유의한지 확인해주세요"
```

#### 변형 2: 비율 강조
```bash
poetry run python -m src.main --file manufacturing_quality_control.csv --request "LineA와 LineB에서 defective_units의 비율이 통계적으로 다른지 비율검정을 수행해주세요"
```

#### 변형 3: 품질관리 관점
```bash
poetry run python -m src.main --file manufacturing_quality_control.csv --request "각 production_line별 품질 수준을 비교분석하여 개선이 필요한 라인을 식별해주세요"
```

#### 변형 4: 가설검정 표현
```bash
poetry run python -m src.main --file manufacturing_quality_control.csv --request "두 생산라인의 결함률이 동일하다는 귀무가설을 검증하고 싶습니다"
```

## 예상 결과

### 분석 절차
1. 각 그룹별 불량률 계산
2. 표본 크기 및 성공 조건 확인
3. 이표본 비율 Z-검정 수행
4. p-value 계산 및 해석
5. 신뢰구간 구성
6. 효과 크기 측정

### 기대 결과
- LineB가 LineA보다 높은 불량률
- 통계적으로 유의한 차이 존재 (p < 0.05)
- 실용적으로도 의미 있는 차이
- LineB 개선 권장사항 제시

## 평가 기준

### 성공 기준
- [x] 올바른 비율 계산
- [x] 적절한 Z-검정 수행
- [x] 신뢰구간 제시
- [x] 실용적 의미 해석
- [x] 개선 방안 제시

### 강건성 평가
- [x] 비율/불량률 용어 인식
- [x] 그룹 비교 의도 파악
- [x] 적절한 검정 방법 선택
- [x] 품질관리 맥락 이해

## 데이터 특성
- **LineA**: 약 2.4% 불량률 (신형 기계)
- **LineB**: 약 5.2% 불량률 (구형 기계)
- **표본 크기**: 각 라인당 20배치
- **배치당 평균**: 약 500단위 