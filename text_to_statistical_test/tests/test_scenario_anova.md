# ANOVA 분석 테스트 시나리오

## 데이터셋 정보
- **파일명**: `marketing_campaign_analysis.csv`
- **분석 대상**: 마케팅 캠페인 유형별 전환율 비교
- **그룹**: email, social_media, traditional_ads
- **종속변수**: conversion_rate

## 테스트 명령어

### 기본 테스트
```bash
poetry run python -m src.main --file marketing_campaign_analysis.csv --request "서로 다른 캠페인 유형들 사이에서 전환율에 유의미한 차이가 있는지 검증해주세요"
```

### 대안 표현 테스트 (강건성 평가)

#### 변형 1: 다른 용어 사용
```bash
poetry run python -m src.main --file marketing_campaign_analysis.csv --request "각 마케팅 전략의 효과성을 비교분석하여 어떤 방법이 가장 우수한 성과를 보이는지 확인해주세요"
```

#### 변형 2: 구체적 그룹 언급
```bash
poetry run python -m src.main --file marketing_campaign_analysis.csv --request "이메일 마케팅과 소셜미디어, 그리고 전통적 광고 방식의 전환율을 비교해주세요"
```

#### 변형 3: 통계적 용어 사용
```bash
poetry run python -m src.main --file marketing_campaign_analysis.csv --request "campaign_type에 따른 conversion_rate의 평균값 차이가 통계적으로 유의한지 ANOVA로 검정해주세요"
```

## 예상 결과

### 분석 절차
1. 데이터 탐색 및 기술통계
2. 정규성 검정 (각 그룹별 Shapiro-Wilk)
3. 등분산성 검정 (Levene's test)
4. One-way ANOVA 실행
5. 사후 검정 (Tukey HSD 또는 Bonferroni)

### 기대 결과
- social_media 그룹이 가장 높은 전환율
- traditional_ads 그룹이 가장 낮은 전환율
- 통계적으로 유의한 그룹 간 차이 존재

## 평가 기준

### 성공 기준
- [x] 올바른 ANOVA 분석 수행
- [x] 적절한 사전 가정 검정
- [x] 사후 검정 실행
- [x] 비즈니스적 해석 제공
- [x] 실행 가능한 권장사항 제시

### 강건성 평가
- [x] 다양한 자연어 표현 이해
- [x] 컬럼명과 요청 용어 매핑
- [x] 적절한 통계 기법 선택
- [x] 오류 상황 대응 능력 