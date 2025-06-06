# 통계적 가설 검정 (Hypothesis Testing)

## 개요
통계적 가설 검정은 표본 데이터를 사용하여 모집단에 대한 가설을 검증하는 통계적 추론 방법입니다.

## 기본 개념

### 가설 설정
- **귀무가설 (H₀)**: 기존 이론이나 현상을 나타내는 가설
- **대립가설 (H₁)**: 새로운 주장이나 변화를 나타내는 가설

### 검정 통계량
각 검정마다 적절한 검정 통계량을 계산하여 가설을 검증합니다.

### 유의수준 (α)
- 일반적으로 0.05 (5%) 사용
- 제1종 오류(Type I Error)의 확률

## 주요 검정 방법

### 1. t-검정 (t-test)
**적용 조건:**
- 정규분포 가정
- 표본 크기가 작거나 모집단 분산을 모를 때

**종류:**
- 일표본 t-검정: 표본 평균과 특정 값 비교
- 독립표본 t-검정: 두 독립 그룹의 평균 비교
- 대응표본 t-검정: 동일 대상의 전후 비교

### 2. 분산분석 (ANOVA)
**적용 조건:**
- 세 개 이상의 그룹 비교
- 정규성, 등분산성, 독립성 가정

**종류:**
- 일원분산분석 (One-way ANOVA)
- 이원분산분석 (Two-way ANOVA)

### 3. 카이제곱 검정 (Chi-square test)
**적용 조건:**
- 범주형 데이터
- 기대빈도가 5 이상

**종류:**
- 적합도 검정 (Goodness of fit)
- 독립성 검정 (Test of independence)

### 4. 비모수 검정 (Non-parametric tests)
**적용 조건:**
- 정규성 가정 불만족
- 서열척도 데이터

**종류:**
- Mann-Whitney U 검정
- Kruskal-Wallis 검정
- Wilcoxon 부호순위 검정

## 검정 가정 확인

### 정규성 검정
- Shapiro-Wilk 검정
- Kolmogorov-Smirnov 검정
- Anderson-Darling 검정

### 등분산성 검정
- Levene's 검정
- Bartlett's 검정
- Brown-Forsythe 검정

### 독립성 확인
- 잔차 플롯 분석
- Durbin-Watson 검정

## 결과 해석

### p-값 해석
- p < α: 귀무가설 기각, 통계적으로 유의
- p ≥ α: 귀무가설 채택, 통계적으로 유의하지 않음

### 효과 크기 (Effect Size)
- Cohen's d: t-검정의 효과 크기
- η² (Eta-squared): ANOVA의 효과 크기
- Cramer's V: 카이제곱 검정의 효과 크기

### 신뢰구간
검정 결과와 함께 모수의 신뢰구간을 제시하여 실질적 의미 파악

## 사후 분석 (Post-hoc Analysis)

### ANOVA 후 사후 검정
- Tukey HSD: 모든 쌍 비교
- Bonferroni: 보수적 접근
- Scheffe: 대비 분석에 적합

### 다중비교 문제
여러 검정을 동시에 수행할 때 제1종 오류율 증가 문제 해결

## 검정력 분석 (Power Analysis)

### 개념
- 검정력 = 1 - β (제2종 오류 확률)
- 실제 효과가 있을 때 이를 검출할 확률

### 표본 크기 결정
사전 검정력 분석을 통한 적정 표본 크기 산출

## 실무 적용 가이드

### 검정 방법 선택 플로우차트
1. 데이터 유형 확인 (연속형/범주형)
2. 그룹 수 확인 (1개/2개/3개 이상)
3. 가정 확인 (정규성, 등분산성)
4. 적절한 검정 방법 선택

### 결과 보고 방법
- 기술통계: 평균, 표준편차, 빈도
- 검정 결과: 검정통계량, 자유도, p-값
- 효과 크기와 신뢰구간
- 실질적 해석

## 주의사항

### 흔한 오류
- 다중비교 미고려
- 가정 검토 생략
- p-hacking (결과 조작)
- 효과 크기 무시

### 윤리적 고려사항
- 투명한 분석 과정 공개
- 모든 결과 보고 (유의/비유의 모두)
- 연구 설계의 한계점 명시

## 관련 통계 소프트웨어
- Python: scipy.stats, statsmodels
- R: base R, car package
- SPSS: 기본 메뉴 제공
- SAS: PROC TTEST, PROC ANOVA

## 참고문헌
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
- Field, A. (2013). Discovering Statistics Using IBM SPSS Statistics
- Montgomery, D. C. (2017). Design and Analysis of Experiments 