# services/statistics/functions/effect_sizes.py
"""
통계적 효과 크기(Effect Size) 계산 함수 모음.

이 모듈의 함수들은 t-test, ANOVA, 카이제곱 검정 등의
결과를 바탕으로 효과의 실제적인 크기를 측정하는 순수 계산 로직을 담당합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

def calculate_cohens_d(group1: pd.Series, group2: pd.Series) -> Dict[str, float]:
    """
    두 그룹 간의 평균 차이에 대한 Cohen's d 효과 크기를 계산합니다.
    """
    n1, n2 = len(group1), len(group2)
    s1, s2 = group1.var(ddof=1), group2.var(ddof=1)
    # 합동 표준편차 (Pooled Standard Deviation)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = group1.mean(), group2.mean()
    cohen_d = (u1 - u2) / s
    return {"Cohen's d": cohen_d}

def calculate_partial_eta_squared(anova_table: Dict[str, Any]) -> Dict[str, Any]:
    """이원 분산 분석(Two-way ANOVA) 결과에서 부분 에타 제곱(Partial Eta-squared)을 계산합니다."""
    partial_eta_sq = {}
    ss_error = anova_table.get('Residual', {}).get('sum_sq')

    if ss_error is None:
        return {'partial_eta_squared': {}, 'error': 'Residual sum of squares not found.'}
        
    for factor, values in anova_table.items():
        if factor == 'Residual':
            continue
            
        ss_effect = values.get('sum_sq')
        if ss_effect is not None:
            partial_eta_sq[factor] = ss_effect / (ss_effect + ss_error)
            
    return {'partial_eta_squared': partial_eta_sq}

def calculate_eta_squared(f_statistic: float, df_between: int, df_within: int) -> Dict[str, float]:
    """일원 분산 분석(One-way ANOVA) 결과에서 에타 제곱(Eta-squared)을 계산합니다."""
    eta_sq = (f_statistic * df_between) / (f_statistic * df_between + df_within)
    return {"Eta-squared": eta_sq}

def calculate_cramers_v(chi2_stat: float, n: int, contingency_table: pd.DataFrame) -> Dict[str, float]:
    """카이제곱 검정 결과에서 크래머 V(Cramer's V) 효과 크기를 계산합니다."""
    min_dim = min(contingency_table.shape) - 1
    if min_dim == 0: 
        return {"Cramer's V": 0.0}
    cramers_v = np.sqrt(chi2_stat / (n * min_dim))
    return {"Cramer's V": cramers_v}

def calculate_cohens_h(prop1: float, prop2: float) -> Dict[str, float]:
    """두 표본 비율 검정 결과에 대한 Cohen's h 효과 크기를 계산합니다."""
    # np.arcsin의 입력값은 -1과 1 사이여야 하므로, np.sqrt의 결과가 1을 넘지 않도록 clip 처리
    h = 2 * np.arcsin(np.sqrt(np.clip(prop1, 0, 1))) - 2 * np.arcsin(np.sqrt(np.clip(prop2, 0, 1)))
    return {"Cohen's h": h}