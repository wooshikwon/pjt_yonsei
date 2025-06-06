# 파일명: services/statistics/effect_size.py

import pandas as pd
import numpy as np
from typing import Dict, Any

def calculate_cohens_d(group1: pd.Series, group2: pd.Series) -> Dict[str, float]:
    """두 그룹 간의 Cohen's d 효과 크기를 계산합니다."""
    n1, n2 = len(group1), len(group2)
    s1, s2 = group1.var(ddof=1), group2.var(ddof=1)
    
    # 합동 분산 (pooled variance)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    
    u1, u2 = group1.mean(), group2.mean()
    cohen_d = (u1 - u2) / s
    return {"Cohen's d": cohen_d}

def calculate_eta_squared(f_statistic: float, df_between: int, df_within: int) -> Dict[str, float]:
    """ANOVA 결과로부터 에타 제곱(eta-squared) 효과 크기를 계산합니다."""
    eta_sq = (f_statistic * df_between) / (f_statistic * df_between + df_within)
    return {"Eta-squared": eta_sq}

def calculate_cramers_v(chi2_stat: float, n: int, contingency_table: pd.DataFrame) -> Dict[str, float]:
    """카이제곱 검정 결과로부터 크래머 V(Cramer's V) 효과 크기를 계산합니다."""
    min_dim = min(contingency_table.shape) - 1
    if min_dim == 0: return {"Cramer's V": 0.0}
    cramers_v = np.sqrt(chi2_stat / (n * min_dim))
    return {"Cramer's V": cramers_v}