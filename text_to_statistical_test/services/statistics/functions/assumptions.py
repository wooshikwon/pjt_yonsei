# services/statistics/functions/assumptions.py
"""
통계 분석을 위한 가정 검토 함수 모음.

이 모듈의 함수들은 정규성, 등분산성 등 통계적 가정을 검증하는
순수 계산 로직을 담당합니다.
"""

import pandas as pd
from scipy import stats
from typing import Dict, Any

def check_normality(data: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
    """
    데이터의 정규성을 Shapiro-Wilk 검정으로 확인합니다.
    """
    non_na = data.dropna()
    if len(non_na) < 3:
        return {'test': 'Shapiro-Wilk', 'passed': False, 'p_value': 0.0, 'reason': '샘플 크기가 3 미만입니다.'}
    try:
        stat, p_value = stats.shapiro(non_na)
        return {'test': 'Shapiro-Wilk', 'statistic': stat, 'p_value': p_value, 'passed': p_value > alpha}
    except Exception as e:
        return {'test': 'Shapiro-Wilk', 'passed': False, 'error': str(e)}

def check_equal_variance(*groups: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
    """
    여러 그룹 간의 등분산성을 Levene 검정으로 확인합니다.
    """
    # 비어있지 않고, 최소 2개 이상의 샘플을 가진 그룹만 필터링합니다. Levene 검정은 적은 샘플에도 동작합니다.
    valid_groups = [g.dropna() for g in groups if len(g.dropna()) >= 2]
    
    # 필터링 후 그룹이 2개 미만이면 검정 자체가 무의미합니다.
    if len(valid_groups) < 2:
        return {'test': 'Levene', 'passed': True, 'reason': '등분산성을 비교할 유효 그룹(샘플 수 2 이상)이 2개 미만입니다.'}
        
    try:
        stat, p_value = stats.levene(*valid_groups)
        return {'test': 'Levene', 'statistic': stat, 'p_value': p_value, 'passed': p_value > alpha}
    except Exception as e:
        return {'test': 'Levene', 'passed': False, 'error': str(e)}