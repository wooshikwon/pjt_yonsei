# 파일명: services/statistics/assumption_checks.py

import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, Any, List


def check_normality(data: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
    """샤피로-윌크 검정으로 데이터의 정규성을 확인합니다."""
    non_na = data.dropna()
    if len(non_na) < 3:
        return {
            'test': 'Shapiro-Wilk',
            'passed': False,
            'p_value': 0.0,
            'reason': '샘플 크기가 3 미만입니다.'
        }

    try:
        stat, p_value = stats.shapiro(non_na)
        return {
            'test': 'Shapiro-Wilk',
            'statistic': stat,
            'p_value': p_value,
            'passed': p_value > alpha
        }
    except Exception as e:
        return {
            'test': 'Shapiro-Wilk',
            'passed': False,
            'error': str(e)
        }


def check_equal_variance(*groups: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
    """레빈 검정으로 그룹 간 등분산성을 확인합니다."""
    # 각 그룹별로 NA 제거 후 샘플 수 확인
    cleaned = [g.dropna() for g in groups if len(g.dropna()) >= 1]
    if any(len(g.dropna()) < 3 for g in groups):
        return {
            'test': 'Levene',
            'passed': False,
            'p_value': 0.0,
            'reason': '그룹별 샘플 크기가 3 미만입니다.'
        }
    if len(cleaned) < 2:
        return {
            'test': 'Levene',
            'passed': True,
            'reason': '비교할 그룹이 1개 이하입니다.'
        }

    try:
        stat, p_value = stats.levene(*cleaned)
        return {
            'test': 'Levene',
            'statistic': stat,
            'p_value': p_value,
            'passed': p_value > alpha
        }
    except Exception as e:
        return {
            'test': 'Levene',
            'passed': False,
            'error': str(e)
        }


def check_multicollinearity(
    df: pd.DataFrame, independent_vars: List[str]
) -> Dict[str, Any]:
    """VIF(분산 팽창 인수)를 사용하여 다중공선성을 확인합니다."""
    if len(independent_vars) < 2:
        return {
            'test': 'VIF',
            'passed': True,
            'reason': '독립변수가 2개 미만입니다.'
        }

    try:
        X = df[independent_vars].copy()
        X['const'] = 1

        vif_data = pd.DataFrame({
            "feature": [col for col in X.columns if col != 'const'],
            "VIF": [
                variance_inflation_factor(X.values, i)
                for i, col in enumerate(X.columns)
                if col != 'const'
            ]
        })
        max_vif = float(vif_data["VIF"].max())
        passed = max_vif < 10  # 일반적으로 VIF >= 10이면 다중공선성 위험

        return {
            'test': 'VIF',
            'max_vif': max_vif,
            'passed': passed,
            'details': vif_data.to_dict('records')
        }
    except Exception as e:
        return {
            'test': 'VIF',
            'passed': False,
            'error': str(e)
        }
