# 파일명: services/statistics/nonparametric_tests.py

import pandas as pd
from scipy import stats
from typing import Dict, Any

def run_mann_whitney_u(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
    """Mann-Whitney U 검정을 수행합니다. (독립표본 t-검정의 비모수 버전)"""
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError("Mann-Whitney U 검정은 정확히 두 개의 그룹이 필요합니다.")

    group1 = df[df[group_col] == groups[0]][value_col].dropna()
    group2 = df[df[group_col] == groups[1]][value_col].dropna()

    if len(group1) == 0 or len(group2) == 0:
        raise ValueError("데이터가 없는 그룹이 있어 검정을 수행할 수 없습니다.")
        
    stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    return {
        'test_name': 'Mann-Whitney U Test',
        'U_statistic': stat,
        'p_value': p_value
    }

def run_kruskal_wallis(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
    """Kruskal-Wallis H 검정을 수행합니다. (일원분산분석의 비모수 버전)"""
    groups = df[group_col].dropna().unique()
    if len(groups) < 3:
        raise ValueError("Kruskal-Wallis 검정은 세 개 이상의 그룹이 필요합니다.")
        
    grouped_data = [df[value_col][df[group_col] == g].dropna() for g in groups]
    
    # 비어있는 그룹 제거
    cleaned_groups = [g for g in grouped_data if not g.empty]
    if len(cleaned_groups) < 2:
         raise ValueError("검정을 수행하기에 유효한 그룹 수가 부족합니다.")

    stat, p_value = stats.kruskal(*cleaned_groups)
    return {
        'test_name': 'Kruskal-Wallis H Test',
        'H_statistic': stat,
        'p_value': p_value
    }
    
def run_wilcoxon_signed_rank(df: pd.DataFrame, before_col: str, after_col: str) -> Dict[str, Any]:
    """Wilcoxon 부호-순위 검정을 수행합니다. (대응표본 t-검정의 비모수 버전)"""
    before_data = df[before_col].dropna()
    after_data = df[after_col].dropna()

    if len(before_data) != len(after_data):
        raise ValueError("Wilcoxon 부호-순위 검정을 위해 두 컬럼의 데이터 개수가 일치해야 합니다.")
    if len(before_data) < 1:
        raise ValueError("샘플 크기가 너무 작습니다.")

    # 0인 차이는 제외하고 계산
    diff = before_data - after_data
    diff = diff[diff != 0]

    if len(diff) == 0:
        return {
            'test_name': 'Wilcoxon Signed-Rank Test',
            'W_statistic': 0,
            'p_value': 1.0,
            'note': '모든 차이가 0이므로 검정을 수행할 수 없습니다.'
        }
        
    stat, p_value = stats.wilcoxon(diff)
    return {
        'test_name': 'Wilcoxon Signed-Rank Test',
        'W_statistic': stat,
        'p_value': p_value
    }