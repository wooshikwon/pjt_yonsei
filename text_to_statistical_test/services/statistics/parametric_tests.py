# 파일명: services/statistics/parametric_tests.py

import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from typing import Dict, Any, List

def run_independent_t_test(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
    """독립표본 t-검정을 수행합니다."""
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError("독립표본 t-검정은 정확히 두 개의 그룹이 필요합니다.")
    
    group1 = df[df[group_col] == groups[0]][value_col].dropna()
    group2 = df[df[group_col] == groups[1]][value_col].dropna()
    
    if len(group1) < 2 or len(group2) < 2:
        raise ValueError("t-검정을 수행하기에 각 그룹의 샘플 크기가 너무 작습니다.")

    # 참고: 등분산성(equal_var)은 stats_service에서 가정을 검토한 후 결정하여 전달할 수 있습니다.
    # 여기서는 기본값(True)을 사용합니다.
    stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
    
    return {
        'test_name': 'Independent Samples t-test',
        't_statistic': stat,
        'p_value': p_value,
        'degrees_of_freedom': len(group1) + len(group2) - 2
    }

def run_paired_t_test(df: pd.DataFrame, before_col: str, after_col: str) -> Dict[str, Any]:
    """대응표본 t-검정을 수행합니다."""
    before_data = df[before_col].dropna()
    after_data = df[after_col].dropna()

    if len(before_data) != len(after_data):
        raise ValueError("대응표본 t-검정을 위해 두 컬럼의 데이터 개수가 일치해야 합니다.")
    if len(before_data) < 2:
        raise ValueError("샘플 크기가 너무 작습니다.")
        
    stat, p_value = stats.ttest_rel(before_data, after_data)
    return {
        'test_name': 'Paired Samples t-test',
        't_statistic': stat,
        'p_value': p_value,
        'degrees_of_freedom': len(before_data) - 1
    }

def run_one_way_anova(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
    """일원분산분석(One-way ANOVA)을 수행합니다."""
    groups = df[group_col].dropna().unique()
    if len(groups) < 3:
        raise ValueError("ANOVA는 세 개 이상의 그룹이 필요합니다.")
        
    grouped_data = [df[value_col][df[group_col] == g].dropna() for g in groups]
    
    # 각 그룹의 샘플 크기 확인
    if any(len(g) < 2 for g in grouped_data):
        raise ValueError("ANOVA를 수행하기에 그룹별 샘플 크기가 너무 작습니다.")

    stat, p_value = stats.f_oneway(*grouped_data)
    return {
        'test_name': 'One-way ANOVA',
        'F_statistic': stat,
        'p_value': p_value
    }

def run_two_way_anova(df: pd.DataFrame, dependent_var: str, independent_vars: List[str]) -> Dict[str, Any]:
    """이원분산분석(Two-way ANOVA)을 수행합니다."""
    if len(independent_vars) != 2:
        raise ValueError("이원분산분석은 정확히 두 개의 독립변수가 필요합니다.")
    
    iv1, iv2 = independent_vars[0], independent_vars[1]
    
    # statsmodels를 사용한 ANOVA 수행
    formula = f"`{dependent_var}` ~ C(`{iv1}`) * C(`{iv2}`)"
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # [UTIL-REQ] statsmodels의 anova_table(DataFrame)을 깔끔한 dict로 파싱하는 함수가 utils.helpers에 필요합니다.
    # 예시: anova_results = parse_anova_table(anova_table)
    anova_results = anova_table.reset_index().rename(columns={'index': 'source'}).to_dict('records')

    return {
        'test_name': 'Two-way ANOVA',
        'formula': formula,
        'results_table': anova_results
    }