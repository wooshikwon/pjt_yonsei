# 파일명: services/statistics/posthoc_analysis.py

import pandas as pd
from typing import Dict, Any
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison

def run_tukey_hsd(
    df: pd.DataFrame, group_col: str, value_col: str, alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Tukey HSD (Honest Significant Difference) 사후 검정을 수행합니다.
    :param df: 분석에 사용할 DataFrame
    :param group_col: 그룹화할 범주형 변수
    :param value_col: 비교할 연속형 변수
    :param alpha: 유의수준 (기본값 0.05)
    :return: 그룹 간 비교 결과를 담은 dict
    """
    # NA 제거
    valid_df = df[[group_col, value_col]].dropna()
    if valid_df[group_col].nunique() < 2:
        raise ValueError("Tukey HSD 사후 검정을 수행하려면 두 개 이상의 그룹이 필요합니다.")
    if valid_df.shape[0] < 3:
        raise ValueError("데이터가 충분하지 않아 Tukey HSD 사후 검정을 수행할 수 없습니다.")

    # MultiComparison 객체 생성
    mc = MultiComparison(valid_df[value_col], valid_df[group_col])
    tukey_result = mc.tukeyhsd(alpha=alpha)

    # 결과를 pandas DataFrame으로 변환
    result_df = pd.DataFrame(data=tukey_result._results_table.data[1:],  # 첫 번째 행은 header
                             columns=tukey_result._results_table.data[0])  # header

    return {
        'test_name': 'Tukey HSD Post-hoc Test',
        'summary': tukey_result.summary().as_text(),
        'results_table': result_df.to_dict('records')
    }


def run_mann_whitney_posthoc(
    df: pd.DataFrame, group_col: str, value_col: str, alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Mann-Whitney U 검정 기반 다중 비교(비모수 사후 검정) 예시:
    Bonferroni 교정을 적용하여 그룹 쌍별로 Mann-Whitney U 검정을 수행합니다.
    :param df: 분석에 사용할 DataFrame
    :param group_col: 그룹화할 범주형 변수
    :param value_col: 비교할 연속형 변수
    :param alpha: 총 유효 유의수준
    :return: 각 그룹 쌍별 U 통계량과 보정된 p-value를 담은 dict
    """
    from itertools import combinations
    results = []

    # NA 제거
    valid_df = df[[group_col, value_col]].dropna()
    groups = valid_df[group_col].unique().tolist()
    if len(groups) < 2:
        raise ValueError("비모수 사후 검정을 수행하려면 두 개 이상의 그룹이 필요합니다.")

    # pairwise 조합 생성
    pair_list = list(combinations(groups, 2))
    m = len(pair_list)
    for (g1, g2) in pair_list:
        data1 = valid_df[valid_df[group_col] == g1][value_col]
        data2 = valid_df[valid_df[group_col] == g2][value_col]
        if data1.empty or data2.empty:
            continue

        stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        # Bonferroni 보정
        adj_p = min(p_value * m, 1.0)
        results.append({
            'group1': g1,
            'group2': g2,
            'U_statistic': stat,
            'raw_p_value': p_value,
            'adjusted_p_value': adj_p
        })

    return {
        'test_name': 'Pairwise Mann-Whitney U (Bonferroni-corrected)',
        'comparisons': results
    }
