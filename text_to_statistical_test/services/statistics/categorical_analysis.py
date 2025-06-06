# 파일명: services/statistics/categorical_analysis.py

import pandas as pd
from scipy import stats
from typing import Dict, Any

def run_chi_square_independence(
    df: pd.DataFrame, col1: str, col2: str
) -> Dict[str, Any]:
    """
    카이제곱 독립성 검정을 수행합니다.
    :param df: 분석에 사용할 DataFrame
    :param col1: 첫 번째 범주형 변수
    :param col2: 두 번째 범주형 변수
    :return: chi2 통계량, p-value, 자유도, 기대도표(expected frequencies) 등을 담은 dict
    """
    # NA 제거
    valid_df = df[[col1, col2]].dropna()
    if valid_df.shape[0] < 5:
        raise ValueError("데이터가 충분하지 않아 카이제곱 독립성 검정을 수행할 수 없습니다.")

    # 교차표 생성
    contingency_table = pd.crosstab(valid_df[col1], valid_df[col2])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    # 기대도표를 DataFrame으로 변환
    expected_df = pd.DataFrame(
        expected,
        index=contingency_table.index,
        columns=contingency_table.columns
    )

    return {
        'test_name': 'Chi-Square Independence Test',
        'chi2_statistic': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'observed_table': contingency_table.to_dict(),
        'expected_table': expected_df.to_dict()
    }
