# services/statistics/functions/posthoc.py
"""
사후 분석(Post-hoc Analysis) 함수 모음.

이 모듈의 함수들은 분산 분석(ANOVA) 등에서 통계적으로 유의미한 차이가
발견되었을 때, 구체적으로 어떤 그룹 간에 차이가 있는지를 식별하는
순수 계산 로직을 담당합니다.
"""

import pandas as pd
from statsmodels.stats.multicomp import MultiComparison
from typing import Dict, Any

def run_tukey_hsd(df: pd.DataFrame, group_col: str, value_col: str, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Tukey's Honestly Significant Difference (HSD) 사후 검정을 수행합니다.
    """
    valid_df = df[[group_col, value_col]].dropna()
    if valid_df[group_col].nunique() < 2:
        raise ValueError("Tukey HSD 사후 검정을 수행하려면 두 개 이상의 그룹이 필요합니다.")
        
    mc = MultiComparison(valid_df[value_col], valid_df[group_col])
    tukey_result = mc.tukeyhsd(alpha=alpha)
    
    # 결과를 사용하기 쉬운 DataFrame으로 변환 후 딕셔너리로 반환
    result_df = pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])
    
    return {'test_name': 'Tukey HSD Post-hoc Test', 'results_table': result_df.to_dict('records')}