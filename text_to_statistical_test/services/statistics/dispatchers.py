# services/statistics/dispatchers.py
"""
서비스 레이어에서 사용할 디스패처(매핑 정보)를 정의합니다.

이 모듈은 functions 폴더의 순수 함수들을 임포트하여,
외부에서 전달되는 test_id와 같은 식별자를 실제 실행 가능한 함수와 연결하는
딕셔너리(매핑 테이블)를 생성하고 제공합니다.
"""
import pandas as pd
from .functions import analysis, assumptions, effect_sizes, posthoc

# --- 핵심 통계 분석 함수 매핑 (변경 없음) ---
TEST_DISPATCHER = {
    # Parametric Tests
    'independent_t_test': {'function': analysis.run_independent_t_test, 'required_params': ['group_col', 'value_col'], 'assumptions': ['normality', 'homoscedasticity'], 'takes_dataframe': True},
    'paired_t_test': {'function': analysis.run_paired_t_test, 'required_params': ['before_col', 'after_col'], 'assumptions': ['normality'], 'takes_dataframe': True},
    'one_way_anova': {'function': analysis.run_one_way_anova, 'required_params': ['group_col', 'value_col'], 'assumptions': ['normality', 'homoscedasticity'], 'takes_dataframe': True},
    'two_way_anova': {'function': analysis.run_two_way_anova, 'required_params': ['dependent_var', 'independent_vars'], 'assumptions': [], 'takes_dataframe': True},

    # Categorical Analysis
    'chi_square_independence': {'function': analysis.run_chi_square_independence, 'required_params': ['col1', 'col2'], 'assumptions': [], 'takes_dataframe': True},
    'chi_square_goodness_of_fit': {'function': analysis.run_chi_square_goodness_of_fit, 'required_params': ['observed', 'expected'], 'assumptions': [], 'takes_dataframe': False},
    'logistic_regression': {'function': analysis.run_logistic_regression, 'required_params': ['dependent_var', 'independent_vars'], 'assumptions': [], 'takes_dataframe': True},
    
    # Correlation Analysis
    'correlation': {'function': analysis.run_correlation_analysis, 'required_params': ['vars', 'method'], 'assumptions': [], 'takes_dataframe': True},

    # Proportion Tests
    'one_proportion_test': {'function': analysis.run_one_proportion_test, 'required_params': ['count', 'nobs', 'p0'], 'assumptions': [], 'takes_dataframe': False},
    'two_proportion_test': {'function': analysis.run_two_proportion_test, 'required_params': ['count', 'nobs'], 'assumptions': [], 'takes_dataframe': False},

    # Regression Analysis
    'linear_regression': {'function': analysis.run_linear_regression, 'required_params': ['dependent_var', 'independent_vars'], 'assumptions': [], 'takes_dataframe': True},
    'multiple_linear_regression': {'function': analysis.run_linear_regression, 'required_params': ['dependent_var', 'independent_vars'], 'assumptions': [], 'takes_dataframe': True}, # 별칭

    # Non-parametric Tests
    'mann_whitney_u': {'function': analysis.run_mann_whitney_u, 'required_params': ['group_col', 'value_col'], 'assumptions': [], 'takes_dataframe': True},
    'kruskal_wallis': {'function': analysis.run_kruskal_wallis, 'required_params': ['group_col', 'value_col'], 'assumptions': [], 'takes_dataframe': True},
    'wilcoxon_signed_rank': {'function': analysis.run_wilcoxon_signed_rank, 'required_params': ['before_col', 'after_col'], 'assumptions': [], 'takes_dataframe': True},
}


# [!!!] 효과 크기 파라미터 준비 함수들 정의
def _prepare_cohens_d_params(data, params):
    groups = data[params['group_col']].unique()
    return {
        'group1': data[data[params['group_col']] == groups[0]][params['value_col']],
        'group2': data[data[params['group_col']] == groups[1]][params['value_col']]
    }

def _prepare_eta_squared_params(data, params):
    test_results = params.get('test_results', {}).get('test_results', {})
    df_between = data[params['group_col']].nunique() - 1
    df_within = len(data) - data[params['group_col']].nunique()
    return {
        'f_statistic': test_results.get('F_statistic'),
        'df_between': df_between,
        'df_within': df_within
    }
    
def _prepare_partial_eta_squared_params(data, params):
    test_results = params.get('test_results', {}).get('test_results', {})
    return {'anova_table': test_results.get('anova_table')}
    
def _prepare_cramers_v_params(data, params):
    test_results = params.get('test_results', {}).get('test_results', {})
    return {
        'chi2_stat': test_results.get('chi2_statistic'),
        'n': len(data),
        'contingency_table': pd.crosstab(data[params['col1']], data[params['col2']])
    }

def _prepare_cohens_h_params(data, params):
    test_results = params.get('test_results', {}).get('test_results', {})
    proportions = test_results.get('_internal_proportions')
    if not proportions or len(proportions) != 2:
         raise ValueError("두 표본 비율 검사의 효과 크기를 계산하기 위한 비율(proportion) 값이 없습니다.")
    return {'prop1': proportions[0], 'prop2': proportions[1]}


# [!!!] 수정된 효과 크기 계산 함수 매핑
EFFECT_SIZE_DISPATCHER = {
    'independent_t_test': {
        'function': effect_sizes.calculate_cohens_d, 
        'preparer': _prepare_cohens_d_params
    },
    'one_way_anova': {
        'function': effect_sizes.calculate_eta_squared, 
        'preparer': _prepare_eta_squared_params
    },
    'two_way_anova': {
        'function': effect_sizes.calculate_partial_eta_squared,
        'preparer': _prepare_partial_eta_squared_params
    },
    'chi_square_independence': {
        'function': effect_sizes.calculate_cramers_v,
        'preparer': _prepare_cramers_v_params
    },
    'two_proportion_test': {
        'function': effect_sizes.calculate_cohens_h,
        'preparer': _prepare_cohens_h_params
    },
    'logistic_regression': {
        'function': lambda test_results: {'odds_ratios': test_results.get('odds_ratios')},
        'preparer': lambda data, params: {'test_results': params.get('test_results', {}).get('test_results', {})}
    },
}

# --- 사후 분석 함수 매핑 (변경 없음) ---
POSTHOC_DISPATCHER = {
    'one_way_anova': posthoc.run_tukey_hsd,
    'two_way_anova': posthoc.run_tukey_hsd,
}

# --- 가정 검토 함수 매핑 (변경 없음) ---
ASSUMPTION_CHECKER_MAP = {
    'normality': assumptions.check_normality,
    'homoscedasticity': assumptions.check_equal_variance,
}