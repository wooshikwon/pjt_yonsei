# 파일명: services/statistics/stats_service.py
"""
Statistics Service

모든 통계 분석, 가정 검토, 사후 분석, 효과 크기 계산 기능을 단일 파일로 통합하여 제공합니다.
이 설계는 복잡한 내부 모듈 의존성을 제거하고 순환 참조 문제를 원천적으로 해결하기 위해 채택되었습니다.
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, Any, List, Callable, Union
import logging

from utils import StatisticalException

# ==============================================================================
# 통계 테스트 함수들 (Parametric Tests)
# ==============================================================================

def run_independent_t_test(df: pd.DataFrame, group_col: str, value_col: str, equal_var: bool = True) -> Dict[str, Any]:
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError("독립표본 t-검정은 정확히 두 개의 그룹이 필요합니다.")
    group1 = df[df[group_col] == groups[0]][value_col].dropna()
    group2 = df[df[group_col] == groups[1]][value_col].dropna()
    if len(group1) < 2 or len(group2) < 2:
        raise ValueError("t-검정을 수행하기에 각 그룹의 샘플 크기가 너무 작습니다.")
    stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
    test_name = 'Independent Samples t-test' if equal_var else "Welch's t-test"
    return {'test_name': test_name, 't_statistic': stat, 'p_value': p_value, 'degrees_of_freedom': len(group1) + len(group2) - 2}

def run_paired_t_test(df: pd.DataFrame, before_col: str, after_col: str) -> Dict[str, Any]:
    before_data = df[before_col].dropna()
    after_data = df[after_col].dropna()
    if len(before_data) != len(after_data):
        raise ValueError("대응표본 t-검정을 위해 두 컬럼의 데이터 개수가 일치해야 합니다.")
    if len(before_data) < 2:
        raise ValueError("샘플 크기가 너무 작습니다.")
    stat, p_value = stats.ttest_rel(before_data, after_data)
    return {'test_name': 'Paired Samples t-test', 't_statistic': stat, 'p_value': p_value, 'degrees_of_freedom': len(before_data) - 1}

def run_one_way_anova(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
    groups = df[group_col].dropna().unique()
    if len(groups) < 3:
        raise ValueError("ANOVA는 세 개 이상의 그룹이 필요합니다.")
    grouped_data = [df[value_col][df[group_col] == g].dropna() for g in groups]
    if any(len(g) < 2 for g in grouped_data):
        raise ValueError("ANOVA를 수행하기에 그룹별 샘플 크기가 너무 작습니다.")
    stat, p_value = stats.f_oneway(*grouped_data)
    return {'test_name': 'One-way ANOVA', 'F_statistic': stat, 'p_value': p_value}

def run_two_way_anova(df: pd.DataFrame, dependent_var: str, independent_vars: List[str]) -> Dict[str, Any]:
    """이원 분산 분석을 수행합니다."""
    if len(independent_vars) != 2:
        raise ValueError("이원 분산 분석은 정확히 두 개의 독립 변수가 필요합니다.")
    
    # Patsy 포뮬러에서 변수 이름에 특수 문자가 있어도 안전하게 처리하기 위해 Q() 사용
    formula = f"Q('{dependent_var}') ~ C(Q('{independent_vars[0]}')) * C(Q('{independent_vars[1]}'))"
    
    model = ols(formula, data=df.rename(columns=lambda x: x.replace("'", ""))).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # 결과를 보기 쉬운 형식으로 재구성
    results = {
        'test_name': 'Two-Way ANOVA',
        'formula': formula,
        'anova_table': anova_table.to_dict('index')
    }
    return results

# ==============================================================================
# 통계 테스트 함수들 (Categorical Analysis)
# ==============================================================================

def run_chi_square_independence(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    valid_df = df[[col1, col2]].dropna()
    if valid_df.shape[0] < 5:
        raise ValueError("데이터가 충분하지 않아 카이제곱 독립성 검정을 수행할 수 없습니다.")
    contingency_table = pd.crosstab(valid_df[col1], valid_df[col2])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
    return {'test_name': 'Chi-Square Independence Test', 'chi2_statistic': chi2, 'p_value': p_value, 'degrees_of_freedom': dof, 'observed_table': contingency_table.to_dict(), 'expected_table': expected_df.to_dict()}

def run_chi_square_goodness_of_fit(observed: List[int], expected: List[int] = None) -> Dict[str, Any]:
    """카이제곱 적합도 검정을 수행합니다."""
    if sum(observed) == 0:
        raise ValueError("관측값의 총합은 0보다 커야 합니다.")
    stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
    return {'test_name': 'Chi-Square Goodness-of-Fit Test', 'chi2_statistic': stat, 'p_value': p_value}

def run_logistic_regression(df: pd.DataFrame, dependent_var: str, independent_vars: List[str]) -> Dict[str, Any]:
    """로지스틱 회귀 분석을 수행합니다."""
    cols = [dependent_var] + independent_vars
    valid_df = df[cols].dropna()

    if valid_df[dependent_var].nunique() != 2:
        raise ValueError("종속 변수는 두 개의 고유한 값을 가져야 합니다 (이진 분류).")
    
    X = valid_df[independent_vars]
    y = valid_df[dependent_var]
    X_const = sm.add_constant(X)
    
    model = sm.Logit(y, X_const).fit()
    
    # 결과 구성
    summary = model.summary2().tables[1]
    coef_table = summary.to_dict('records')
    
    # Odds Ratios 계산
    odds_ratios = np.exp(model.params)
    
    return {
        'test_name': 'Logistic Regression',
        'formula': f"{dependent_var} ~ {' + '.join(independent_vars)}",
        'model_summary': {
            'pseudo_r_squared': model.prsquared,
            'log_likelihood': model.llf,
            'll_null': model.llnull,
            'p_value_ll_ratio': model.llr_pvalue,
            'n_observations': int(model.nobs)
        },
        'coefficients': coef_table,
        'odds_ratios': odds_ratios.to_dict()
    }

def run_correlation_analysis(df: pd.DataFrame, vars: List[str], method: str = 'pearson') -> Dict[str, Any]:
    """상관 분석을 수행합니다."""
    if len(vars) < 2:
        raise ValueError("상관 분석을 위해서는 최소 두 개 이상의 변수가 필요합니다.")
    
    valid_df = df[vars].dropna()
    
    if not np.issubdtype(valid_df.to_numpy().dtype, np.number):
         raise ValueError("상관 분석은 수치형 데이터에만 적용할 수 있습니다.")

    corr_matrix = valid_df.corr(method=method)
    
    # p-value 매트릭스 계산
    p_value_matrix = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns, dtype=float)
    for r_idx, row in enumerate(corr_matrix.index):
        for c_idx, col in enumerate(corr_matrix.columns):
            if r_idx >= c_idx:
                p_value_matrix.loc[row, col] = np.nan
                continue
            # scipy.stats.pearsonr은 두 개의 1D 배열을 인자로 받습니다.
            corr_test = stats.pearsonr(valid_df[row], valid_df[col])
            p_value_matrix.loc[row, col] = corr_test[1]

    return {
        'test_name': f'{method.capitalize()} Correlation Analysis',
        'correlation_matrix': corr_matrix.to_dict(),
        'p_value_matrix': p_value_matrix.to_dict()
    }

# ==============================================================================
# 통계 테스트 함수들 (Proportion Tests)
# ==============================================================================

def run_one_proportion_test(count: int, nobs: int, p0: float) -> Dict[str, Any]:
    """단일 표본 비율 검정을 수행합니다."""
    if count > nobs:
        raise ValueError("성공 횟수(count)는 전체 관측치 수(nobs)보다 클 수 없습니다.")
    stat, p_value = sm.stats.proportions_ztest(count, nobs, value=p0)
    return {'test_name': 'One-Proportion Z-Test', 'z_statistic': stat, 'p_value': p_value}

def run_two_proportion_test(count: List[int], nobs: List[int]) -> Dict[str, Any]:
    """두 표본 비율 검정을 수행합니다."""
    if len(count) != 2 or len(nobs) != 2:
        raise ValueError("count와 nobs는 각각 두 개의 원소를 가진 리스트여야 합니다.")
    if any(c > n for c, n in zip(count, nobs)):
        raise ValueError("각 그룹의 성공 횟수는 전체 관측치 수보다 클 수 없습니다.")
    stat, p_value = sm.stats.proportions_ztest(count, nobs)
    return {'test_name': 'Two-Proportion Z-Test', 'z_statistic': stat, 'p_value': p_value}

# ==============================================================================
# 통계 테스트 함수들 (Regression Analysis)
# ==============================================================================

def run_linear_regression(df: pd.DataFrame, dependent_var: str, independent_vars: List[str]) -> Dict[str, Any]:
    """
    다중 선형 회귀 분석을 수행하고, 주요 가정 검토 결과를 함께 반환합니다.
    - 모델 요약 (R-squared, 계수 등)
    - 잔차 정규성 (Jarque-Bera test)
    - 잔차 등분산성 (Breusch-Pagan test)
    - 다중공선성 (VIF)
    """
    cols = [dependent_var] + independent_vars
    valid_df = df[cols].dropna()
    if valid_df.shape[0] < len(independent_vars) + 2:
        raise ValueError("데이터가 충분하지 않아 다중 선형회귀분석을 수행할 수 없습니다.")
    
    X = valid_df[independent_vars]
    y = valid_df[dependent_var]
    X_const = sm.add_constant(X, has_constant='add')
    
    model = sm.OLS(y, X_const).fit()
    
    # 가정 검토
    residuals = model.resid
    
    # 1. 잔차 정규성 (Jarque-Bera test)
    jb_stat, jb_p_value, _, _ = sm.stats.jarque_bera(residuals)
    
    # 2. 잔차 등분산성 (Breusch-Pagan test)
    bp_test = sm.stats.het_breuschpagan(residuals, model.model.exog)
    
    # 3. 다중공선성 (VIF)
    vif_results = {}
    if X.shape[1] > 1: # 독립변수가 2개 이상일 때만 VIF 계산
        vif_results = {var: variance_inflation_factor(X_const.values, i) 
                       for i, var in enumerate(X_const.columns) if var != 'const'}

    # 결과 구성
    summary = model.summary2().tables[1]
    coef_table = summary.to_dict('records')
    
    return {
        'test_name': 'Multiple Linear Regression',
        'formula': f"{dependent_var} ~ {' + '.join(independent_vars)}",
        'model_summary': {
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': float(model.fvalue),
            'f_pvalue': float(model.f_pvalue),
            'df_resid': int(model.df_resid),
            'n_observations': int(model.nobs)
        },
        'coefficients': coef_table,
        'assumption_checks': {
            'residual_normality': {
                'test': 'Jarque-Bera',
                'statistic': jb_stat,
                'p_value': jb_p_value,
                'passed': jb_p_value > 0.05
            },
            'homoscedasticity': {
                'test': 'Breusch-Pagan',
                'lagrange_multiplier': bp_test[0],
                'p_value': bp_test[1],
                'passed': bp_test[1] > 0.05
            },
            'multicollinearity': {
                'test': 'VIF',
                'vif_values': vif_results,
                'note': 'VIF > 10 이면 다중공선성을 의심할 수 있습니다. (독립변수가 1개일 때는 계산되지 않습니다.)'
            }
        }
    }

# ==============================================================================
# 통계 테스트 함수들 (Non-parametric Tests)
# ==============================================================================

def run_mann_whitney_u(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError("Mann-Whitney U 검정은 정확히 두 개의 그룹이 필요합니다.")
    group1 = df[df[group_col] == groups[0]][value_col].dropna()
    group2 = df[df[group_col] == groups[1]][value_col].dropna()
    if len(group1) == 0 or len(group2) == 0:
        raise ValueError("데이터가 없는 그룹이 있어 검정을 수행할 수 없습니다.")
    stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    return {'test_name': 'Mann-Whitney U Test', 'U_statistic': stat, 'p_value': p_value}

def run_kruskal_wallis(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
    groups = df[group_col].dropna().unique()
    if len(groups) < 3:
        raise ValueError("Kruskal-Wallis 검정은 세 개 이상의 그룹이 필요합니다.")
    grouped_data = [df[value_col][df[group_col] == g].dropna() for g in groups]
    cleaned_groups = [g for g in grouped_data if not g.empty]
    if len(cleaned_groups) < 2:
         raise ValueError("검정을 수행하기에 유효한 그룹 수가 부족합니다.")
    stat, p_value = stats.kruskal(*cleaned_groups)
    return {'test_name': 'Kruskal-Wallis H Test', 'H_statistic': stat, 'p_value': p_value}

def run_wilcoxon_signed_rank(df: pd.DataFrame, before_col: str, after_col: str) -> Dict[str, Any]:
    before_data = df[before_col].dropna()
    after_data = df[after_col].dropna()
    if len(before_data) != len(after_data):
        raise ValueError("Wilcoxon 부호-순위 검정을 위해 두 컬럼의 데이터 개수가 일치해야 합니다.")
    if len(before_data) < 1:
        raise ValueError("샘플 크기가 너무 작습니다.")
    diff = before_data - after_data
    diff = diff[diff != 0]
    if len(diff) == 0:
        return {'test_name': 'Wilcoxon Signed-Rank Test', 'W_statistic': 0, 'p_value': 1.0, 'note': '모든 차이가 0이므로 검정을 수행할 수 없습니다.'}
    stat, p_value = stats.wilcoxon(diff)
    return {'test_name': 'Wilcoxon Signed-Rank Test', 'W_statistic': stat, 'p_value': p_value}


# ==============================================================================
# 가정 검토 함수 (Assumption Checks)
# ==============================================================================

def check_normality(data: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
    non_na = data.dropna()
    if len(non_na) < 3:
        return {'test': 'Shapiro-Wilk', 'passed': False, 'p_value': 0.0, 'reason': '샘플 크기가 3 미만입니다.'}
    try:
        stat, p_value = stats.shapiro(non_na)
        return {'test': 'Shapiro-Wilk', 'statistic': stat, 'p_value': p_value, 'passed': p_value > alpha}
    except Exception as e:
        return {'test': 'Shapiro-Wilk', 'passed': False, 'error': str(e)}

def check_equal_variance(*groups: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
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

# ==============================================================================
# 사후 분석 함수 (Post-hoc Analysis)
# ==============================================================================

def run_tukey_hsd(df: pd.DataFrame, group_col: str, value_col: str, alpha: float = 0.05) -> Dict[str, Any]:
    valid_df = df[[group_col, value_col]].dropna()
    if valid_df[group_col].nunique() < 2:
        raise ValueError("Tukey HSD 사후 검정을 수행하려면 두 개 이상의 그룹이 필요합니다.")
    mc = MultiComparison(valid_df[value_col], valid_df[group_col])
    tukey_result = mc.tukeyhsd(alpha=alpha)
    result_df = pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])
    return {'test_name': 'Tukey HSD Post-hoc Test', 'results_table': result_df.to_dict('records')}

# ==============================================================================
# 효과 크기 계산 함수 (Effect Size)
# ==============================================================================

def calculate_cohens_d(group1: pd.Series, group2: pd.Series) -> Dict[str, float]:
    n1, n2 = len(group1), len(group2)
    s1, s2 = group1.var(ddof=1), group2.var(ddof=1)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = group1.mean(), group2.mean()
    cohen_d = (u1 - u2) / s
    return {"Cohen's d": cohen_d}

def calculate_partial_eta_squared(anova_table: Dict[str, Any]) -> Dict[str, Any]:
    """이원 분산 분석 결과에서 부분 에타 제곱을 계산합니다."""
    partial_eta_sq = {}
    for factor, values in anova_table.items():
        ss_effect = values.get('sum_sq')
        ss_error = anova_table.get('Residual', {}).get('sum_sq')
        if ss_effect is not None and ss_error is not None:
            partial_eta_sq[factor] = ss_effect / (ss_effect + ss_error)
    return {'partial_eta_squared': partial_eta_sq}

def calculate_eta_squared(f_statistic: float, df_between: int, df_within: int) -> Dict[str, float]:
    eta_sq = (f_statistic * df_between) / (f_statistic * df_between + df_within)
    return {"Eta-squared": eta_sq}

def calculate_cramers_v(chi2_stat: float, n: int, contingency_table: pd.DataFrame) -> Dict[str, float]:
    min_dim = min(contingency_table.shape) - 1
    if min_dim == 0: return {"Cramer's V": 0.0}
    cramers_v = np.sqrt(chi2_stat / (n * min_dim))
    return {"Cramer's V": cramers_v}

def calculate_cohens_h(prop1: float, prop2: float) -> Dict[str, float]:
    """두 비율에 대한 Cohen's h 효과 크기를 계산합니다."""
    h = 2 * np.arcsin(np.sqrt(prop1)) - 2 * np.arcsin(np.sqrt(prop2))
    return {"Cohen's h": h}

# ==============================================================================
# Statistics Service Facade Class
# ==============================================================================

logger = logging.getLogger(__name__)

class StatisticsService:
    """
    통계 분석 기능을 통합 제공하는 서비스 (Facade).
    모든 통계 함수를 내부에 포함하여 외부 의존성을 최소화합니다.
    """
    def __init__(self):
        self._test_dispatcher = {
            # 데이터프레임 기반 함수
            'independent_t_test': {'function': run_independent_t_test, 'required_params': ['group_col', 'value_col'], 'assumptions': ['normality', 'homoscedasticity'], 'takes_dataframe': True},
            'paired_t_test': {'function': run_paired_t_test, 'required_params': ['before_col', 'after_col'], 'assumptions': ['normality'], 'takes_dataframe': True},
            'one_way_anova': {'function': run_one_way_anova, 'required_params': ['group_col', 'value_col'], 'assumptions': ['normality', 'homoscedasticity'], 'takes_dataframe': True},
            'two_way_anova': {'function': run_two_way_anova, 'required_params': ['dependent_var', 'independent_vars'], 'assumptions': [], 'takes_dataframe': True},
            'chi_square_independence': {'function': run_chi_square_independence, 'required_params': ['col1', 'col2'], 'assumptions': [], 'takes_dataframe': True},
            'linear_regression': {'function': run_linear_regression, 'required_params': ['dependent_var', 'independent_vars'], 'assumptions': [], 'takes_dataframe': True},
            'multiple_linear_regression': {'function': run_linear_regression, 'required_params': ['dependent_var', 'independent_vars'], 'assumptions': [], 'takes_dataframe': True}, # 별칭
            'mann_whitney_u': {'function': run_mann_whitney_u, 'required_params': ['group_col', 'value_col'], 'assumptions': [], 'takes_dataframe': True},
            'kruskal_wallis': {'function': run_kruskal_wallis, 'required_params': ['group_col', 'value_col'], 'assumptions': [], 'takes_dataframe': True},
            'wilcoxon_signed_rank': {'function': run_wilcoxon_signed_rank, 'required_params': ['before_col', 'after_col'], 'assumptions': [], 'takes_dataframe': True},
            
            # 파라미터 기반 함수
            'chi_square_goodness_of_fit': {'function': run_chi_square_goodness_of_fit, 'required_params': ['observed', 'expected'], 'assumptions': [], 'takes_dataframe': False},
            'one_proportion_test': {'function': run_one_proportion_test, 'required_params': ['count', 'nobs', 'p0'], 'assumptions': [], 'takes_dataframe': False},
            'two_proportion_test': {'function': run_two_proportion_test, 'required_params': ['count', 'nobs'], 'assumptions': [], 'takes_dataframe': False},
            'logistic_regression': {'function': run_logistic_regression, 'required_params': ['dependent_var', 'independent_vars'], 'assumptions': [], 'takes_dataframe': True},
            'correlation': {'function': run_correlation_analysis, 'required_params': ['vars', 'method'], 'assumptions': [], 'takes_dataframe': True},
        }
        self._effect_size_dispatcher = {
            'independent_t_test': {'function': calculate_cohens_d, 'required_params': ['group1', 'group2']},
            'one_way_anova': {'function': calculate_eta_squared, 'required_params': ['f_statistic', 'df_between', 'df_within']},
            'two_way_anova': {'function': calculate_partial_eta_squared, 'required_params': ['anova_table']},
            'chi_square_independence': {'function': calculate_cramers_v, 'required_params': ['chi2_stat', 'n', 'contingency_table']},
            'two_proportion_test': {'function': calculate_cohens_h, 'required_params': ['prop1', 'prop2']},
            'logistic_regression': {'function': lambda test_results: {'odds_ratios': test_results.get('odds_ratios')}, 'required_params': ['test_results']},
        }
        self._posthoc_dispatcher = {
            'one_way_anova': run_tukey_hsd,
            'two_way_anova': run_tukey_hsd, # 이원분산분석도 동일한 사후분석 사용 가능
        }
        logger.info("StatisticsService가 통합된 분석 함수들과 함께 초기화되었습니다.")

    def list_available_tests(self) -> Dict[str, List[str]]:
        return {test_id: info.get('required_params', []) for test_id, info in self._test_dispatcher.items()}

    def check_assumptions(self, data: pd.DataFrame, test_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if test_id not in self._test_dispatcher:
            raise NotImplementedError(f"'{test_id}' 통계 검정은 아직 지원되지 않습니다.")
        assumptions_to_check = self._test_dispatcher[test_id].get('assumptions', [])
        results = {}
        for assumption in assumptions_to_check:
            try:
                if assumption == 'normality':
                    value_col = params.get('value_col') or params.get('dependent_var') # 이원분산분석 지원
                    # 다른 테스트(e.g. 대응표본 t검정)의 value_col도 처리
                    if not value_col:
                        if params.get('after_col'): value_col = params.get('after_col')
                        elif params.get('before_col'): value_col = params.get('before_col')

                    if value_col:
                        # 그룹별 정규성 검정 (그룹이 있으면)
                        group_col = params.get('group_col')
                        if group_col and data[group_col].nunique() > 1: # 일원분산분석/t-test
                            results['normality'] = {}
                            for group_name in data[group_col].unique():
                                group_data = data[data[group_col] == group_name][value_col]
                                results['normality'][group_name] = check_normality(group_data)
                        elif params.get('independent_vars'): # 이원분산분석
                             results['normality'] = check_normality(data[value_col])
                        else: # 전체 데이터 정규성 검정
                            results['normality'] = check_normality(data[value_col])

                elif assumption == 'homoscedasticity':
                    value_col = params.get('value_col') or params.get('dependent_var')
                    
                    if params.get('group_col'): # 일원분산분석/t-test
                        group_col = params.get('group_col')
                        if group_col and value_col and data[group_col].nunique() > 1:
                            groups = [data[value_col][data[group_col] == g].dropna() for g in data[group_col].unique()]
                            non_empty_groups = [g for g in groups if not g.empty]
                            results['homoscedasticity'] = check_equal_variance(*non_empty_groups)
                    elif params.get('independent_vars'): # 이원분산분석
                        # 잔차의 등분산성을 봐야 하지만, 여기서는 그룹간 분산으로 근사
                        ind_vars = params['independent_vars']
                        groups = [data[value_col][(data[ind_vars[0]] == g1) & (data[ind_vars[1]] == g2)].dropna() 
                                  for g1 in data[ind_vars[0]].unique() for g2 in data[ind_vars[1]].unique()]
                        non_empty_groups = [g for g in groups if not g.empty]
                        if len(non_empty_groups) > 1:
                             results['homoscedasticity'] = check_equal_variance(*non_empty_groups)

            except Exception as e:
                results[assumption] = {'passed': False, 'error': str(e)}
        return results

    def execute_test(self, test_id: str, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        test_id에 따라 적절한 통계 검증 함수를 찾아 실행합니다.
        가정 검토 결과를 함께 반환합니다.
        """
        logger.info(f"'{test_id}' 통계 검정 실행 요청 수신. 파라미터: {params}")
        if test_id not in self._test_dispatcher:
            error_msg = f"'{test_id}' 통계 검정은 아직 지원되지 않습니다."
            logger.error(error_msg)
            raise NotImplementedError(error_msg)

        # 파라미터 이름 불일치에 대한 유연한 처리 (어댑터 로직)
        if 'dependent_variable' in params:
            params['dependent_var'] = params.pop('dependent_variable')
        if 'independent_variables' in params:
            params['independent_vars'] = params.pop('independent_variables')

        dispatcher_info = self._test_dispatcher[test_id]
        test_function = dispatcher_info['function']
        required_params = dispatcher_info.get('required_params', [])
        
        # 1. 가정 검토 실행
        assumption_results = self.check_assumptions(data, test_id, params)
        
        # 2. 가정 검토 결과에 따른 테스트 로직 분기
        test_info = self._test_dispatcher[test_id]
        test_function = test_info['function']
        final_params = params.copy()

        # [!!!] 서비스 계층의 책임 강화: 파라미터 변환 로직
        # LLM이 상위 수준의 파라미터만 제공하면, 서비스가 하위 수준 파라미터로 변환합니다.
        if test_id == 'two_proportion_test':
            group_col = final_params.pop('group_col', None)
            value_col = final_params.pop('value_col', None)
            
            if not group_col or not value_col:
                raise ValueError("두 표본 비율 검정을 위해서는 'group_col'과 'value_col'이 필요합니다.")
            
            groups = data[group_col].unique()
            if len(groups) != 2:
                raise ValueError("두 표본 비율 검정은 정확히 두 개의 그룹이 필요합니다.")
                
            group_data = data.groupby(group_col)[value_col]
            final_params['count'] = group_data.sum().tolist()
            final_params['nobs'] = group_data.count().tolist()
            
            # 후속 효과 크기 계산을 위해 비율(proportion) 저장
            prop1 = final_params['count'][0] / final_params['nobs'][0]
            prop2 = final_params['count'][1] / final_params['nobs'][1]
            # 이 값을 test_results에 포함시켜 다음 단계로 전달해야 함
            final_params['_internal_proportions'] = [prop1, prop2]

        # 독립표본 t-검정의 경우, 등분산성 가정에 따라 Welch's t-test로 전환
        if test_id == 'independent_t_test':
            homoscedasticity_res = assumption_results.get('homoscedasticity', {})
            equal_var = homoscedasticity_res.get('passed', True)
            final_params['equal_var'] = equal_var
            logger.info(f"등분산성 가정 충족 여부: {equal_var}. 이에 따라 t-검정 방식을 조정합니다.")

        # 3. 최종 테스트 실행
        try:
            # [!!!] 호출 방식 분기
            takes_dataframe = dispatcher_info.get('takes_dataframe', True) # 기본값은 True로 하여 이전 버전과 호환성 유지
            
            if takes_dataframe:
                test_results = test_function(data, **final_params)
            else:
                # 데이터프레임을 받지 않는 함수는 계산된 파라미터만 전달
                # 내부 계산값을 결과에 추가하기 위해 pop으로 가져와서 전달
                internal_props = final_params.pop('_internal_proportions', None)
                test_results = test_function(**final_params)
                if internal_props:
                    test_results['_internal_proportions'] = internal_props

            # 최종 결과에 가정 검토 결과도 함께 포함하여 반환
            return {
                "assumption_checks": assumption_results,
                "test_results": test_results
            }
        except Exception as e:
            logger.error(f"{test_id} 실행 중 오류: {e}", exc_info=True)
            return {
                "assumption_checks": assumption_results,
                "error": f"테스트 실행 중 오류가 발생했습니다: {e}"
            }

    def calculate_effect_size(self, test_id: str, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """효과 크기를 계산합니다."""
        # 상관 분석의 경우, 상관 계수 자체가 효과 크기이므로 별도 계산 없이 메시지 반환
        if test_id == 'correlation':
            return {'message': '상관 분석의 경우, 상관 계수(r)가 효과 크기를 나타냅니다. run_statistical_test 결과를 참고하세요.'}

        if test_id not in self._effect_size_dispatcher:
            raise NotImplementedError(f"'{test_id}'에 대한 효과 크기 계산은 아직 지원되지 않습니다.")
        
        dispatcher_info = self._effect_size_dispatcher[test_id]
        effect_size_function = dispatcher_info['function']
        required_params = dispatcher_info['required_params']
        
        # 필요한 파라미터 준비
        func_params = {}
        if test_id == 'independent_t_test':
            group_col = params.get('group_col')
            value_col = params.get('value_col')
            groups = data[group_col].unique()
            func_params['group1'] = data[data[group_col] == groups[0]][value_col]
            func_params['group2'] = data[data[group_col] == groups[1]][value_col]
        elif test_id == 'one_way_anova':
            # Agent가 주입한 test_results 객체에서 실제 통계 결과를 직접 찾습니다.
            actual_test_results = params.get('test_results', {}).get('test_results', {})
            func_params['f_statistic'] = actual_test_results.get('F_statistic')
            
            group_col = params.get('group_col')
            value_col = params.get('value_col')
            df_between = data[group_col].nunique() - 1
            df_within = len(data) - data[group_col].nunique()
            func_params['df_between'] = df_between
            func_params['df_within'] = df_within
        elif test_id == 'chi_square_independence':
            actual_test_results = params.get('test_results', {}).get('test_results', {})
            func_params['chi2_stat'] = actual_test_results.get('chi2_statistic')
            func_params['n'] = len(data)
            func_params['contingency_table'] = pd.crosstab(data[params['col1']], data[params['col2']])
        elif test_id == 'two_proportion_test':
            # execute_test에서 전달된 내부 계산값을 test_results에서 찾음
            actual_test_results = params.get('test_results', {}).get('test_results', {})
            proportions = actual_test_results.get('_internal_proportions')
            if not proportions or len(proportions) != 2:
                 raise ValueError("두 표본 비율 검사의 효과 크기를 계산하기 위한 비율(proportion) 값이 없습니다.")
            func_params['prop1'] = proportions[0]
            func_params['prop2'] = proportions[1]
        elif test_id == 'logistic_regression':
            actual_test_results = params.get('test_results', {}).get('test_results', {})
            func_params['test_results'] = actual_test_results
        elif test_id == 'two_way_anova':
            actual_test_results = params.get('test_results', {}).get('test_results', {})
            func_params['anova_table'] = actual_test_results.get('anova_table')

        missing = [p for p in required_params if p not in func_params or func_params[p] is None]
        if missing:
            raise ValueError(f"효과 크기 계산에 필요한 파라미터가 누락되었습니다: {missing}")

        return effect_size_function(**func_params)

    def run_posthoc_test(self, test_id: str, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """사후 분석을 실행합니다."""
        if test_id not in self._posthoc_dispatcher:
            raise NotImplementedError(f"'{test_id}'에 대한 사후 분석은 지원되지 않습니다.")
        
        posthoc_function = self._posthoc_dispatcher[test_id]
        
        # 'run_tukey_hsd'와 같은 함수는 'group_col', 'value_col' 등을 직접 받음
        # 이원 분산 분석의 경우, 두 개의 독립 변수가 필요
        if test_id == 'two_way_anova' and 'independent_vars' in params:
             # MultiComparison에는 하나의 그룹핑 변수만 전달 가능. 주 효과나 상호작용 효과를 특정해야 함.
             # 여기서는 단순화를 위해 첫 번째 독립변수를 기준으로 사후분석.
             # 실제로는 LLM이 어떤 변수로 사후분석할지 지정해야 더 정확함.
             group_col = params['independent_vars'][0]
             value_col = params['dependent_var']
             return posthoc_function(df=data, group_col=group_col, value_col=value_col)

        return posthoc_function(df=data, **params)