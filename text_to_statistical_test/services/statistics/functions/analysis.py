# services/statistics/functions/analysis.py
"""
핵심 통계 분석 함수 모음.

이 모듈의 함수들은 순수한 계산 로직을 담당하며,
외부 서비스나 클래스에 대한 의존성이 없습니다.
각 함수는 데이터와 파라미터를 입력받아 통계 분석 결과를 반환합니다.
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, Any, List

# ==============================================================================
# 통계 테스트 함수들 (Parametric Tests)
# ==============================================================================

def run_independent_t_test(df: pd.DataFrame, group_col: str, value_col: str, equal_var: bool = True) -> Dict[str, Any]:
    """독립표본 t-검정 또는 Welch's t-검정을 수행합니다."""
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
    """대응표본 t-검정을 수행합니다."""
    before_data = df[before_col].dropna()
    after_data = df[after_col].dropna()
    if len(before_data) != len(after_data):
        raise ValueError("대응표본 t-검정을 위해 두 컬럼의 데이터 개수가 일치해야 합니다.")
    if len(before_data) < 2:
        raise ValueError("샘플 크기가 너무 작습니다.")
    stat, p_value = stats.ttest_rel(before_data, after_data)
    return {'test_name': 'Paired Samples t-test', 't_statistic': stat, 'p_value': p_value, 'degrees_of_freedom': len(before_data) - 1}

def run_one_way_anova(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
    """일원 분산 분석(One-way ANOVA)을 수행합니다."""
    groups = df[group_col].dropna().unique()
    if len(groups) < 3:
        raise ValueError("ANOVA는 세 개 이상의 그룹이 필요합니다.")
    grouped_data = [df[value_col][df[group_col] == g].dropna() for g in groups]
    if any(len(g) < 2 for g in grouped_data):
        raise ValueError("ANOVA를 수행하기에 그룹별 샘플 크기가 너무 작습니다.")
    stat, p_value = stats.f_oneway(*grouped_data)
    return {'test_name': 'One-way ANOVA', 'F_statistic': stat, 'p_value': p_value}

def run_two_way_anova(df: pd.DataFrame, dependent_var: str, independent_vars: List[str]) -> Dict[str, Any]:
    """이원 분산 분석(Two-way ANOVA)을 수행합니다."""
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
    """카이제곱 독립성 검정을 수행합니다."""
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
    """Mann-Whitney U 검정을 수행합니다."""
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
    """Kruskal-Wallis H-검정을 수행합니다."""
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
    """Wilcoxon 부호-순위 검정을 수행합니다."""
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