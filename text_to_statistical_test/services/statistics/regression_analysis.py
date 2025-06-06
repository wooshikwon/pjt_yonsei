# 파일명: services/statistics/regression_analysis.py

import pandas as pd
from typing import Dict, Any, List
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, Logit

def run_pearson_correlation(
    df: pd.DataFrame, x_col: str, y_col: str
) -> Dict[str, Any]:
    """
    피어슨 상관분석을 수행합니다.
    :param df: 분석에 사용할 DataFrame
    :param x_col: 독립 변수(연속형)
    :param y_col: 종속 변수(연속형)
    :return: 상관계수와 p-value를 담은 dict
    """
    # NA 제거
    valid_df = df[[x_col, y_col]].dropna()
    if valid_df.shape[0] < 2:
        raise ValueError("데이터가 충분하지 않아 피어슨 상관분석을 수행할 수 없습니다.")

    x = valid_df[x_col]
    y = valid_df[y_col]

    corr_coef, p_value = stats.pearsonr(x, y)
    return {
        'test_name': 'Pearson Correlation',
        'correlation_coefficient': corr_coef,
        'p_value': p_value,
        'n': len(valid_df)
    }


def run_linear_regression(
    df: pd.DataFrame, dependent_var: str, independent_vars: List[str]
) -> Dict[str, Any]:
    """
    다중 선형회귀분석을 수행합니다.
    :param df: 분석에 사용할 DataFrame
    :param dependent_var: 종속 변수
    :param independent_vars: 독립 변수 목록
    :return: 회귀계수, p-value, R-squared 등을 담은 dict
    """
    # NA 제거: 종속·독립 변수 모두 결측치가 없도록 필터링
    cols = [dependent_var] + independent_vars
    valid_df = df[cols].dropna()
    if valid_df.shape[0] < len(independent_vars) + 2:
        raise ValueError("데이터가 충분하지 않아 다중 선형회귀분석을 수행할 수 없습니다.")

    # 독립변수 행렬 X와 종속변수 벡터 y 준비
    X = valid_df[independent_vars]
    X = sm.add_constant(X, has_constant='add')  # 상수항 추가
    y = valid_df[dependent_var]

    model = sm.OLS(y, X).fit()
    summary = model.summary2().tables[1]  # 계수 표(table 1)
    coef_table = summary.to_dict('records')

    return {
        'test_name': 'Multiple Linear Regression',
        'formula': f"{dependent_var} ~ {' + '.join(independent_vars)}",
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'coefficients': coef_table,
        'f_statistic': float(model.fvalue),
        'f_pvalue': float(model.f_pvalue),
        'degrees_of_freedom': int(model.df_resid),
    }


def run_logistic_regression(
    df: pd.DataFrame, dependent_var: str, independent_vars: List[str]
) -> Dict[str, Any]:
    """
    로지스틱 회귀분석을 수행합니다.
    :param df: 분석에 사용할 DataFrame
    :param dependent_var: 종속 변수(이진 분류; 0/1)
    :param independent_vars: 독립 변수 목록
    :return: 회귀계수, p-value, 오즈비(odds ratio) 등을 담은 dict
    """
    # NA 제거 및 종속변수가 0/1로만 이루어져 있는지 확인
    cols = [dependent_var] + independent_vars
    valid_df = df[cols].dropna()
    if valid_df.shape[0] < len(independent_vars) + 10:
        # 샘플 수가 지나치게 적으면 추정이 불안정할 수 있음
        raise ValueError("데이터가 충분하지 않아 로지스틱 회귀분석을 수행할 수 없습니다.")

    if not set(valid_df[dependent_var].unique()).issubset({0, 1}):
        raise ValueError("종속 변수가 0과 1의 이진값이 아닙니다.")

    # 독립변수 행렬 X와 종속변수 벡터 y 준비
    X = valid_df[independent_vars]
    X = sm.add_constant(X, has_constant='add')
    y = valid_df[dependent_var]

    model = Logit(y, X).fit(disp=False)
    summary = model.summary2().tables[1]  # 계수 표
    coef_table = summary.to_dict('records')

    # 오즈비(odds ratio) 계산
    odds_ratios = {row['index']: float(np.exp(row['Coef.'])) for row in summary.reset_index().to_dict('records')}

    return {
        'test_name': 'Logistic Regression',
        'formula': f"{dependent_var} ~ {' + '.join(independent_vars)}",
        'coefficients': coef_table,
        'odds_ratios': odds_ratios,
        'pseudo_r_squared': float(model.prsquared),
        'llr_pvalue': float(model.llr_pvalue),
    }
