# 파일명: services/statistics/stats_service.py

import pandas as pd
from typing import Dict, Any, List

# 하위 모듈 임포트
from . import assumption_checks, parametric_tests, nonparametric_tests, regression_analysis, categorical_analysis, posthoc_analysis, effect_size
from utils.error_handler import StatisticalException

class StatisticsService:
    """통계 분석 기능을 통합 제공하는 서비스 (Facade)"""

    def __init__(self):
        # 테스트 이름과 실제 실행 함수를 매핑하는 디스패처
        self._test_dispatcher = {
            '독립표본 t-검정': parametric_tests.run_independent_t_test,
            '대응표본 t-검정': parametric_tests.run_paired_t_test,
            '일원분산분석': parametric_tests.run_one_way_anova,
            'Mann-Whitney U 검정': nonparametric_tests.run_mann_whitney_u,
            'Kruskal-Wallis 검정': nonparametric_tests.run_kruskal_wallis,
            '피어슨 상관분석': regression_analysis.run_pearson_correlation,
            '다중선형회귀분석': regression_analysis.run_linear_regression,
            '로지스틱 회귀분석': regression_analysis.run_logistic_regression,
            '카이제곱 독립성 검정': categorical_analysis.run_chi_square_independence,
        }
        self._effect_size_dispatcher = {
            "Cohen's d": effect_size.calculate_cohens_d,
            "Eta-squared": effect_size.calculate_eta_squared,
            "Cramer's V": effect_size.calculate_cramers_v,
        }

    def check_assumptions(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        """계획에 명시된 가정들을 검토합니다."""
        results = {}
        variables = plan['request'].get('variables', {})
        
        for assumption in plan.get('assumptions', []):
            try:
                if assumption == '정규성':
                    dep_var = variables.get('dependent', [None])[0]
                    if dep_var:
                        results['정규성'] = assumption_checks.check_normality(df[dep_var])
                elif assumption == '등분산성':
                    dep_var = variables.get('dependent', [None])[0]
                    indep_var = variables.get('independent', [None])[0]
                    if dep_var and indep_var and df[indep_var].nunique() > 1:
                        groups = [df[dep_var][df[indep_var] == g] for g in df[indep_var].unique()]
                        results['등분산성'] = assumption_checks.check_equal_variance(*groups)
                elif assumption == '다중공선성':
                    indep_vars = variables.get('independent', [])
                    if len(indep_vars) > 1:
                        results['다중공선성'] = assumption_checks.check_multicollinearity(df, indep_vars)
            except Exception as e:
                results[assumption] = {'passed': False, 'error': str(e)}
        return results

    def run_test(self, df: pd.DataFrame, test_name: str, variables: Dict) -> Dict[str, Any]:
        """테스트 이름에 따라 적절한 통계 검정을 실행합니다."""
        if test_name not in self._test_dispatcher:
            raise NotImplementedError(f"'{test_name}' 통계 검정은 아직 구현되지 않았습니다.")
        
        try:
            test_function = self._test_dispatcher[test_name]
            # [UTIL-REQ] 각 분석 함수에 맞는 변수를 동적으로 전달하는 로직이 필요할 수 있습니다.
            # 예시: 독립변수 1개, 종속변수 1개 등 함수의 시그니처에 맞춰 변수 전달
            # 아래는 간단한 예시입니다.
            if "t-검정" in test_name or "ANOVA" in test_name:
                 return test_function(df, group_col=variables['independent'][0], value_col=variables['dependent'][0])
            elif "회귀" in test_name:
                return test_function(df, dependent_var=variables['dependent'][0], independent_vars=variables['independent'])
            # ... 다른 분석 유형에 대한 처리 ...
            else: # 상관, 카이제곱 등
                return test_function(df, variables['dependent'] + variables['independent'])
        except Exception as e:
            raise StatisticalException(f"'{test_name}' 실행 중 오류 발생: {e}")

    def calculate_effect_size(self, df: pd.DataFrame, test_results: Dict, method: str) -> Dict[str, Any]:
        """효과 크기를 계산합니다."""
        if method not in self._effect_size_dispatcher:
            return {'error': f"지원하지 않는 효과 크기 계산법: {method}"}
        
        try:
            # [TODO] 각 효과 크기 계산에 필요한 인자를 test_results와 df에서 추출하는 로직 필요
            # 아래는 Cohen's d에 대한 간단한 예시입니다.
            if method == "Cohen's d":
                # 이 부분은 예시이며, 실제로는 run_test의 인자로부터 그룹 정보를 받아야 합니다.
                # 지금은 간단화를 위해 이 함수가 직접 계산한다고 가정합니다.
                return {"Cohen's d": 0.5} # 임시 값
            return self._effect_size_dispatcher[method](...)
        except Exception as e:
            return {'error': f"효과 크기 계산 중 오류: {e}"}

    def run_posthoc_test(self, df: pd.DataFrame, main_test_results: Dict, plan: Dict) -> Dict[str, Any]:
        """사후 검정을 실행합니다."""
        try:
            # [TODO] ANOVA 결과와 데이터를 바탕으로 Tukey HSD 등 사후 검정 실행
            # return posthoc_analysis.run_tukey_hsd(...)
            return {"Tukey HSD": "Group A와 Group C 간에 유의미한 차이가 발견되었습니다."}
        except Exception as e:
            raise StatisticalException(f"사후 검정 실행 중 오류 발생: {e}")