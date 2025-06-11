# services/statistics/stats_service.py
"""
통계 분석 기능을 통합 제공하는 서비스 파사드(Facade).

이 모듈의 StatisticsService 클래스는 모든 통계 관련 요청을 받는 단일 창구 역할을 합니다.
내부적으로 디스패처를 사용하여 요청을 적절한 순수 함수로 전달하고,
파라미터 변환, 가정 검토 등 오케스트레이션 로직을 수행합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

from .dispatchers import (
    TEST_DISPATCHER, 
    EFFECT_SIZE_DISPATCHER, 
    POSTHOC_DISPATCHER, 
    ASSUMPTION_CHECKER_MAP
)

logger = logging.getLogger(__name__)

class StatisticsService:
    """통계 분석 기능을 통합 제공하는 서비스 (Facade)."""
    
    def __init__(self):
        self._test_dispatcher = TEST_DISPATCHER
        self._effect_size_dispatcher = EFFECT_SIZE_DISPATCHER
        self._posthoc_dispatcher = POSTHOC_DISPATCHER
        logger.info("StatisticsService가 리팩토링된 디스패처와 함께 초기화되었습니다.")

    # ... (list_available_tests, check_assumptions, execute_test 메서드는 변경 없음) ...
    def list_available_tests(self) -> Dict[str, List[str]]:
        """사용 가능한 통계 테스트와 필수 파라미터 목록을 반환합니다."""
        return {test_id: info.get('required_params', []) for test_id, info in self._test_dispatcher.items()}

    def check_assumptions(self, data: pd.DataFrame, test_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """분석에 필요한 통계적 가정을 검토합니다."""
        if test_id not in self._test_dispatcher:
            raise NotImplementedError(f"'{test_id}' 통계 검정은 아직 지원되지 않습니다.")
        
        assumptions_to_check = self._test_dispatcher[test_id].get('assumptions', [])
        results = {}
        for assumption_name in assumptions_to_check:
            try:
                checker_function = ASSUMPTION_CHECKER_MAP.get(assumption_name)
                if not checker_function:
                    continue

                if assumption_name == 'normality':
                    value_col = params.get('value_col') or params.get('before_col') or params.get('after_col')
                    if value_col:
                        group_col = params.get('group_col')
                        if group_col and data[group_col].nunique() > 1:
                            results[assumption_name] = {}
                            for group_name in data[group_col].unique():
                                group_data = data[data[group_col] == group_name][value_col]
                                results[assumption_name][group_name] = checker_function(group_data)
                        else:
                            results[assumption_name] = checker_function(data[value_col])
                
                elif assumption_name == 'homoscedasticity':
                    group_col = params.get('group_col')
                    value_col = params.get('value_col')
                    if group_col and value_col and data[group_col].nunique() > 1:
                        groups = [data[value_col][data[group_col] == g].dropna() for g in data[group_col].unique()]
                        non_empty_groups = [g for g in groups if not g.empty]
                        results[assumption_name] = checker_function(*non_empty_groups)

            except Exception as e:
                results[assumption_name] = {'passed': False, 'error': str(e)}
        return results

    def execute_test(self, test_id: str, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """test_id에 따라 적절한 통계 검증 함수를 찾아 실행합니다."""
        logger.info(f"'{test_id}' 통계 검정 실행 요청 수신. 파라미터: {params}")
        if test_id not in self._test_dispatcher:
            raise NotImplementedError(f"'{test_id}' 통계 검정은 아직 지원되지 않습니다.")

        dispatcher_info = self._test_dispatcher[test_id]
        test_function = dispatcher_info['function']
        final_params = params.copy()

        assumption_results = self.check_assumptions(data, test_id, final_params)
        
        if test_id == 'two_proportion_test':
            group_col = final_params.pop('group_col', None)
            value_col = final_params.pop('value_col', None)
            if not group_col or not value_col:
                raise ValueError("두 표본 비율 검정을 위해서는 'group_col'과 'value_col'이 필요합니다.")
            groups = data[group_col].unique()
            if len(groups) != 2: raise ValueError("두 표본 비율 검정은 정확히 두 개의 그룹이 필요합니다.")
            group_data = data.groupby(group_col)[value_col]
            final_params['count'] = group_data.sum().tolist()
            final_params['nobs'] = group_data.count().tolist()
            prop1 = final_params['count'][0] / final_params['nobs'][0]
            prop2 = final_params['count'][1] / final_params['nobs'][1]
            final_params['_internal_proportions'] = [prop1, prop2]

        if test_id == 'independent_t_test':
            homoscedasticity_res = assumption_results.get('homoscedasticity', {})
            final_params['equal_var'] = homoscedasticity_res.get('passed', True)
            logger.info(f"등분산성 가정 충족 여부: {final_params['equal_var']}. t-검정 방식 조정.")

        try:
            takes_dataframe = dispatcher_info.get('takes_dataframe', True)
            if takes_dataframe:
                test_results = test_function(data, **final_params)
            else:
                internal_props = final_params.pop('_internal_proportions', None)
                test_results = test_function(**final_params)
                if internal_props:
                    test_results['_internal_proportions'] = internal_props
            
            if isinstance(test_results, dict) and 'assumption_checks' in test_results:
                assumption_results = test_results.pop('assumption_checks')

            return {"assumption_checks": assumption_results, "test_results": test_results}
        except Exception as e:
            logger.error(f"{test_id} 실행 중 오류: {e}", exc_info=True)
            return {"assumption_checks": assumption_results, "error": f"테스트 실행 중 오류: {e}"}

    # [!!!] 완전히 새로워진 calculate_effect_size 메서드
    def calculate_effect_size(self, test_id: str, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        효과 크기를 계산합니다.
        복잡한 파라미터 생성 로직을 디스패처의 'preparer' 함수에 위임합니다.
        """
        logger.info(f"'{test_id}' 효과 크기 계산 요청. 파라미터: {params}")
        if test_id == 'correlation':
            return {'message': '상관 분석의 경우, 상관 계수(r)가 효과 크기를 나타냅니다.'}
        if test_id not in self._effect_size_dispatcher:
            raise NotImplementedError(f"'{test_id}'에 대한 효과 크기 계산은 지원되지 않습니다.")
        
        dispatcher_info = self._effect_size_dispatcher[test_id]
        effect_size_function = dispatcher_info['function']
        preparer_function = dispatcher_info.get('preparer')

        if not preparer_function:
             raise NotImplementedError(f"'{test_id}'에 대한 효과 크기 파라미터 준비 함수(preparer)가 정의되지 않았습니다.")

        try:
            # '준비 함수'를 통해 실제 계산 함수에 필요한 모든 파라미터를 생성합니다.
            func_params = preparer_function(data, params)
    
            # 생성된 파라미터에 누락된 값이 있는지 확인합니다.
            if any(p is None for p in func_params.values()):
                raise ValueError(f"효과 크기 계산에 필요한 파라미터 값이 누락되었습니다. 생성된 파라미터: {func_params}")
    
            return effect_size_function(**func_params)
        except Exception as e:
            logger.error(f"'{test_id}' 효과 크기 계산 중 오류 발생: {e}", exc_info=True)
            raise ValueError(f"'{test_id}' 효과 크기 계산 중 오류가 발생했습니다: {e}")


    def run_posthoc_test(self, test_id: str, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """사후 분석을 실행합니다."""
        if test_id not in self._posthoc_dispatcher:
            raise NotImplementedError(f"'{test_id}'에 대한 사후 분석은 지원되지 않습니다.")
        
        posthoc_function = self._posthoc_dispatcher[test_id]
        
        final_params = params.copy()
        if test_id == 'two_way_anova' and 'independent_vars' in params:
             final_params['group_col'] = params['independent_vars'][0]
             final_params['value_col'] = params['dependent_var']
             del final_params['independent_vars'], final_params['dependent_var']

        return posthoc_function(df=data, **final_params)