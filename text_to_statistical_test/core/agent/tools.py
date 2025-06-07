# 파일명: core/agent/tools.py
"""
Agent가 사용하는 도구(Tools)의 레지스트리 및 구현.

각 도구는 서비스 계층의 기능을 호출하는 단순한 '래퍼(wrapper)' 역할을 수행합니다.
이를 통해 Agent의 의사결정 로직과 실제 기능 구현을 분리합니다.
"""
from typing import Dict, Any, List, Callable
import pandas as pd

from services.statistics.stats_service import StatisticsService

class ToolRegistry:
    """
    Agent가 사용할 수 있는 도구(함수)의 컬렉션입니다.
    서비스 인스턴스를 주입받아, 해당 서비스의 기능을 호출하는 메서드를 제공합니다.
    """

    def __init__(self, stats_service: StatisticsService):
        """
        서비스 인스턴스를 주입받아 초기화합니다.
        
        Args:
            stats_service: 통계 서비스 인스턴스.
        """
        self.stats_service = stats_service
        # 실제 함수 매핑
        self._tool_functions: Dict[str, Callable] = {
            "run_statistical_test": self.run_statistical_test,
            "check_assumption": self.check_assumption,
            "run_posthoc_test": self.run_posthoc_test,
            "calculate_effect_size": self.calculate_effect_size,
        }

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """LLM 에이전트에게 제공할 도구의 명세 목록을 반환합니다."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "run_statistical_test",
                    "description": "t-test, one_way_anova, two_way_anova, linear_regression, logistic_regression, correlation, chi_square_independence 등 다양한 핵심 통계 검정을 수행합니다. 각 test_id에 맞는 파라미터를 사용해야 합니다. 예를 들어, 'correlation' 분석은 'vars' 파라미터에 변수 리스트를 받으며 종속/독립 변수 구분이 없습니다. 반면 'linear_regression'은 'dependent_var'와 'independent_vars'를 사용합니다. 선형 회귀의 경우, 관련 가정 검토가 내장되어 있어 'check_assumption'을 별도로 호출할 필요가 없습니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "test_id": {
                                "type": "string",
                                "description": "실행할 통계 검정의 ID (예: 'independent_t_test', 'one_way_anova', 'linear_regression', 'two_proportion_test', 'correlation', 'logistic_regression', 'two_way_anova')."
                            }
                        },
                        "required": ["test_id"],
                        "additionalProperties": True
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_assumption",
                    "description": "오직 t-test, ANOVA와 같은 **그룹 간 평균 비교 분석**을 수행하기 전에만 사용합니다. 데이터가 해당 분석의 통계적 가정을 충족하는지(예: 정규성, 등분산성) 검증합니다. **절대로 회귀 분석이나 상관 분석 전에는 호출하지 마세요.**",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "test_id": {
                                "type": "string",
                                "description": "가정을 검토할 본 분석의 ID (예: 'independent_t_test', 'one_way_anova')."
                            },
                             "params": {
                                "type": "object",
                                "description": "가정 검토에 필요한 변수들을 포함하는 객체. (예: `{\"group_col\": \"UsedSupport\", \"value_col\": \"MonthlySpend\"}`)"
                            }
                        },
                        "required": ["test_id", "params"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_posthoc_test",
                    "description": "ANOVA 등 분산 분석 후에 유의미한 차이가 발견되었을 때, 어떤 그룹 간에 차이가 있는지 확인하기 위한 사후 분석(예: Tukey's HSD)을 실행합니다. 'test_id'와 분석에 필요한 파라미터(예: group_col, value_col)를 최상위 레벨에 직접 포함시켜야 합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "test_id": {"type": "string", "description": "실행할 사후 분석의 기반이 된 본 분석 ID (예: 'one_way_anova')."}
                        },
                        "required": ["test_id"],
                        "additionalProperties": True
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_effect_size",
                    "description": "통계적 유의성뿐만 아니라 효과의 실제적인 중요성(크기)을 측정합니다. (예: Cohen's d, Eta-squared). **'correlation' 분석의 경우, 상관 계수(r)가 이미 효과 크기이므로 이 도구를 호출할 필요가 없습니다.** 마찬가지로, 선형/로지스틱 회귀 분석의 경우, R-제곱 또는 Pseudo R-제곱 값이 효과 크기 역할을 하므로 이 도구를 호출할 필요가 없습니다. **이전 단계에서 `run_statistical_test`를 실행했다면, 그 단계의 전체 결과 객체를 `test_results` 파라미터에 반드시 전달해야 합니다.**",
                    "parameters": {
                        "type": "object",
                        "properties": {
                           "test_id": {"type": "string", "description": "효과 크기를 계산할 분석의 ID (예: 'independent_t_test', 'two_proportion_test')."},
                           "test_results": {"type": "object", "description": "본 분석으로부터 얻은 결과 딕셔너리."}
                        },
                        "required": ["test_id", "test_results"],
                        "additionalProperties": True
                    }
                }
            }
        ]

    def run_statistical_test(self, data: pd.DataFrame, test_id: str, **params) -> Dict[str, Any]:
        """통계 서비스의 execute_test 메서드를 호출합니다."""
        return self.stats_service.execute_test(test_id=test_id, data=data, params=params)

    def check_assumption(self, data: pd.DataFrame, test_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """통계 서비스의 check_assumptions 메서드를 호출합니다."""
        return self.stats_service.check_assumptions(data=data, test_id=test_id, params=params)

    def run_posthoc_test(self, data: pd.DataFrame, test_id: str, **params) -> Dict[str, Any]:
        """통계 서비스의 run_posthoc_test 메서드를 호출합니다."""
        return self.stats_service.run_posthoc_test(data=data, test_id=test_id, params=params)

    def calculate_effect_size(self, data: pd.DataFrame, test_id: str, test_results: Dict[str, Any], **params) -> Dict[str, Any]:
        """통계 서비스의 calculate_effect_size 메서드를 호출합니다."""
        # 서비스가 기대하는 단일 'params' 객체로 모든 정보를 통합합니다.
        final_params = params.copy()
        final_params['test_results'] = test_results
        return self.stats_service.calculate_effect_size(test_id=test_id, data=data, params=final_params)

    def get_tool(self, name: str) -> Callable:
        """이름으로 실제 실행 가능한 도구 함수를 가져옵니다."""
        if name not in self._tool_functions:
            raise ValueError(f"'{name}'은(는) 유효하지 않은 도구입니다.")
        return self._tool_functions[name]