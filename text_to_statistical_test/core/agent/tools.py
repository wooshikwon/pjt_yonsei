# 파일명: core/agent/tools.py
from typing import Dict, Any, List, Callable
import pandas as pd

from services.statistics.stats_service import StatisticsService
# dispatchers를 임포트하여 필요한 파라미터 정보를 동적으로 참조할 수 있습니다.
from services.statistics.dispatchers import TEST_DISPATCHER, POSTHOC_DISPATCHER, EFFECT_SIZE_DISPATCHER

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
        
        # run_statistical_test의 상세 설명을 동적으로 생성
        test_desc = "핵심 통계 검정을 수행합니다. test_id에 따라 필요한 파라미터가 다릅니다.\n"
        test_desc += "아래 목록을 정확히 참고하여 필요한 파라미터를 전달하세요.\n"
        for test_id, info in TEST_DISPATCHER.items():
            params = info.get('required_params', [])

            # two_proportion_test는 내부적으로 파라미터를 변환하므로, LLM에게는 다른 파라미터를 안내해야 함
            if test_id == 'two_proportion_test':
                params_to_show = "['group_col', 'value_col'] (주의: 시스템이 내부적으로 두 그룹의 count와 nobs로 자동 변환하여 처리합니다.)"
            else:
                params_to_show = str(params)

            test_desc += f"- **{test_id}**: 필수 파라미터 = {params_to_show}\n"
        test_desc += "*참고: 'linear_regression'은 자체적으로 가정 검증을 포함하므로 'check_assumption'을 별도로 호출할 필요가 없습니다.*"

        # check_assumption의 상세 설명
        assumption_desc = (
            "t-test, ANOVA 등 그룹 간 평균 비교 분석의 통계적 가정을 검증합니다. **회귀나 상관 분석에는 절대 사용하지 마세요.**\n"
            "test_id에 따라 필요한 파라미터가 달라집니다:\n"
            "- **독립 그룹 분석 (independent_t_test, one_way_anova 등)**: `group_col` (그룹 구분 컬럼)과 `value_col` (검정할 값 컬럼)을 전달해야 합니다.\n"
            "- **대응표본 분석 (paired_t_test, wilcoxon_signed_rank 등)**: `before_col` ('이전' 값 컬럼)과 `after_col` ('이후' 값 컬럼)을 전달해야 합니다."
        )

        # run_posthoc_test의 상세 설명
        posthoc_desc = "ANOVA 분석 결과가 유의미할 때, 구체적으로 어떤 그룹 간에 차이가 있는지 확인하기 위한 사후 분석(Tukey's HSD)을 실행합니다.\n"
        posthoc_desc += "지원하는 분석과 필수 파라미터는 다음과 같습니다:\n"
        for test_id, info in POSTHOC_DISPATCHER.items():
            params = info.get('required_params', [])
            note = "(주의: 시스템이 내부적으로 이를 사후분석에 맞게 변환합니다.)" if test_id == 'two_way_anova' else ""
            posthoc_desc += f"- **{test_id}**: 필수 파라미터 = {params} {note}\n"
        
        # calculate_effect_size의 상세 설명
        effect_size_desc = (
            "통계적 유의성 외에 효과의 실제적인 크기를 측정합니다. **'correlation', 'linear_regression', 'logistic_regression' 분석에는 절대 사용하지 마세요.**\n"
            "이 도구는 **두 가지 정보가 모두 필요**합니다:\n"
            "1. 이전 `run_statistical_test` 단계의 **전체 결과 객체**를 `test_results` 파라미터에 전달해야 합니다.\n"
            "2. 이전 `run_statistical_test` 단계에 사용했던 **원본 파라미터들**(예: `group_col`, `value_col`)을 함께 전달해야 합니다.\n"
            "지원하는 분석과 필수 파라미터는 다음과 같습니다:\n"
            f"- **{' , '.join(EFFECT_SIZE_DISPATCHER.keys())}**"
        )


        return [
            {
                "type": "function",
                "function": {
                    "name": "run_statistical_test",
                    "description": test_desc,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "test_id": { "type": "string", "description": "실행할 통계 검정의 ID." }
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
                    "description": assumption_desc,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "test_id": { "type": "string", "description": "가정을 검토할 본 분석의 ID." },
                            "group_col": { "type": "string", "description": "그룹 구분 변수 컬럼명." },
                            "value_col": { "type": "string", "description": "검정할 숫자 값 컬럼명." },
                            "before_col": { "type": "string", "description": "대응표본의 '이전' 값 컬럼명." },
                            "after_col": { "type": "string", "description": "대응표본의 '이후' 값 컬럼명." }
                        },
                        "required": ["test_id"],
                        "additionalProperties": True
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_posthoc_test",
                    "description": posthoc_desc,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "test_id": {"type": "string", "description": "사후 분석의 기반이 된 본 분석 ID."}
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
                    "description": effect_size_desc,
                    "parameters": {
                        "type": "object",
                        "properties": {
                           "test_id": {"type": "string", "description": "효과 크기를 계산할 분석의 ID."},
                           "test_results": {"type": "object", "description": "이전 `run_statistical_test` 단계에서 반환된 전체 결과 객체."}
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

    def check_assumption(self, data: pd.DataFrame, test_id: str, **params) -> Dict[str, Any]:
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