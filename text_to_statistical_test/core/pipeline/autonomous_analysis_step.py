# 파일명: core/pipeline/autonomous_analysis_step.py

import logging
from typing import Dict, Any

from .base_pipeline_step import BasePipelineStep
from services.llm.llm_service import LLMService
from services.statistics.stats_service import StatisticsService

logger = logging.getLogger(__name__)

class AutonomousAnalysisStep(BasePipelineStep):
    """3단계: AI Agent의 자율적 통계 분석 실행"""
    def __init__(self):
        super().__init__("자율 통계 분석", 3)
        self.llm_service = LLMService()
        self.stats_service = StatisticsService()

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return 'data_object' in input_data and 'structured_request' in input_data

    def get_expected_output_schema(self) -> Dict[str, Any]:
        return {'analysis_results': dict, 'final_plan': dict}

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 요청에 따라 통계 분석의 전 과정을 자율적으로 수행합니다."""
        df = input_data['data_object']
        request = input_data['structured_request']
        
        self.logger.info("자율 분석 프로세스를 시작합니다.")

        # [SERVICE-REQ] llm_service.py에 create_detailed_analysis_plan 구현 필요
        # LLM이 요청과 데이터 특성을 보고 분석에 필요한 모든 절차(가정, 본검정, 대안, 사후검정)가 포함된 계획을 수립
        final_plan = self.llm_service.create_detailed_analysis_plan(request)
        logger.info(f"AI Agent가 수립한 분석 계획: {final_plan}")
        
        # [SERVICE-REQ] statistics_service.py에 check_assumptions 구현 필요
        assumption_results = self.stats_service.check_assumptions(df, final_plan.get('assumptions', []))
        logger.info(f"사전 가정 검토 결과: {assumption_results}")
        
        executed_test_name = final_plan['primary_test']
        # 가정 검토 결과에 따른 동적 분석 방법 결정
        if not all(result.get('passed', True) for result in assumption_results.values()):
            if final_plan.get('fallback_test'):
                executed_test_name = final_plan['fallback_test']
                logger.info(f"하나 이상의 가정을 충족하지 못하여 대안 분석('{executed_test_name}')을 실행합니다.")
            else:
                logger.warning("가정을 충족하지 못했으나, 대안 분석이 계획에 없어 기본 분석을 강행합니다.")
        
        logger.info(f"핵심 분석 '{executed_test_name}'을 실행합니다.")
        # [SERVICE-REQ] statistics_service.py에 run_test 구현 필요
        main_test_results = self.stats_service.run_test(df, executed_test_name, request['variables'])

        # [SERVICE-REQ] statistics_service.py에 calculate_effect_size 구현 필요
        effect_size = self.stats_service.calculate_effect_size(df, main_test_results, final_plan.get('effect_size_method'))
        
        posthoc_results = None
        if final_plan.get('posthoc_needed') and main_test_results.get('p_value', 1.0) < 0.05:
            logger.info("유의한 결과에 따라 사후 검정을 실행합니다.")
            # [SERVICE-REQ] statistics_service.py에 run_posthoc_test 구현 필요
            posthoc_results = self.stats_service.run_posthoc_test(df, main_test_results)

        analysis_results = {
            'assumption_results': assumption_results,
            'main_test': {'name': executed_test_name, **main_test_results},
            'effect_size': effect_size,
            'posthoc_test': posthoc_results
        }
        
        input_data.update({'analysis_results': analysis_results, 'final_plan': final_plan})
        return input_data