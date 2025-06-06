# 파일명: core/pipeline/reporting_step.py

import logging
from typing import Dict, Any
from .base_pipeline_step import BasePipelineStep
from services.llm.llm_service import LLMService
from core.reporting.report_builder import ReportBuilder

class ReportingStep(BasePipelineStep):
    """5단계: 최종 보고서 생성"""
    def __init__(self):
        super().__init__("최종 보고서 생성", 5)
        self.llm_service = LLMService()
        self.report_builder = ReportBuilder()

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return 'analysis_results' in input_data and 'visual_artifacts' in input_data

    def get_expected_output_schema(self) -> Dict[str, Any]:
        return {'report_path': str}

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """통계 결과와 시각화를 종합하여 최종 보고서를 생성합니다."""
        self.logger.info("최종 보고서 생성을 시작합니다.")
        
        # [SERVICE-REQ] llm_service.py에 generate_report_narrative 구현 필요
        narrative = self.llm_service.generate_report_narrative(
            results=input_data['analysis_results'],
            plan=input_data['final_plan'],
            artifacts=input_data['visual_artifacts']
        )
        self.logger.info("보고서 서술부(Narrative) 생성을 완료했습니다.")
        
        # [SERVICE-REQ] report_builder.py에 build 구현 필요
        report_path = self.report_builder.build(
            narrative=narrative, 
            visuals=input_data.get('visual_artifacts', [])
        )
        self.logger.info(f"최종 보고서가 생성되었습니다: {report_path}")
        
        input_data['report_path'] = report_path
        return input_data