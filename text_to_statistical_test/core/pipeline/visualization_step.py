# 파일명: core/pipeline/visualization_step.py

import logging
from typing import Dict, Any, List
from .base_pipeline_step import BasePipelineStep
from services.llm.llm_service import LLMService
from services.visualization.viz_service import VisualizationService

class VisualizationStep(BasePipelineStep):
    """4단계: 분석 결과 기반 자율 시각화"""
    def __init__(self):
        super().__init__("자율 시각화", 4)
        self.llm_service = LLMService()
        self.viz_service = VisualizationService()

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return 'analysis_results' in input_data

    def get_expected_output_schema(self) -> Dict[str, Any]:
        return {'visual_artifacts': List[str]}

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """분석 결과에 가장 적합한 시각화를 자율적으로 생성합니다."""
        analysis_results = input_data['analysis_results']
        df = input_data['data_object']
        
        # [SERVICE-REQ] llm_service.py에 recommend_visualizations 구현 필요
        recommended_charts = self.llm_service.recommend_visualizations(analysis_results)
        self.logger.info(f"AI Agent가 추천한 시각화: {recommended_charts}")
        
        # [SERVICE-REQ] viz_service.py에 create_plots 구현 필요
        visual_artifacts = self.viz_service.create_plots(df, recommended_charts)
        self.logger.info(f"생성된 시각화 파일: {visual_artifacts}")

        input_data['visual_artifacts'] = visual_artifacts
        return input_data