# 파일명: core/pipeline/user_request_step.py

import logging
from typing import Dict, Any

from .base_pipeline_step import BasePipelineStep
from utils.ui_helpers import get_user_input
from services.llm.llm_service import LLMService

class UserRequestStep(BasePipelineStep):
    """2단계: 사용자 자연어 요청 및 목표 정의"""

    def __init__(self):
        super().__init__("사용자 요청 분석", 2)
        self.llm_service = LLMService()

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return 'data_object' in input_data and 'file_metadata' in input_data

    def get_expected_output_schema(self) -> Dict[str, Any]:
        return {'user_request': str, 'structured_request': dict}

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """사용자의 자연어 요청을 받아 분석 목표를 구조화합니다."""
        user_request = get_user_input("분석하고 싶은 내용을 자연어로 설명해주세요: ")
        if not user_request:
            raise ValueError("사용자 요청이 비어있습니다.")
            
        self.logger.info(f"사용자 요청 수신: '{user_request}'")
        
        metadata = input_data['file_metadata']
        data_context = f"""
        - 파일명: {metadata.get('file_name')}
        - 데이터 크기: {metadata.get('shape', (0,0))[0]}행, {metadata.get('shape', (0,0))[1]}열
        - 컬럼: {metadata.get('columns', [])}
        """
        
        # [SERVICE-REQ] llm_service.py에 interpret_user_request 구현 필요
        structured_request = self.llm_service.interpret_user_request(
            request=user_request, 
            data_context=data_context
        )

        self.logger.info(f"구조화된 분석 목표: {structured_request}")
        
        input_data.update({
            'user_request': user_request,
            'structured_request': structured_request,
        })
        return input_data