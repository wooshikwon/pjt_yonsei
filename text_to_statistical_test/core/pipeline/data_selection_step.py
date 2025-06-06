# 파일명: core/pipeline/data_selection_step.py

import logging
from typing import Dict, Any

from .base_pipeline_step import BasePipelineStep
from utils.data_loader import DataLoader
from utils.ui_helpers import display_file_selection_menu
from utils.data_utils import get_available_data_files

class DataSelectionStep(BasePipelineStep):
    """1단계: 데이터 파일 선택 및 초기 이해"""
    
    def __init__(self):
        super().__init__("데이터 파일 선택", 1)
        self.data_loader = DataLoader()
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return isinstance(input_data, dict)
    
    def get_expected_output_schema(self) -> Dict[str, Any]:
        return {'selected_file': str, 'data_object': object, 'file_metadata': dict}
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 파일을 선택하고 로드합니다."""
        self.logger.info("사용 가능한 데이터 파일을 검색합니다.")
        available_files = get_available_data_files()
        
        if not available_files:
            raise FileNotFoundError("분석할 데이터 파일이 'input_data/data_files/' 폴더에 없습니다.")
            
        selected_file = display_file_selection_menu(available_files)
        if not selected_file:
            raise InterruptedError("사용자가 파일 선택을 취소했습니다.")
            
        self.logger.info(f"선택된 파일 로딩 중: {selected_file}")
        df, metadata = self.data_loader.load_file(selected_file)
        
        if df is None:
            raise IOError(metadata.get('error', '파일 로딩에 실패했습니다.'))
            
        self.logger.info(f"파일 로딩 완료: {df.shape[0]}행, {df.shape[1]}열")

        return {
            'selected_file': selected_file,
            'data_object': df,
            'file_metadata': metadata
        }