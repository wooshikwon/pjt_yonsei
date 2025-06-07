# 파일명: core/pipeline/data_selection_step.py

import logging
from typing import Dict, Any

from core.context import AppContext
from core.pipeline.pipeline_step import PipelineStep
from utils.data_loader import DataLoader # helpers가 아닌 data_loader에서 DataLoader를 임포트
# [Note] UI 관련 로직 (ui_helpers)은 main.py로 이동되어 해당 의존성을 제거했습니다.

class DataSelectionStep(PipelineStep):
    """1단계: 데이터 파일 로드 및 기본 정보 추출"""
    
    def __init__(self):
        super().__init__("데이터 로드")
        self.data_loader = DataLoader() # DataLoader 인스턴스 사용
        self.logger = logging.getLogger(__name__)
        
    async def run(self, context: AppContext) -> AppContext:
        """
        주어진 파일 경로에서 데이터를 로드하고 기본 정보를 context에 추가합니다.
        
        Args:
            context: 현재 워크플로우의 context. 'file_path' 키가 필요합니다.

        Returns:
            context: 'dataframe'이 추가된 업데이트된 context.
        """
        file_path = context.file_path
        if not file_path:
            raise ValueError("DataSelectionStep: 컨텍스트에 'file_path'가 제공되지 않았습니다.")
            
        self.logger.info(f"주어진 경로의 데이터 파일 로드를 시작합니다: {file_path}")
        
        try:
            # DataLoader.load_file은 DataFrame 객체 하나만 반환
            df = self.data_loader.load_file(file_path)
        except Exception as e:
            self.logger.error(f"파일 로딩 중 오류 발생: {file_path}", exc_info=True)
            raise IOError(f"'{file_path}' 파일을 로드할 수 없습니다.") from e
            
        self.logger.info(f"파일 로딩 완료. 데이터 크기: {df.shape[0]}행, {df.shape[1]}열")

        # 다음 단계를 위해 업데이트된 정보를 context에 추가
        context.dataframe = df
        
        # DataFrame.info()는 None을 반환하고 결과를 stdout에 출력하므로, 문자열로 캡처해야 함
        from io import StringIO
        buffer = StringIO()
        df.info(buf=buffer)
        context.dataframe_info = buffer.getvalue()
        
        context.dataframe_head = df.head().to_markdown()

        self.logger.info(f"'{file_path}' 로드 성공. {df.shape[0]}행, {df.shape[1]}열.")
        
        return context