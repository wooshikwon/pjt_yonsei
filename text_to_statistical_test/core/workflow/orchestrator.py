"""
Workflow Orchestrator

프로젝트의 핵심 5단계 파이프라인 실행을 관리하는 '지휘자(Orchestrator)'.
`Orchestrator-Engine` 모델에 따라, 실제 로직은 각 단계(Step)와 서비스(Service)에 위임하고
자신은 전체적인 흐름만 담당하여 단순성과 명확성을 유지합니다.
"""

import asyncio
import logging
import importlib
from typing import Dict, Any

from core.context import AppContext
from core.pipeline.pipeline_step import PipelineStep
# [Note] 이전 버전의 복잡한 의존성(StateManager, ErrorHandler 등)은 모두 제거되었습니다.

class Orchestrator:
    """
    프로젝트의 핵심 워크플로우 실행을 관리하는 지휘자.
    각 단계를 순차적으로 실행하는 역할에 집중합니다.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # 새로운 3단계 워크플로우 정의 (CodeExecutionStep 제거)
        self._step_definitions = {
            1: ('core.pipeline.data_selection_step', 'DataSelectionStep'),
            2: ('core.pipeline.autonomous_analysis_step', 'AutonomousAnalysisStep'),
            3: ('core.pipeline.reporting_step', 'ReportingStep'),
        }
        self.logger.info("Orchestrator가 새로운 3단계 워크플로우로 초기화되었습니다.")

    def _get_step_instance(self, step_num: int) -> PipelineStep:
        """
        필요한 시점에 각 파이프라인 단계의 인스턴스를 동적으로 로드하고 반환합니다.
        이를 통해 의존성을 중앙에서 관리하고, 시작 시점의 부하를 줄입니다.
        """
        try:
            module_name, class_name = self._step_definitions[step_num]
            module = importlib.import_module(module_name)
            step_class = getattr(module, class_name)
            return step_class()
        except (ImportError, AttributeError, KeyError) as e:
            self.logger.error(f"파이프라인 단계({step_num})를 로드하거나 인스턴스화할 수 없습니다: {e}", exc_info=True)
            raise RuntimeError(f"Pipeline step {step_num} could not be loaded.") from e

    async def run(self, file_path: str, user_request: str) -> AppContext:
        """
        정의된 3단계 파이프라인 전체를 순차적으로 실행합니다.

        Args:
            file_path (str): 사용자가 선택한 분석 대상 데이터 파일의 전체 경로.
            user_request (str): 사용자의 분석 요청 (자연어).

        Returns:
            AppContext: 모든 분석 결과가 포함된 최종 컨텍스트 객체.
        """
        self.logger.info("="*20 + " 전체 워크플로우 시작 " + "="*20)
        self.logger.info(f"입력 데이터: {file_path}")
        self.logger.info(f"사용자 요청: {user_request}")

        # 모든 단계의 결과를 담을 중앙 데이터 저장소 (AppContext 사용)
        context = AppContext(
            file_path=file_path,
            user_request=user_request
        )

        for step_num in sorted(self._step_definitions.keys()):
            step_instance = self._get_step_instance(step_num)
            step_name = step_instance.step_name
            self.logger.info(f"\n>>>>> 단계 {step_num}: {step_name} 실행 시작 <<<<<")
            
            try:
                # 각 단계는 업데이트된 context를 반환하고, 이를 다음 단계로 전달합니다.
                context = await step_instance.run(context)
                
            except Exception as e:
                self.logger.error(f"!!!!!! 단계 {step_num}: {step_name} 실행 중 심각한 오류 발생 !!!!!!", exc_info=True)
                self.logger.error(f"오류 메시지: {e}")
                self.logger.error("워크플로우를 중단합니다.")
                # 오류 발생 시에도 현재까지의 컨텍스트를 반환하여 디버깅에 활용
                return context
            
            self.logger.info(f">>>>> 단계 {step_num}: {step_name} 실행 완료 <<<<<")

        self.logger.info("="*20 + " 전체 워크플로우 성공적으로 완료 " + "="*20)
        
        final_report_path = context.get("final_report_path")
        if not final_report_path:
            self.logger.error("최종 보고서 경로가 결과에 포함되지 않았습니다.")
        
        return context 