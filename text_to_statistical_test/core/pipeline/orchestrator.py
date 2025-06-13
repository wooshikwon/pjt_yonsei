# 파일명: core/pipeline/orchestrator.py
import asyncio
import logging
import importlib
from typing import Dict, Any

from core.pipeline.app_context import AppContext
from core.pipeline.step0_pipeline import PipelineStep
# [수정] 모든 서비스 인스턴스를 Orchestrator 레벨에서 임포트
from services import llm_service, rag_service, report_service
from services.statistics.stats_service import StatisticsService

class Orchestrator:
    """프로젝트의 핵심 워크플로우 실행을 관리하는 지휘자."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # [수정] _step_definitions는 실제 프로젝트에 맞게 채워져 있다고 가정
        self._step_definitions: Dict[int, tuple[str, str]] = {
            1: ('core.pipeline.step1_data_selection', 'DataSelectionStep'),
            2: ('core.pipeline.step2_autonomous_analysis', 'AutonomousAnalysisStep'),
            3: ('core.pipeline.step3_reporting', 'ReportingStep'),
        }
        
        # [수정] 모든 서비스 인스턴스를 Orchestrator가 소유하고 생명주기를 관리
        self.llm_service = llm_service
        self.rag_service = rag_service
        self.report_service = report_service
        self.stats_service = StatisticsService() # StatisticsService도 여기서 생성
        self.logger.info("Orchestrator가 모든 서비스와 함께 초기화되었습니다.")

    def _get_step_instance(self, step_num: int) -> PipelineStep:
        try:
            module_name, class_name = self._step_definitions[step_num]
            module = importlib.import_module(module_name)
            step_class = getattr(module, class_name)

            # [수정] 단계에 따라 필요한 서비스를 명시적으로 주입 (Dependency Injection)
            if class_name == 'AutonomousAnalysisStep':
                # [개선] Agent가 사용하는 stats_service도 함께 주입
                return step_class(
                    llm_service=self.llm_service, 
                    rag_service=self.rag_service,
                    stats_service=self.stats_service 
                )
            elif class_name == 'ReportingStep':
                return step_class(report_service=self.report_service)
            else: # DataSelectionStep 등은 서비스가 필요 없음
                return step_class()
        except (KeyError, AttributeError, ImportError) as e:
            self.logger.error(f"파이프라인 단계({step_num})를 로드하거나 인스턴스화할 수 없습니다: {e}", exc_info=True)
            raise RuntimeError(f"Pipeline step {step_num} could not be loaded.") from e

    async def run(self, file_path: str, user_request: str) -> AppContext:
        self.logger.info("="*20 + " 전체 워크플로우 시작 " + "="*20)
        context = AppContext(file_path=file_path, user_request=user_request)

        for step_num in sorted(self._step_definitions.keys()):
            try:
                step_instance = self._get_step_instance(step_num)
                self.logger.info(f"\n>>>>> 단계 {step_num}: {step_instance.step_name} 실행 시작 <<<<<")
                context = await step_instance.run(context)
                self.logger.info(f">>>>> 단계 {step_num}: {step_instance.step_name} 실행 완료 <<<<<")
            except Exception as e:
                error_message = f"!!!!!! 단계 {step_num}: {step_instance.step_name if 'step_instance' in locals() else 'Unknown'} 실행 중 심각한 오류 발생: {e} !!!!!!"
                self.logger.error(error_message, exc_info=True)
                context.error = error_message # 에러 정보를 컨텍스트에 기록
                self.logger.error("워크플로우를 중단합니다.")
                return context

        self.logger.info("="*20 + " 전체 워크플로우 성공적으로 완료 " + "="*20)
        
        # [수정] @dataclass 필드에 직접 접근. 더 안전하고 명확함.
        if not context.final_report_path:
            self.logger.warning("최종 보고서 경로가 결과에 포함되지 않았습니다.")
        
        return context