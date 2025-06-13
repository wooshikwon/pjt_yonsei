# 파일명: core/pipeline/reporting_step.py

import logging
from core.pipeline.app_context import AppContext
from core.pipeline.step0_pipeline import PipelineStep
from services.reporting.report_service import ReportService

class ReportingStep(PipelineStep):
    """3단계: 모든 분석 결과를 종합하여 최종 Markdown 보고서를 생성합니다."""
    
    def __init__(self, report_service: ReportService):
        super().__init__("최종 보고서 생성")
        self.logger = logging.getLogger(__name__)
        self.report_service = report_service

    async def run(self, context: AppContext) -> AppContext:
        self.logger.info("최종 Markdown 보고서 생성을 시작합니다...")

        # [수정] @dataclass 필드에 직접 접근
        final_summary = context.final_summary
        execution_results = context.execution_results

        if not final_summary or not execution_results:
            raise ValueError("ReportingStep: 컨텍스트에 'final_summary' 또는 'execution_results'가 없습니다.")

        try:
            # ReportService의 메서드 호출은 변경 없음
            report_path, markdown_content = self.report_service.create_markdown_report(
                final_summary=final_summary,
                execution_results=execution_results
            )
            
            self.logger.info(f"Markdown 보고서를 성공적으로 생성했습니다: {report_path}")
            # [수정] @dataclass 필드에 직접 할당
            context.final_report_path = report_path
            context.final_report_content = markdown_content
        except Exception as e:
            self.logger.error(f"보고서 생성 중 오류 발생: {e}", exc_info=True)
            raise IOError("최종 보고서를 생성하는 데 실패했습니다.") from e

        return context