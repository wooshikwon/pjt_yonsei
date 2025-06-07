# 파일명: core/pipeline/code_execution_step.py
from typing import Dict, Any

from core.context import AppContext
from core.pipeline.pipeline_step import PipelineStep
from services.code_executor.safe_code_runner import SafeCodeRunner

class CodeExecutionStep(PipelineStep):
    """
    4단계: 생성된 통계 분석 코드를 안전한 환경에서 실행합니다.
    """
    def __init__(self):
        super().__init__("Code Execution")
        # SafeCodeRunner는 실행 시점에 context의 dataframe을 사용하여 초기화됩니다.

    async def run(self, context: AppContext) -> AppContext:
        """
        자율 분석 단계에서 생성된 코드를 실행하고 결과를 context에 추가합니다.
        """
        self.logger.info("생성된 코드 실행을 시작합니다...")

        generated_code = context.get("generated_code")
        dataframe = context.get("dataframe")
        
        if not generated_code:
            raise ValueError("실행할 코드가 생성되지 않았습니다.")
        if dataframe is None:
            raise ValueError("코드 실행을 위한 데이터프레임이 없습니다.")

        session_id = context.get("session_id", "default_session")
        code_runner = SafeCodeRunner(session_id=session_id, df=dataframe)

        try:
            execution_results = code_runner.run(generated_code)
            
            context.set("execution_results", execution_results)
            self.logger.info("코드 실행이 성공적으로 완료되었습니다.")
            self.logger.debug(f"실행 결과: {execution_results}")

        except Exception as e:
            self.logger.error(f"코드 실행 중 오류 발생: {e}", exc_info=True)
            context.set("error", str(e))
            raise e

        return context 