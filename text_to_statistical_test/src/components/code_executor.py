import sys
from typing import Dict, Any, Tuple
import io
import contextlib
import pandas as pd

# 통합된 경고 설정 사용
from src.utils.warnings_config import suppress_warnings

class CodeExecutor:
    """
    LLM 에이전트가 생성한 Python 코드를 안전하게 실행하고,
    그 결과를 캡처하는 격리된 실행기입니다.
    """

    def __init__(self):
        pass

    def run(self, code: str, global_vars: Dict[str, Any] = None) -> Tuple[str, str, pd.DataFrame]:
        """
        주어진 코드 문자열을 실행하고, 계약에 따라 상태를 판별합니다.

        Args:
            code (str): 실행할 Python 코드 문자열.
            global_vars (Dict[str, Any], optional): 코드 실행 환경에 주입할 전역 변수.

        Returns:
            Tuple[str, str, pd.DataFrame]: (캡처된 출력, 실행 상태, 최종 데이터프레임).
                                           상태: 'SUCCESS', 'SKIPPED', 'ERROR'
        """
        if global_vars is None:
            global_vars = {}

        captured_output = io.StringIO()
        status = 'SUCCESS'
        
        try:
            with suppress_warnings(), contextlib.redirect_stdout(captured_output):
                execution_globals = global_vars.copy()
                exec(code, execution_globals)

            result = captured_output.getvalue()
            
            # SKIPPED 상태 계약 확인
            if result.strip().startswith('###STATUS:SKIPPED###'):
                status = 'SKIPPED'
                # 매직 문자열 및 그 다음 줄바꿈까지 제거하여 순수한 이유만 남김
                reason = result.replace('###STATUS:SKIPPED###\n', '', 1)
                result = f"Step skipped: {reason}"

            # 변경된 'df'를 가져옴
            final_df = execution_globals.get('df')
            
            if not isinstance(final_df, pd.DataFrame):
                final_df = None
                
            return result if result else "Code executed successfully.", status, final_df

        except Exception as e:
            # 오류가 발생한 경우 오류 메시지 반환
            error_message = captured_output.getvalue()
            if not error_message:
                error_message = str(e)
            return f"Traceback (most recent call last):\n{error_message}", 'ERROR', None 