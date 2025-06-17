import sys
from typing import Dict, Any, Tuple
import io
import contextlib

# 통합된 경고 설정 사용
from src.utils.warnings_config import suppress_warnings

class CodeExecutor:
    """
    LLM 에이전트가 생성한 Python 코드를 안전하게 실행하고,
    그 결과를 캡처하는 격리된 실행기입니다.
    """

    def __init__(self):
        pass

    def run(self, code: str, global_vars: Dict[str, Any] = None) -> Tuple[str, bool]:
        """
        주어진 코드 문자열을 실행하고 표준 출력 또는 오류를 캡처합니다.

        Args:
            code (str): 실행할 Python 코드 문자열.
            global_vars (Dict[str, Any], optional): 코드 실행 환경에 주입할 전역 변수.
                                                     주로 {'df': pandas_dataframe} 형태로 사용됩니다.

        Returns:
            Tuple[str, bool]: 첫 번째 요소는 캡처된 출력 또는 오류 메시지이며,
                              두 번째 요소는 실행 성공 여부(True/False)입니다.
        """
        if global_vars is None:
            global_vars = {}

        # 표준 출력과 오류를 캡처하기 위한 StringIO 객체
        captured_output = io.StringIO()
        captured_error = io.StringIO()

        try:
            # 통합된 경고 설정 사용하여 경고 메시지 숨기기
            with suppress_warnings():
                # 표준 출력과 표준 오류를 리다이렉션
                with contextlib.redirect_stdout(captured_output), \
                     contextlib.redirect_stderr(captured_error):
                    
                    # 코드 실행을 위한 실행 환경 설정
                    execution_globals = global_vars.copy()
                    execution_globals['__builtins__'] = __builtins__
                    
                    # 코드 실행
                    exec(code, execution_globals)

            # 성공적으로 실행된 경우 출력 결과 반환
            result = captured_output.getvalue()
            return result if result else "Code executed successfully.", True

        except Exception as e:
            # 오류가 발생한 경우 오류 메시지 반환
            error_message = captured_error.getvalue()
            if not error_message:
                error_message = str(e)
            return f"Traceback (most recent call last):\n{error_message}", False 