import io
import contextlib
import traceback
from typing import Tuple, Dict, Any

class CodeExecutor:
    """
    LLM 에이전트가 생성한 Python 코드를 안전하게 실행하고,
    그 결과를 캡처하는 격리된 실행기입니다.
    """

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
        output_buffer = io.StringIO()
        
        # 실행 환경 준비. 제공된 global_vars가 없으면 빈 딕셔너리 사용.
        execution_globals = global_vars or {}

        try:
            with contextlib.redirect_stdout(output_buffer):
                # exec() 함수는 동적으로 코드 실행을 지원합니다.
                # 제공된 전역 네임스페이스 내에서 코드를 실행합니다.
                exec(code, execution_globals)
            
            captured_output = output_buffer.getvalue()
            return captured_output, True
        
        except Exception:
            # 실행 중 예외가 발생하면 전체 트레이스백을 캡처합니다.
            error_message = traceback.format_exc()
            return error_message, False 