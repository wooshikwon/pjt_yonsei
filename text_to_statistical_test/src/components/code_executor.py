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

    def run(self, code: str, global_vars: Dict[str, Any] = None) -> Tuple[str, bool, pd.DataFrame]:
        """
        주어진 코드 문자열을 실행하고 표준 출력 또는 오류를 캡처합니다.
        또한, 코드 실행 후 'df_result' 변수에 할당된 최종 데이터프레임을 반환합니다.

        Args:
            code (str): 실행할 Python 코드 문자열.
            global_vars (Dict[str, Any], optional): 코드 실행 환경에 주입할 전역 변수.
                                                     주로 {'df': pandas_dataframe} 형태로 사용됩니다.

        Returns:
            Tuple[str, bool, pd.DataFrame]: (캡처된 출력/오류, 성공 여부, 최종 데이터프레임).
                                            상태 변경이 없거나 실패 시 데이터프레임은 None입니다.
        """
        if global_vars is None:
            global_vars = {}

        captured_output = io.StringIO()
        
        try:
            with suppress_warnings(), contextlib.redirect_stdout(captured_output):
                execution_globals = global_vars.copy()
                exec(code, execution_globals)

            result = captured_output.getvalue()
            
            # 'df_result' 계약에 따라 최종 df를 가져옴
            final_df = execution_globals.get('df_result')
            
            # df_result가 DataFrame이 아니면 None으로 처리
            if not isinstance(final_df, pd.DataFrame):
                final_df = None
                
            return result if result else "Code executed successfully.", True, final_df

        except Exception as e:
            # 오류가 발생한 경우 오류 메시지 반환
            error_message = captured_output.getvalue()
            if not error_message:
                error_message = str(e)
            return f"Traceback (most recent call last):\n{error_message}", False, None 