import pytest
import sys
import os

# 테스트 대상 모듈을 import하기 위해 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.components.code_executor import CodeExecutor

def test_run_success():
    """
    CodeExecutor가 유효한 코드를 성공적으로 실행하는지 테스트합니다.
    """
    # 준비
    executor = CodeExecutor()
    code = "a = 5\nb = 10\nprint(a + b)"
    
    # 실행
    result, success = executor.run(code)
    
    # 검증
    assert success is True
    assert result == '15\n'

def test_run_failure():
    """
    CodeExecutor가 유효하지 않은 코드를 실행했을 때 실패를 올바르게 처리하는지 테스트합니다.
    """
    # 준비
    executor = CodeExecutor()
    code = "print(undefined_variable)"
    
    # 실행
    result, success = executor.run(code)
    
    # 검증
    assert success is False
    assert ('NameError' in result or 'undefined_variable' in result)

def test_run_with_global_vars():
    """
    CodeExecutor가 global_vars를 사용하여 외부 변수에 접근할 수 있는지 테스트합니다.
    """
    # 준비
    executor = CodeExecutor()
    import pandas as pd
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    code = "print(df['col1'].sum())"
    
    # 실행
    result, success = executor.run(code, global_vars={'df': df})
    
    # 검증
    assert success is True
    assert result == '3\n' 