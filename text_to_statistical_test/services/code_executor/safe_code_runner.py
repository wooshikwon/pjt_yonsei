"""
Safe Code Runner

LLM이 생성하거나 참조한 코드의 안전한 실행 환경
- 샌드박스 환경에서의 안전한 코드 실행
- 리소스 제한 및 보안 관리
- 실행 결과 및 오류 처리
"""

import sys
import os
import subprocess
import signal
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import threading
import queue
import time
import traceback
import ast
import re

# 조건부 임포트
try:
    from RestrictedPython import compile_restricted
    from RestrictedPython.Guards import safe_globals, safe_builtins
    from RestrictedPython.transformer import RestrictingNodeTransformer
    RESTRICTED_PYTHON_AVAILABLE = True
except ImportError:
    RESTRICTED_PYTHON_AVAILABLE = False
    compile_restricted = None
    safe_globals = None
    safe_builtins = None
    RestrictingNodeTransformer = None

import io
import resource
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import pandas as pd
import numpy as np

from utils.error_handler import handle_error, ErrorHandler, StatisticsException


@dataclass
class ExecutionResult:
    """코드 실행 결과"""
    success: bool
    output: str
    error: Optional[str]
    execution_time: float
    memory_usage: int
    variables: Dict[str, Any]
    plots: List[str]  # 생성된 플롯 파일 경로들
    warnings: List[str]


@dataclass
class SecurityConfig:
    """보안 설정"""
    max_execution_time: int = 30  # 최대 실행 시간 (초)
    max_memory_mb: int = 512  # 최대 메모리 사용량 (MB)
    allowed_imports: List[str] = None  # 허용된 import 모듈들
    forbidden_functions: List[str] = None  # 금지된 함수들
    enable_file_access: bool = False  # 파일 시스템 접근 허용 여부
    enable_network_access: bool = False  # 네트워크 접근 허용 여부


class SafeCodeRunner:
    """안전한 코드 실행 환경"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """
        SafeCodeRunner 초기화
        
        Args:
            config: 보안 설정
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or SecurityConfig()
        
        # 기본 허용 모듈 설정
        if self.config.allowed_imports is None:
            self.config.allowed_imports = [
                'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
                'plotly', 'statsmodels', 'sklearn', 'math', 'statistics',
                'datetime', 'collections', 'itertools', 'functools',
                'warnings', 'json', 're'
            ]
        
        # 기본 금지 함수 설정
        if self.config.forbidden_functions is None:
            self.config.forbidden_functions = [
                'exec', 'eval', 'compile', 'open', '__import__',
                'input', 'raw_input', 'file', 'reload', 'vars',
                'globals', 'locals', 'dir', 'hasattr', 'getattr',
                'setattr', 'delattr', 'exit', 'quit'
            ]
        
        # 안전한 전역 네임스페이스 설정
        self.safe_globals = self._create_safe_globals()
        
        # 실행 통계
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'security_violations': 0
        }
    
    def execute_code(self, code: str, context: Optional[Dict[str, Any]] = None,
                    output_dir: Optional[Path] = None) -> ExecutionResult:
        """
        코드 안전 실행
        
        Args:
            code: 실행할 Python 코드
            context: 실행 컨텍스트 (변수, 데이터 등)
            output_dir: 출력 파일 저장 디렉토리
            
        Returns:
            ExecutionResult: 실행 결과
        """
        self.logger.info("코드 실행 시작")
        start_time = time.time()
        
        try:
            # 1. 코드 보안 검증
            security_check = self._validate_code_security(code)
            if not security_check['is_safe']:
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"보안 위반: {security_check['reason']}",
                    execution_time=0,
                    memory_usage=0,
                    variables={},
                    plots=[],
                    warnings=[]
                )
            
            # 2. 실행 환경 준비
            execution_env = self._prepare_execution_environment(context, output_dir)
            
            # 3. 리소스 제한 설정
            with self._resource_limits():
                # 4. 코드 컴파일 및 실행
                result = self._execute_in_sandbox(code, execution_env)
            
            # 5. 실행 통계 업데이트
            self.execution_stats['total_executions'] += 1
            if result.success:
                self.execution_stats['successful_executions'] += 1
            else:
                self.execution_stats['failed_executions'] += 1
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            self.logger.info(f"코드 실행 완료 - 성공: {result.success}, 시간: {execution_time:.2f}초")
            return result
            
        except Exception as e:
            self.logger.error(f"코드 실행 중 오류: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=f"실행 오류: {str(e)}",
                execution_time=time.time() - start_time,
                memory_usage=0,
                variables={},
                plots=[],
                warnings=[]
            )
    
    def execute_statistical_analysis(self, analysis_type: str, data: pd.DataFrame,
                                   parameters: Dict[str, Any]) -> ExecutionResult:
        """
        통계 분석 실행
        
        Args:
            analysis_type: 분석 유형 (t_test, anova, regression 등)
            data: 분석 데이터
            parameters: 분석 파라미터
            
        Returns:
            ExecutionResult: 실행 결과
        """
        try:
            # 분석 유형에 따른 코드 생성
            code = self._generate_analysis_code(analysis_type, parameters)
            
            # 실행 컨텍스트 준비
            context = {
                'data': data,
                'parameters': parameters,
                'analysis_type': analysis_type
            }
            
            return self.execute_code(code, context)
            
        except Exception as e:
            self.logger.error(f"통계 분석 실행 오류: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=f"통계 분석 오류: {str(e)}",
                execution_time=0,
                memory_usage=0,
                variables={},
                plots=[],
                warnings=[]
            )
    
    def validate_code_syntax(self, code: str) -> Dict[str, Any]:
        """
        코드 구문 검증
        
        Args:
            code: 검증할 코드
            
        Returns:
            Dict: 검증 결과
        """
        try:
            # AST 파싱으로 구문 검증
            ast.parse(code)
            return {
                'is_valid': True,
                'error': None,
                'suggestions': []
            }
            
        except SyntaxError as e:
            return {
                'is_valid': False,
                'error': f"구문 오류: {str(e)}",
                'suggestions': self._get_syntax_suggestions(e)
            }
        except Exception as e:
            return {
                'is_valid': False,
                'error': f"검증 오류: {str(e)}",
                'suggestions': []
            }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """실행 통계 반환"""
        return self.execution_stats.copy()
    
    def reset_stats(self):
        """실행 통계 초기화"""
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'security_violations': 0
        }
    
    def _validate_code_security(self, code: str) -> Dict[str, Any]:
        """코드 보안 검증"""
        try:
            # AST 파싱
            tree = ast.parse(code)
            
            # 보안 위험 요소 검사
            security_visitor = SecurityVisitor(self.config)
            security_visitor.visit(tree)
            
            if security_visitor.violations:
                self.execution_stats['security_violations'] += 1
                return {
                    'is_safe': False,
                    'reason': '; '.join(security_visitor.violations)
                }
            
            return {'is_safe': True, 'reason': None}
            
        except Exception as e:
            return {
                'is_safe': False,
                'reason': f"보안 검증 오류: {str(e)}"
            }
    
    def _prepare_execution_environment(self, context: Optional[Dict[str, Any]],
                                     output_dir: Optional[Path]) -> Dict[str, Any]:
        """실행 환경 준비"""
        env = self.safe_globals.copy()
        
        # 컨텍스트 추가
        if context:
            for key, value in context.items():
                if self._is_safe_value(value):
                    env[key] = value
        
        # 출력 디렉토리 설정
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            env['OUTPUT_DIR'] = str(output_dir)
        
        # 플롯 저장을 위한 리스트
        env['_plots'] = []
        env['_warnings'] = []
        
        return env
    
    @contextmanager
    def _resource_limits(self):
        """리소스 제한 설정"""
        # 메모리 제한 설정
        memory_limit = self.config.max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        
        # 시간 제한을 위한 알람 설정
        def timeout_handler(signum, frame):
            raise TimeoutError(f"실행 시간 초과 ({self.config.max_execution_time}초)")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.config.max_execution_time)
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def _execute_in_sandbox(self, code: str, env: Dict[str, Any]) -> ExecutionResult:
        """샌드박스에서 코드 실행"""
        # 출력 캡처를 위한 버퍼
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        try:
            # RestrictedPython으로 코드 컴파일
            compiled_code = compile_restricted(code, '<string>', 'exec')
            if compiled_code is None:
                return ExecutionResult(
                    success=False,
                    output="",
                    error="코드 컴파일 실패",
                    execution_time=0,
                    memory_usage=0,
                    variables={},
                    plots=[],
                    warnings=[]
                )
            
            # 메모리 사용량 측정 시작
            initial_memory = self._get_memory_usage()
            
            # 출력 리다이렉션과 함께 실행
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(compiled_code, env)
            
            # 메모리 사용량 계산
            final_memory = self._get_memory_usage()
            memory_usage = max(0, final_memory - initial_memory)
            
            # 결과 변수 추출
            result_vars = {k: v for k, v in env.items() 
                          if not k.startswith('_') and not callable(v)}
            
            return ExecutionResult(
                success=True,
                output=stdout_buffer.getvalue(),
                error=stderr_buffer.getvalue() if stderr_buffer.getvalue() else None,
                execution_time=0,  # 호출자에서 설정
                memory_usage=memory_usage,
                variables=result_vars,
                plots=env.get('_plots', []),
                warnings=env.get('_warnings', [])
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=stdout_buffer.getvalue(),
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                execution_time=0,
                memory_usage=0,
                variables={},
                plots=[],
                warnings=[]
            )
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """안전한 전역 네임스페이스 생성"""
        # RestrictedPython이 사용 가능한 경우
        if RESTRICTED_PYTHON_AVAILABLE and safe_globals is not None:
            safe_globals_dict = safe_globals.copy()
            safe_builtins_dict = safe_builtins.copy()
        else:
            # RestrictedPython이 없는 경우 기본 안전한 환경 구성
            safe_globals_dict = {}
            safe_builtins_dict = {
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'min': min,
                'max': max,
                'sum': sum,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'enumerate': enumerate,
                'zip': zip,
                'range': range,
                'print': print
            }
        
        # 허용된 모듈들 추가
        allowed_modules = {}
        for module_name in self.config.allowed_imports:
            try:
                if module_name == 'pandas':
                    import pandas as pd
                    allowed_modules['pd'] = pd
                    allowed_modules['pandas'] = pd
                elif module_name == 'numpy':
                    import numpy as np
                    allowed_modules['np'] = np
                    allowed_modules['numpy'] = np
                elif module_name == 'scipy':
                    import scipy
                    allowed_modules['scipy'] = scipy
                elif module_name == 'matplotlib':
                    import matplotlib.pyplot as plt
                    allowed_modules['plt'] = plt
                    allowed_modules['matplotlib'] = plt
                elif module_name == 'seaborn':
                    import seaborn as sns
                    allowed_modules['sns'] = sns
                    allowed_modules['seaborn'] = sns
                elif module_name == 'plotly':
                    import plotly
                    allowed_modules['plotly'] = plotly
                elif module_name == 'statsmodels':
                    import statsmodels
                    allowed_modules['statsmodels'] = statsmodels
                elif module_name == 'sklearn':
                    import sklearn
                    allowed_modules['sklearn'] = sklearn
                else:
                    # 기타 모듈들
                    module = __import__(module_name)
                    allowed_modules[module_name] = module
            except ImportError:
                self.logger.warning(f"모듈 {module_name}을 import할 수 없습니다.")
        
        # 금지된 함수들 제거
        for func_name in self.config.forbidden_functions:
            safe_builtins_dict.pop(func_name, None)
        
        # 추가 안전한 함수들
        safe_builtins_dict.update({
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'min': min,
            'max': max,
            'sum': sum,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'enumerate': enumerate,
            'zip': zip,
            'range': range,
            'print': print
        })
        
        safe_globals_dict.update({
            '__builtins__': safe_builtins_dict,
            **allowed_modules
        })
        
        return safe_globals_dict
    
    def _is_safe_value(self, value: Any) -> bool:
        """값이 안전한지 검증"""
        # 기본 타입들은 안전
        safe_types = (str, int, float, bool, list, dict, tuple, set, type(None))
        if isinstance(value, safe_types):
            return True
        
        # pandas, numpy 객체들은 안전
        if hasattr(value, '__module__'):
            module = value.__module__
            if module and any(safe_mod in module for safe_mod in ['pandas', 'numpy', 'scipy']):
                return True
        
        return False
    
    def _get_memory_usage(self) -> int:
        """현재 메모리 사용량 반환 (바이트)"""
        try:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except:
            return 0
    
    def _generate_analysis_code(self, analysis_type: str, parameters: Dict[str, Any]) -> str:
        """분석 유형에 따른 코드 생성"""
        code_templates = {
            't_test': """
# 독립표본 t-검정
from scipy import stats
import pandas as pd

group1 = data[data['{group_col}'] == '{group1_val}']['{value_col}']
group2 = data[data['{group_col}'] == '{group2_val}']['{value_col}']

statistic, p_value = stats.ttest_ind(group1, group2)
result = {{
    'statistic': statistic,
    'p_value': p_value,
    'group1_mean': group1.mean(),
    'group2_mean': group2.mean(),
    'group1_std': group1.std(),
    'group2_std': group2.std()
}}
""",
            'anova': """
# 일원분산분석 (ANOVA)
from scipy import stats
import pandas as pd

groups = []
for group_val in data['{group_col}'].unique():
    group_data = data[data['{group_col}'] == group_val]['{value_col}']
    groups.append(group_data)

statistic, p_value = stats.f_oneway(*groups)
result = {{
    'statistic': statistic,
    'p_value': p_value,
    'group_means': [group.mean() for group in groups],
    'group_stds': [group.std() for group in groups]
}}
""",
            'correlation': """
# 상관분석
import pandas as pd
from scipy import stats

correlation, p_value = stats.pearsonr(data['{var1}'], data['{var2}'])
result = {{
    'correlation': correlation,
    'p_value': p_value,
    'var1_mean': data['{var1}'].mean(),
    'var2_mean': data['{var2}'].mean()
}}
"""
        }
        
        template = code_templates.get(analysis_type, "# 지원되지 않는 분석 유형")
        return template.format(**parameters)
    
    def _get_syntax_suggestions(self, error: SyntaxError) -> List[str]:
        """구문 오류에 대한 제안사항 생성"""
        suggestions = []
        error_msg = str(error).lower()
        
        if 'invalid syntax' in error_msg:
            suggestions.append("구문을 확인하고 괄호, 따옴표 등이 올바르게 닫혔는지 확인하세요.")
        if 'indentation' in error_msg:
            suggestions.append("들여쓰기를 확인하세요. Python은 일관된 들여쓰기가 필요합니다.")
        if 'unexpected eof' in error_msg:
            suggestions.append("코드가 완전하지 않습니다. 누락된 부분이 있는지 확인하세요.")
        
        return suggestions


class SecurityVisitor(ast.NodeVisitor):
    """코드 보안 검증을 위한 AST 방문자"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.violations = []
    
    def visit_Import(self, node):
        """import 문 검증"""
        for alias in node.names:
            if alias.name not in self.config.allowed_imports:
                self.violations.append(f"허용되지 않은 모듈 import: {alias.name}")
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """from ... import 문 검증"""
        if node.module and node.module not in self.config.allowed_imports:
            self.violations.append(f"허용되지 않은 모듈 import: {node.module}")
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """함수 호출 검증"""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.config.forbidden_functions:
                self.violations.append(f"금지된 함수 호출: {func_name}")
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        """속성 접근 검증"""
        # 파일 시스템 접근 검증
        if not self.config.enable_file_access:
            if isinstance(node.value, ast.Name) and node.value.id == 'os':
                self.violations.append("파일 시스템 접근이 금지되어 있습니다.")
        
        # 네트워크 접근 검증
        if not self.config.enable_network_access:
            network_modules = ['urllib', 'requests', 'socket', 'http']
            if isinstance(node.value, ast.Name) and node.value.id in network_modules:
                self.violations.append("네트워크 접근이 금지되어 있습니다.")
        
        self.generic_visit(node) 