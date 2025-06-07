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

from utils import StatisticalException, BaseAppException
from .restricted_env.exec_within_docker import ExecWithinDocker

logger = logging.getLogger(__name__)

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
    """
    LLM이 생성한 통계 분석 코드를 안전한 환경에서 실행하는 클래스.
    - Docker 컨테이너 내에서 코드를 실행하여 시스템 격리
    - 실행 결과, 생성된 데이터, 시각화 파일 경로 등을 반환
    """
    def __init__(self, session_id: str, df: pd.DataFrame):
        self.session_id = session_id
        self.df = df
        self.code_runner = ExecWithinDocker(session_id=session_id, df=df)
        logger.info(f"[{session_id}] SafeCodeRunner initialized with a dataframe of shape {df.shape}.")

    def run(self, code: str) -> Dict[str, Any]:
        """
        주어진 코드를 Docker 컨테이너 내에서 실행하고 결과를 반환합니다.
        """
        if not code:
            raise ValueError("실행할 코드가 비어있습니다.")

        try:
            logger.info(f"[{self.session_id}] Executing code in a restricted environment...")
            result_dict = self.code_runner.run(code)
            logger.info(f"[{self.session_id}] Code execution completed successfully.")
            return result_dict
        except Exception as e:
            logger.error(f"[{self.session_id}] An error occurred during code execution: {e}")
            # 여기서 에러를 다시 발생시키거나, 혹은 처리된 형태로 반환할 수 있습니다.
            # 예를 들어, 통계 관련 에러로 래핑하여 상위로 전달합니다.
            raise StatisticalException(f"코드 실행 중 오류가 발생했습니다: {e}")

def handle_error(e: Exception, message: str, should_raise: bool = True):
    """오류를 로깅하고 필요시 다시 발생시키는 간단한 헬퍼 함수"""
    logger.error(f"{message}: {e}")
    if should_raise:
        raise BaseAppException(f"{message}: {e}")


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