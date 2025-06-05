"""
SafeCodeExecutor: Enhanced RAG 기반 안전한 통계 코드 실행

비즈니스 컨텍스트 인식 AI 통계 분석 시스템에서 생성된 Python 통계 코드를
최대한 안전한 방식으로 실행하고, 실행 결과를 캡처하여 반환합니다.

Enhanced RAG 시스템 통합 특징:
- 비즈니스 컨텍스트 인식 실행 환경
- RAG 기반 오류 분석 및 해결 제안
- 멀티턴 대화 세션 지원
- 데이터베이스 스키마 컨텍스트 활용
- 실행 결과의 비즈니스 인사이트 추출

보안 특징:
- 다중 레벨 보안 검증 (AST, 패턴, 샌드박스)
- 리소스 제한 (메모리, 시간, 파일 접근)
- 화이트리스트 기반 모듈 허용
- 실행 컨텍스트 격리
"""

import logging
import subprocess
import tempfile
import sys
import io
import contextlib
import time
import signal
import os
import resource
import ast
import re
import uuid
import shutil
import hashlib
import traceback
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 백엔드를 Agg로 설정 (GUI 없이 사용)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import warnings


@dataclass
class BusinessContext:
    """비즈니스 컨텍스트 정보"""
    domain: str  # 비즈니스 도메인 (e.g., finance, healthcare, retail)
    analysis_type: str  # 분석 유형 (descriptive, predictive, prescriptive)
    target_audience: str  # 대상 사용자 (executive, analyst, researcher)
    business_questions: List[str]  # 비즈니스 질문들
    key_metrics: List[str]  # 핵심 지표들
    constraints: Dict[str, Any]  # 제약사항들
    

@dataclass
class SchemaContext:
    """데이터베이스 스키마 컨텍스트"""
    table_schemas: Dict[str, List[str]]  # 테이블별 컬럼 정보
    relationships: Dict[str, List[str]]  # 테이블 간 관계
    data_types: Dict[str, Dict[str, str]]  # 컬럼별 데이터 타입
    business_meanings: Dict[str, str]  # 컬럼의 비즈니스 의미


@dataclass
class ExecutionSession:
    """실행 세션 정보"""
    session_id: str
    user_query: str
    conversation_history: List[Dict[str, str]]
    analysis_context: Dict[str, Any]
    previous_results: List['ExecutionResult']
    

class ExecutionResult:
    """Enhanced RAG 기반 코드 실행 결과를 담는 클래스"""
    
    def __init__(self):
        # 기본 실행 결과
        self.success: bool = False
        self.stdout: str = ""
        self.stderr: str = ""
        self.result_variables: Dict[str, Any] = {}
        self.generated_plots: List[str] = []
        self.execution_time: float = 0.0
        self.memory_usage: float = 0.0
        self.error_message: str = ""
        self.warning_messages: List[str] = []
        self.execution_id: str = str(uuid.uuid4())
        self.timestamp: str = datetime.now().isoformat()
        
        # Enhanced RAG 관련 결과
        self.business_insights: Dict[str, Any] = {}
        self.statistical_interpretation: Dict[str, Any] = {}
        self.next_analysis_suggestions: List[str] = []
        self.data_quality_issues: List[str] = []
        self.code_explanations: List[str] = []
        self.business_context_used: Optional[BusinessContext] = None
        self.schema_context_used: Optional[SchemaContext] = None
        self.session_info: Optional[ExecutionSession] = None
        
        # 코드 실행 메타데이터
        self.code_complexity_score: float = 0.0
        self.security_validation_results: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, float] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """결과를 딕셔너리로 변환"""
        result_dict = {
            'execution_id': self.execution_id,
            'timestamp': self.timestamp,
            'success': self.success,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'error_message': self.error_message,
            'warning_messages': self.warning_messages,
            'result_variables': self._make_serializable(self.result_variables),
            'generated_plots': self.generated_plots,
            'business_insights': self.business_insights,
            'statistical_interpretation': self.statistical_interpretation,
            'next_analysis_suggestions': self.next_analysis_suggestions,
            'data_quality_issues': self.data_quality_issues,
            'code_explanations': self.code_explanations,
            'code_complexity_score': self.code_complexity_score,
            'security_validation_results': self.security_validation_results,
            'performance_metrics': self.performance_metrics
        }
        
        if self.business_context_used:
            result_dict['business_context'] = asdict(self.business_context_used)
        if self.schema_context_used:
            result_dict['schema_context'] = asdict(self.schema_context_used)
        if self.session_info:
            result_dict['session_info'] = asdict(self.session_info)
            
        return result_dict
    
    def _make_serializable(self, obj: Any) -> Any:
        """객체를 JSON 직렬화 가능한 형태로 변환"""
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict() if hasattr(obj, 'to_dict') else str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj


class EnhancedSecurityValidator:
    """Enhanced RAG 시스템을 위한 강화된 보안 검증기"""
    
    def __init__(self):
        # 허용된 모듈 리스트 (화이트리스트) - RAG 시스템에 맞게 확장
        self.allowed_modules = {
            'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
            'statsmodels', 'sklearn', 'scikit-learn', 'math', 'statistics',
            'warnings', 'itertools', 'collections', 'functools',
            'operator', 're', 'json', 'datetime', 'pathlib',
            'random', 'typing', 'copy', 'pickle', 'plotly',
            'bokeh', 'altair', 'wordcloud', 'networkx'
        }
        
        # Enhanced RAG 분석에 특화된 허용 함수들
        self.rag_specific_allowed = {
            'describe', 'corr', 'crosstab', 'pivot_table', 'groupby',
            'value_counts', 'unique', 'nunique', 'isnull', 'isna',
            'fillna', 'dropna', 'merge', 'join', 'concat'
        }
        
        # 허용된 내장 함수들
        self.allowed_builtins = {
            'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'dir',
            'enumerate', 'filter', 'float', 'format', 'frozenset', 'hash',
            'hex', 'int', 'isinstance', 'issubclass', 'len', 'list', 'map',
            'max', 'min', 'oct', 'ord', 'pow', 'print', 'range', 'repr',
            'reversed', 'round', 'set', 'slice', 'sorted', 'str', 'sum',
            'tuple', 'type', 'zip', 'help'
        }
        
        # 금지된 함수/키워드 패턴 (블랙리스트)
        self.forbidden_patterns = [
            r'\b(exec|eval|compile|__import__|globals|locals|vars|dir)\b',
            r'\b(open|file|input|raw_input)\b(?!\s*\(.*\.csv|\.xlsx|\.json)',  # 데이터 파일 읽기는 허용
            r'\b(subprocess|os\.system|os\.popen|os\.spawn|os\.exec)\b',
            r'\bos\.(?!path|environ\[)',  # os 모듈 대부분 차단 (path, environ 일부 허용)
            r'\bsys\.(?!version|platform)',  # sys 모듈 대부분 차단
            r'\b(shutil|tempfile|pickle|marshal|types|code)\b',
            r'\b(__file__|__name__|__package__|__loader__|__spec__)\b',
            r'\bwhile\s+True\s*:',  # 무한 루프 방지
            r'\bfor\s+\w+\s+in\s+iter\(',  # 위험한 iterator 패턴
            r'\b(setattr|delattr|hasattr)\b',
            r'\b(lambda.*:.*exec|lambda.*:.*eval)\b',
            r'\bimport\s+(socket|urllib|requests|http)\b',  # 네트워크 접근 차단
        ]
        
        # RAG 분석에 특화된 안전한 패턴들
        self.safe_analysis_patterns = [
            r'\.describe\(\)',
            r'\.corr\(\)',
            r'\.plot\(',
            r'\.hist\(',
            r'\.scatter\(',
            r'\.boxplot\(',
            r'\.value_counts\(',
            r'stats\.\w+\(',
            r'sm\.\w+\(',
            r'sklearn\.\w+',
        ]
        
        # 위험한 AST 노드 타입들
        self.dangerous_ast_nodes = {
            ast.Delete, ast.Global, ast.Nonlocal,
        }
        
        # 비즈니스 컨텍스트별 추가 검증 규칙
        self.context_specific_rules = {
            'finance': {
                'required_validations': ['data_privacy_check', 'regulatory_compliance'],
                'additional_forbidden': [r'\b(insider|trading|confidential)\b']
            },
            'healthcare': {
                'required_validations': ['hipaa_compliance', 'data_anonymization'],
                'additional_forbidden': [r'\b(patient|medical_record|phi)\b']
            },
            'retail': {
                'required_validations': ['customer_privacy', 'gdpr_compliance'],
                'additional_forbidden': [r'\b(personal_data|customer_id)\b']
            }
        }
    
    def validate_code_safety(self, code: str, 
                           business_context: Optional[BusinessContext] = None,
                           schema_context: Optional[SchemaContext] = None) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Enhanced RAG 시스템을 위한 다중 레벨 코드 보안 검증
        
        Args:
            code: 검증할 코드 문자열
            business_context: 비즈니스 컨텍스트
            schema_context: 스키마 컨텍스트
            
        Returns:
            Tuple[bool, List[str], Dict[str, Any]]: (안전 여부, 위험 요소 목록, 검증 세부 정보)
        """
        violations = []
        validation_details = {
            'pattern_check': [],
            'ast_check': [],
            'import_check': [],
            'complexity_check': [],
            'context_check': [],
            'schema_check': []
        }
        
        # 1. 패턴 기반 검증
        pattern_violations = self._check_forbidden_patterns(code, business_context)
        violations.extend(pattern_violations)
        validation_details['pattern_check'] = pattern_violations
        
        # 2. AST 기반 검증
        ast_violations = self._check_ast_safety(code)
        violations.extend(ast_violations)
        validation_details['ast_check'] = ast_violations
        
        # 3. 모듈 import 검증
        import_violations = self._check_imports(code)
        violations.extend(import_violations)
        validation_details['import_check'] = import_violations
        
        # 4. 코드 복잡성 검증
        complexity_violations, complexity_score = self._check_code_complexity(code)
        violations.extend(complexity_violations)
        validation_details['complexity_check'] = complexity_violations
        validation_details['complexity_score'] = complexity_score
        
        # 5. 비즈니스 컨텍스트 기반 검증
        if business_context:
            context_violations = self._check_business_context_compliance(code, business_context)
            violations.extend(context_violations)
            validation_details['context_check'] = context_violations
        
        # 6. 스키마 컨텍스트 기반 검증
        if schema_context:
            schema_violations = self._check_schema_context_compliance(code, schema_context)
            violations.extend(schema_violations)
            validation_details['schema_check'] = schema_violations
        
        is_safe = len(violations) == 0
        return is_safe, violations, validation_details
    
    def _check_forbidden_patterns(self, code: str, business_context: Optional[BusinessContext] = None) -> List[str]:
        """금지된 패턴 검사"""
        violations = []
        
        for pattern in self.forbidden_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                violations.append(f"금지된 패턴 발견: {pattern} -> {matches}")
        
        # 비즈니스 컨텍스트별 추가 검증
        if business_context:
            for rule in self.context_specific_rules.get(business_context.domain, []):
                for forbidden in rule.get('additional_forbidden', []):
                    if re.search(forbidden, code, re.IGNORECASE):
                        violations.append(f"비즈니스 컨텍스트 위반: {forbidden}")
        
        return violations
    
    def _check_ast_safety(self, code: str) -> List[str]:
        """AST 기반 보안 검증"""
        violations = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # 위험한 노드 타입 검사
                if type(node) in self.dangerous_ast_nodes:
                    violations.append(f"위험한 AST 노드: {type(node).__name__}")
                
                # 함수 호출 검사
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name not in self.allowed_builtins:
                            # 허용되지 않은 내장 함수
                            violations.append(f"허용되지 않은 함수 호출: {func_name}")
                
                # Attribute 접근 검사
                elif isinstance(node, ast.Attribute):
                    attr_str = self._ast_to_string(node)
                    if not self._is_safe_attribute_access(attr_str):
                        violations.append(f"잠재적으로 위험한 attribute 접근: {attr_str}")
        
        except SyntaxError as e:
            violations.append(f"구문 오류: {e}")
        
        return violations
    
    def _check_imports(self, code: str) -> List[str]:
        """Import 문 검증"""
        violations = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        if module_name not in self.allowed_modules:
                            violations.append(f"허용되지 않은 모듈 import: {module_name}")
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        if module_name not in self.allowed_modules:
                            violations.append(f"허용되지 않은 모듈 from import: {module_name}")
        
        except SyntaxError:
            pass  # 이미 _check_ast_safety에서 처리됨
        
        return violations
    
    def _check_code_complexity(self, code: str) -> Tuple[List[str], float]:
        """코드 복잡성 검증"""
        violations = []
        
        lines = code.split('\n')
        
        # 라인 수 제한
        if len(lines) > 500:
            violations.append(f"코드가 너무 깁니다: {len(lines)} 라인 (최대 500라인)")
        
        # 중첩 레벨 검사
        max_indent = 0
        for line in lines:
            stripped = line.lstrip()
            if stripped:
                indent_level = (len(line) - len(stripped)) // 4
                max_indent = max(max_indent, indent_level)
        
        if max_indent > 6:
            violations.append(f"중첩 레벨이 너무 깊습니다: {max_indent} (최대 6)")
        
        # 루프 중첩 검사
        loop_keywords = ['for', 'while']
        nested_loops = 0
        for line in lines:
            stripped = line.strip()
            for keyword in loop_keywords:
                if stripped.startswith(keyword):
                    nested_loops += 1
        
        if nested_loops > 3:
            violations.append(f"루프가 너무 많습니다: {nested_loops}개 (최대 3개)")
        
        complexity_score = 1.0 - (len(violations) / 3)
        
        return violations, complexity_score
    
    def _ast_to_string(self, node: ast.AST) -> str:
        """AST 노드를 문자열로 변환"""
        try:
            return ast.unparse(node)
        except:
            return str(node)
    
    def _is_safe_attribute_access(self, attr_str: str) -> bool:
        """안전한 attribute 접근인지 확인"""
        for pattern in self.safe_analysis_patterns:
            if re.search(pattern, attr_str):
                return True
        
        # 허용된 모듈의 속성인지 확인
        for module in self.allowed_modules:
            if attr_str.startswith(f"{module}."):
                return True
        
        return False
    
    def _check_business_context_compliance(self, code: str, business_context: BusinessContext) -> List[str]:
        """비즈니스 컨텍스트 준수 검사"""
        violations = []
        
        # 필수 검증 항목 확인
        for required_validation in business_context.required_validations:
            if required_validation not in code:
                violations.append(f"필수 검증 항목 누락: {required_validation}")
        
        # 추가 검증 항목 확인
        for additional_forbidden in business_context.additional_forbidden:
            if re.search(additional_forbidden, code, re.IGNORECASE):
                violations.append(f"추가 검증 위반: {additional_forbidden}")
        
        return violations
    
    def _check_schema_context_compliance(self, code: str, schema_context: SchemaContext) -> List[str]:
        """스키마 컨텍스트 준수 검사"""
        violations = []
        
        # 테이블 스키마 검사
        for table, columns in schema_context.table_schemas.items():
            for column in columns:
                if f"{table}.{column}" not in code:
                    violations.append(f"테이블 스키마 위반: {table}.{column}")
        
        # 관계 스키마 검사
        for relationship, columns in schema_context.relationships.items():
            for column in columns:
                if f"{relationship}.{column}" not in code:
                    violations.append(f"관계 스키마 위반: {relationship}.{column}")
        
        # 데이터 타입 검사
        for column, data_type in schema_context.data_types.items():
            if column in code:
                if code[column] not in data_type:
                    violations.append(f"데이터 타입 위반: {column} -> {code[column]}")
        
        # 비즈니스 의미 검사
        for column, business_meaning in schema_context.business_meanings.items():
            if column in code:
                if business_meaning not in code:
                    violations.append(f"비즈니스 의미 위반: {column} -> {code[column]}")
        
        return violations


class ResourceManager:
    """실행 리소스 관리"""
    
    def __init__(self, max_memory_mb: int = 512, timeout_seconds: int = 30):
        self.max_memory_mb = max_memory_mb
        self.timeout_seconds = timeout_seconds
        self.temp_dirs: List[Path] = []
    
    def set_resource_limits(self):
        """리소스 제한 설정"""
        try:
            # 메모리 제한 (바이트 단위)
            max_memory_bytes = self.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
            
            # CPU 시간 제한
            resource.setrlimit(resource.RLIMIT_CPU, (self.timeout_seconds, self.timeout_seconds))
            
            # 파일 사이즈 제한 (100MB)
            max_file_size = 100 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_size, max_file_size))
            
            # 프로세스 수 제한
            resource.setrlimit(resource.RLIMIT_NPROC, (50, 50))
            
        except (OSError, ValueError) as e:
            logging.warning(f"리소스 제한 설정 실패: {e}")
    
    def create_temp_workspace(self) -> Path:
        """임시 작업 공간 생성"""
        temp_dir = Path(tempfile.mkdtemp(prefix="safe_code_exec_"))
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def cleanup_temp_workspaces(self):
        """임시 작업 공간 정리"""
        for temp_dir in self.temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logging.warning(f"임시 디렉토리 정리 실패: {temp_dir} - {e}")
        
        self.temp_dirs.clear()


class SafeCodeExecutor:
    """
    Enhanced RAG 기반 안전한 코드 실행 클래스
    
    비즈니스 컨텍스트 인식 AI 통계 분석에서 생성된 코드를
    다중 보안 레벨과 리소스 제한을 통해 안전하게 실행합니다.
    """
    
    def __init__(self, timeout_seconds: int = 30, max_memory_mb: int = 512,
                 enable_plots: bool = True, max_plots: int = 10):
        """
        SafeCodeExecutor 초기화
        
        Args:
            timeout_seconds: 코드 실행 시간 제한
            max_memory_mb: 메모리 사용량 제한 (MB)
            enable_plots: 플롯 생성 허용 여부
            max_plots: 최대 플롯 개수
        """
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        self.enable_plots = enable_plots
        self.max_plots = max_plots
        
        self.logger = logging.getLogger(__name__)
        
        # 보안 및 리소스 관리 초기화
        self.security_validator = EnhancedSecurityValidator()
        self.resource_manager = ResourceManager(max_memory_mb, timeout_seconds)
        
        # 실행 히스토리
        self.execution_history: List[ExecutionResult] = []
    
    def execute_code(self, code_string: str, 
                    input_dataframe: pd.DataFrame = None,
                    parameters: Dict[str, Any] = None,
                    business_context: Dict[str, Any] = None) -> ExecutionResult:
        """
        코드를 안전하게 실행
        
        Args:
            code_string: 실행할 Python 코드
            input_dataframe: 코드 내에서 'df' 변수로 사용될 DataFrame
            parameters: 코드 내 변수로 주입될 추가 파라미터
            business_context: 비즈니스 컨텍스트 정보
            
        Returns:
            ExecutionResult: 실행 결과
        """
        result = ExecutionResult()
        start_time = time.time()
        
        try:
            self.logger.info(f"코드 실행 시작: {result.execution_id}")
            
            # 1. 보안 검증
            is_safe, violations, validation_details = self.security_validator.validate_code_safety(code_string, business_context)
            if not is_safe:
                result.error_message = f"보안 검증 실패: {'; '.join(violations)}"
                result.success = False
                return result
            
            # 2. 코드 전처리
            processed_code = self._preprocess_code(code_string, business_context)
            
            # 3. 실행 컨텍스트 준비
            execution_context = self._prepare_execution_context(
                input_dataframe, parameters, business_context
            )
            
            # 4. 안전한 환경에서 코드 실행
            result = self._execute_in_sandbox(processed_code, execution_context, result)
            
            # 5. 실행 시간 기록
            result.execution_time = time.time() - start_time
            
            # 6. 실행 히스토리에 추가
            self.execution_history.append(result)
            
            self.logger.info(f"코드 실행 완료: {result.execution_id} ({result.execution_time:.2f}초)")
            
        except Exception as e:
            result.success = False
            result.error_message = f"예상치 못한 오류: {str(e)}"
            result.execution_time = time.time() - start_time
            self.logger.error(f"코드 실행 중 오류: {e}")
        
        finally:
            # 임시 파일 정리
            self.resource_manager.cleanup_temp_workspaces()
        
        return result
    
    def _preprocess_code(self, code: str, business_context: Dict[str, Any] = None) -> str:
        """코드 전처리"""
        processed_code = code
        
        # 1. 플롯 설정 추가
        if self.enable_plots:
            plot_setup = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
"""
            processed_code = plot_setup + "\n" + processed_code
        
        # 2. 경고 무시 설정
        warning_setup = """
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
"""
        processed_code = warning_setup + "\n" + processed_code
        
        # 3. 비즈니스 컨텍스트 주석 추가
        if business_context:
            context_comment = f"""
# 비즈니스 컨텍스트: {business_context.get('domain', 'Unknown')}
# 분석 목적: {business_context.get('purpose', 'Unknown')}
"""
            processed_code = context_comment + "\n" + processed_code
        
        return processed_code
    
    def _prepare_execution_context(self, dataframe: pd.DataFrame = None,
                                 parameters: Dict[str, Any] = None,
                                 business_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """실행 컨텍스트 준비"""
        
        # 안전한 기본 라이브러리들
        context = {
            # 데이터 처리
            'pd': pd,
            'np': np,
            'df': dataframe.copy() if dataframe is not None else pd.DataFrame(),
            
            # 통계 및 분석
            'stats': stats,
            'sm': sm,
            
            # 시각화
            'plt': plt,
            'sns': sns,
            
            # 기본 Python
            'len': len,
            'list': list,
            'dict': dict,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'print': print,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'sum': sum,
            'max': max,
            'min': min,
            'round': round,
            'abs': abs,
            
            # 수학
            'math': __import__('math'),
            'random': __import__('random'),
            
            # 유틸리티
            'warnings': warnings,
        }
        
        # 사용자 매개변수 추가 (안전성 검증 후)
        if parameters:
            for key, value in parameters.items():
                if self._is_safe_parameter(key, value):
                    context[key] = value
                else:
                    self.logger.warning(f"안전하지 않은 매개변수 제외: {key}")
        
        # 비즈니스 컨텍스트 변수 추가
        if business_context:
            context['_business_domain'] = business_context.get('domain', 'Unknown')
            context['_analysis_purpose'] = business_context.get('purpose', 'Unknown')
        
        return context
    
    def _is_safe_parameter(self, key: str, value: Any) -> bool:
        """매개변수가 안전한지 확인"""
        # 키 이름 검증
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
            return False
        
        # 예약어 검증
        if key in ['exec', 'eval', 'compile', '__import__', 'open', 'file']:
            return False
        
        # 값 타입 검증
        safe_types = (int, float, str, bool, list, tuple, dict, type(None))
        if not isinstance(value, safe_types):
            return False
        
        # 문자열 내용 검증
        if isinstance(value, str):
            for pattern in self.security_validator.forbidden_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    return False
        
        return True
    
    def _execute_in_sandbox(self, code: str, context: Dict[str, Any], 
                          result: ExecutionResult) -> ExecutionResult:
        """샌드박스에서 코드 실행"""
        
        # 임시 작업 디렉토리 생성
        temp_workspace = self.resource_manager.create_temp_workspace()
        
        # stdout/stderr 캡처 준비
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # 현재 디렉토리를 임시 작업 공간으로 변경
            original_cwd = os.getcwd()
            os.chdir(temp_workspace)
            
            # 플롯 저장 경로 설정
            if self.enable_plots:
                context['_plot_save_dir'] = str(temp_workspace)
                context['_plot_counter'] = [0]  # mutable counter
            
            # stdout/stderr 리다이렉션
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):
                
                # 코드 실행 (제한된 컨텍스트에서)
                restricted_globals = {
                    '__builtins__': {
                        name: getattr(__builtins__, name) 
                        for name in self.security_validator.allowed_builtins
                        if hasattr(__builtins__, name)
                    }
                }
                restricted_globals.update(context)
                
                # 실제 코드 실행
                exec(code, restricted_globals, context)
            
            # 실행 성공
            result.success = True
            result.stdout = stdout_capture.getvalue()
            result.stderr = stderr_capture.getvalue()
            
            # 결과 변수 추출
            result.result_variables = self._extract_result_variables(context)
            
            # 생성된 플롯 수집
            if self.enable_plots:
                result.generated_plots = self._collect_generated_plots(temp_workspace)
            
        except MemoryError:
            result.success = False
            result.error_message = "메모리 사용량 초과"
            
        except TimeoutError:
            result.success = False
            result.error_message = "실행 시간 초과"
            
        except SyntaxError as e:
            result.success = False
            result.error_message = f"구문 오류: {e}"
            
        except Exception as e:
            result.success = False
            result.error_message = f"실행 오류: {e}"
            result.stderr = stderr_capture.getvalue()
            
        finally:
            # 원래 디렉토리로 복원
            try:
                os.chdir(original_cwd)
            except:
                pass
        
        return result
    
    def _extract_result_variables(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """결과 변수 추출"""
        result_vars = {}
        
        # 제외할 키들
        exclude_keys = {
            'pd', 'np', 'plt', 'sns', 'stats', 'sm', 'math', 'random', 'warnings',
            '__builtins__', '_plot_save_dir', '_plot_counter', '_business_domain', '_analysis_purpose'
        }.union(self.security_validator.allowed_builtins)
        
        for key, value in context.items():
            if key.startswith('_') or key in exclude_keys:
                continue
            
            try:
                # 직렬화 가능한 값만 포함
                serializable_value = self._make_serializable(value)
                if serializable_value is not None:
                    result_vars[key] = serializable_value
            except Exception as e:
                self.logger.debug(f"변수 {key} 직렬화 실패: {e}")
        
        return result_vars
    
    def _make_serializable(self, obj: Any) -> Any:
        """객체를 JSON 직렬화 가능한 형태로 변환"""
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict() if hasattr(obj, 'to_dict') else str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def _collect_generated_plots(self, workspace_dir: Path) -> List[str]:
        """생성된 플롯 파일 수집"""
        plot_files = []
        
        try:
            # matplotlib 플롯 자동 저장
            fig_nums = plt.get_fignums()
            for i, fig_num in enumerate(fig_nums[:self.max_plots]):
                fig = plt.figure(fig_num)
                
                plot_filename = f"plot_{i+1}.png"
                plot_path = workspace_dir / plot_filename
                
                fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                
                # Base64 인코딩
                with open(plot_path, 'rb') as f:
                    import base64
                    plot_data = base64.b64encode(f.read()).decode('utf-8')
                    plot_files.append(plot_data)
                
                plt.close(fig)
            
        except Exception as e:
            self.logger.error(f"플롯 수집 중 오류: {e}")
        
        return plot_files
    
    def execute_with_multiprocessing(self, code: str, 
                                   input_dataframe: pd.DataFrame = None,
                                   parameters: Dict[str, Any] = None,
                                   business_context: Dict[str, Any] = None) -> ExecutionResult:
        """
        멀티프로세싱을 사용한 더 안전한 코드 실행
        별도 프로세스에서 실행하여 메인 프로세스 보호
        """
        
        def _execute_in_process(code, df_data, params, context):
            """별도 프로세스에서 실행될 함수"""
            try:
                # 리소스 제한 설정
                resource_manager = ResourceManager(self.max_memory_mb, self.timeout_seconds)
                resource_manager.set_resource_limits()
                
                # DataFrame 복원
                if df_data is not None:
                    dataframe = pd.DataFrame(df_data['data'], 
                                           columns=df_data['columns'], 
                                           index=df_data['index'])
                else:
                    dataframe = None
                
                # 메인 실행 로직
                executor = SafeCodeExecutor(
                    timeout_seconds=self.timeout_seconds,
                    max_memory_mb=self.max_memory_mb,
                    enable_plots=self.enable_plots,
                    max_plots=self.max_plots
                )
                
                return executor.execute_code(code, dataframe, params, context)
                
            except Exception as e:
                result = ExecutionResult()
                result.success = False
                result.error_message = f"프로세스 실행 오류: {e}"
                return result
        
        try:
            # DataFrame 직렬화
            df_data = None
            if input_dataframe is not None:
                df_data = {
                    'data': input_dataframe.values.tolist(),
                    'columns': input_dataframe.columns.tolist(),
                    'index': input_dataframe.index.tolist()
                }
            
            # 별도 프로세스에서 실행
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_execute_in_process, code, df_data, 
                                       parameters, business_context)
                
                try:
                    result = future.result(timeout=self.timeout_seconds + 10)
                    return result
                
                except FutureTimeoutError:
                    result = ExecutionResult()
                    result.success = False
                    result.error_message = "프로세스 실행 시간 초과"
                    return result
                
        except Exception as e:
            result = ExecutionResult()
            result.success = False
            result.error_message = f"멀티프로세싱 실행 오류: {e}"
            return result
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """실행 통계 조회"""
        if not self.execution_history:
            return {'total_executions': 0}
        
        successful_executions = [r for r in self.execution_history if r.success]
        failed_executions = [r for r in self.execution_history if not r.success]
        
        execution_times = [r.execution_time for r in self.execution_history if r.execution_time > 0]
        
        return {
            'total_executions': len(self.execution_history),
            'successful_executions': len(successful_executions),
            'failed_executions': len(failed_executions),
            'success_rate': len(successful_executions) / len(self.execution_history),
            'average_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
            'total_plots_generated': sum(len(r.generated_plots) for r in self.execution_history),
            'common_error_types': self._get_common_error_types()
        }
    
    def _get_common_error_types(self) -> Dict[str, int]:
        """일반적인 오류 유형 분석"""
        error_types = {}
        
        for result in self.execution_history:
            if not result.success and result.error_message:
                # 오류 메시지에서 오류 유형 추출
                if "보안 검증 실패" in result.error_message:
                    error_types['security_violation'] = error_types.get('security_violation', 0) + 1
                elif "메모리 사용량 초과" in result.error_message:
                    error_types['memory_limit'] = error_types.get('memory_limit', 0) + 1
                elif "실행 시간 초과" in result.error_message:
                    error_types['timeout'] = error_types.get('timeout', 0) + 1
                elif "구문 오류" in result.error_message:
                    error_types['syntax_error'] = error_types.get('syntax_error', 0) + 1
                else:
                    error_types['runtime_error'] = error_types.get('runtime_error', 0) + 1
        
        return error_types
    
    def clear_execution_history(self):
        """실행 히스토리 정리"""
        self.execution_history.clear()
        self.logger.info("실행 히스토리가 정리되었습니다.")
    
    def save_plots_to_files(self, plots_base64: List[str], 
                          output_dir: str = "output_results/plots") -> List[str]:
        """Base64 인코딩된 플롯을 파일로 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for i, plot_data in enumerate(plots_base64):
            try:
                import base64
                plot_bytes = base64.b64decode(plot_data)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"plot_{timestamp}_{i+1}.png"
                file_path = output_path / filename
                
                with open(file_path, 'wb') as f:
                    f.write(plot_bytes)
                
                saved_files.append(str(file_path))
                
            except Exception as e:
                self.logger.error(f"플롯 저장 실패 (plot {i+1}): {e}")
        
        return saved_files 