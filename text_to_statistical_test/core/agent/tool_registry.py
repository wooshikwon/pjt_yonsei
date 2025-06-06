"""
Tool Registry

도구 레지스트리 (Agent가 사용할 수 있는 도구들 관리)
- 통계 분석 도구 등록 및 관리
- 도구 검색 및 실행 인터페이스
- 도구 성능 모니터링 및 최적화
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timedelta
import inspect
import importlib

from .autonomous_agent import ActionType
from services.statistics.stats_executor import StatsExecutor
from services.statistics.descriptive_stats import DescriptiveStats
from services.statistics.data_preprocessor import DataPreprocessor
from services.statistics.inferential_tests.assumption_checks import AssumptionChecks
from services.statistics.inferential_tests.parametric_tests import ParametricTests
from services.statistics.inferential_tests.nonparametric_tests import NonParametricTests
from services.code_executor.safe_code_runner import SafeCodeRunner
from utils.error_handler import handle_error, StatisticsException
from core.rag.rag_manager import RAGManager
from core.reporting.visualization_engine import VisualizationEngine
from utils.data_loader import DataLoader
from utils.data_utils import get_available_data_files, validate_file_access


class ToolCategory(Enum):
    """도구 카테고리"""
    STATISTICS = "statistics"
    DATA_PROCESSING = "data_processing"
    DATA_LOADING = "data_loading"  # 새로운 카테고리 추가
    VISUALIZATION = "visualization"
    CODE_EXECUTION = "code_execution"
    ANALYSIS = "analysis"
    REPORTING = "reporting"
    UTILITY = "utility"


class ToolStatus(Enum):
    """도구 상태"""
    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"
    MAINTENANCE = "maintenance"


@dataclass
class ToolMetrics:
    """도구 메트릭"""
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    last_used: Optional[datetime] = None
    error_rate: float = 0.0


@dataclass
class ToolInfo:
    """도구 정보"""
    name: str
    category: ToolCategory
    description: str
    version: str
    capabilities: List[str]
    requirements: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: ToolStatus = ToolStatus.AVAILABLE
    metrics: ToolMetrics = field(default_factory=ToolMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTool:
    """기본 도구 클래스"""
    
    def __init__(self, name: str, category: ToolCategory, description: str):
        self.name = name
        self.category = category
        self.description = description
        self.version = "1.0.0"
        self.capabilities = []
        self.requirements = []
        self.status = ToolStatus.AVAILABLE
        self.metrics = ToolMetrics()
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """도구 실행"""
        raise NotImplementedError("Subclasses must implement execute method")
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """파라미터 검증"""
        return True
    
    async def get_capabilities(self) -> List[str]:
        """도구 기능 반환"""
        return self.capabilities
    
    async def get_status(self) -> ToolStatus:
        """도구 상태 반환"""
        return self.status
    
    def update_metrics(self, execution_time: float, success: bool):
        """메트릭 업데이트"""
        self.metrics.usage_count += 1
        self.metrics.total_execution_time += execution_time
        self.metrics.average_execution_time = self.metrics.total_execution_time / self.metrics.usage_count
        self.metrics.last_used = datetime.now()
        
        if success:
            self.metrics.success_count += 1
        else:
            self.metrics.failure_count += 1
        
        self.metrics.error_rate = self.metrics.failure_count / self.metrics.usage_count


class StatisticalAnalysisTool(BaseTool):
    """통계 분석 도구"""
    
    def __init__(self):
        super().__init__(
            name="statistical_analysis",
            category=ToolCategory.STATISTICS,
            description="통계 분석 수행 도구"
        )
        self.capabilities = [
            "descriptive_statistics", "inferential_tests", "assumption_checks",
            "parametric_tests", "nonparametric_tests"
        ]
        
        # 통계 분석 서비스들
        self.descriptive_stats = DescriptiveStats()
        self.data_preprocessor = DataPreprocessor()
        self.assumption_checker = AssumptionChecks()
        self.parametric_tests = ParametricTests()
        self.nonparametric_tests = NonParametricTests()
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """통계 분석 실행"""
        start_time = datetime.now()
        
        try:
            analysis_type = parameters.get('type', 'descriptive')
            data = parameters.get('data')
            
            if data is None:
                return {'success': False, 'error': 'No data provided'}
            
            result = {}
            
            if analysis_type == 'descriptive':
                result = await self._perform_descriptive_analysis(data, parameters)
            elif analysis_type == 'assumption_check':
                result = await self._perform_assumption_checks(data, parameters)
            elif analysis_type == 't_test':
                result = await self._perform_t_test(data, parameters)
            elif analysis_type == 'anova':
                result = await self._perform_anova(data, parameters)
            elif analysis_type == 'nonparametric_test':
                result = await self._perform_nonparametric_test(data, parameters)
            else:
                result = {'success': False, 'error': f'Unknown analysis type: {analysis_type}'}
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.update_metrics(execution_time, result.get('success', False))
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.update_metrics(execution_time, False)
            self.logger.error(f"통계 분석 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _perform_descriptive_analysis(self, data, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """기술통계 분석 수행"""
        try:
            columns = parameters.get('columns')
            result = self.descriptive_stats.calculate_descriptive_stats(data, columns)
            return {'success': True, 'result': result}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _perform_assumption_checks(self, data, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """가정 검정 수행"""
        try:
            tests = parameters.get('tests', ['normality'])
            results = {}
            
            for test in tests:
                if test == 'normality':
                    results['normality'] = self.assumption_checker.test_normality(data)
                elif test == 'homoscedasticity':
                    groups = parameters.get('groups')
                    if groups:
                        results['homoscedasticity'] = self.assumption_checker.test_homoscedasticity(groups)
                elif test == 'independence':
                    results['independence'] = self.assumption_checker.test_independence(data)
            
            return {'success': True, 'result': results}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _perform_t_test(self, data, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """t-검정 수행"""
        try:
            test_type = parameters.get('test_type', 'independent')
            alpha = parameters.get('alpha', 0.05)
            
            if test_type == 'one_sample':
                population_mean = parameters.get('population_mean', 0)
                result = self.parametric_tests.one_sample_t_test(data, population_mean, alpha)
            elif test_type == 'independent':
                group1 = parameters.get('group1')
                group2 = parameters.get('group2')
                result = self.parametric_tests.independent_t_test(group1, group2, alpha)
            elif test_type == 'paired':
                before = parameters.get('before')
                after = parameters.get('after')
                result = self.parametric_tests.paired_t_test(before, after, alpha)
            else:
                return {'success': False, 'error': f'Unknown t-test type: {test_type}'}
            
            return {'success': True, 'result': result}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _perform_anova(self, data, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """ANOVA 수행"""
        try:
            anova_type = parameters.get('anova_type', 'one_way')
            alpha = parameters.get('alpha', 0.05)
            
            if anova_type == 'one_way':
                groups = parameters.get('groups')
                result = self.parametric_tests.one_way_anova(groups, alpha)
            elif anova_type == 'two_way':
                data_array = parameters.get('data_array')
                factor1 = parameters.get('factor1')
                factor2 = parameters.get('factor2')
                result = self.parametric_tests.two_way_anova(data_array, factor1, factor2, alpha)
            else:
                return {'success': False, 'error': f'Unknown ANOVA type: {anova_type}'}
            
            return {'success': True, 'result': result}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _perform_nonparametric_test(self, data, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """비모수 검정 수행"""
        try:
            test_type = parameters.get('method', 'mann_whitney')
            alpha = parameters.get('alpha', 0.05)
            
            if test_type == 'mann_whitney':
                group1 = parameters.get('group1')
                group2 = parameters.get('group2')
                result = self.nonparametric_tests.mann_whitney_u_test(group1, group2, alpha)
            elif test_type == 'wilcoxon':
                before = parameters.get('before')
                after = parameters.get('after')
                result = self.nonparametric_tests.wilcoxon_signed_rank_test(before, after, alpha)
            elif test_type == 'kruskal_wallis':
                groups = parameters.get('groups')
                result = self.nonparametric_tests.kruskal_wallis_test(groups, alpha)
            else:
                return {'success': False, 'error': f'Unknown nonparametric test: {test_type}'}
            
            return {'success': True, 'result': result}
        except Exception as e:
            return {'success': False, 'error': str(e)}


class DataProcessingTool(BaseTool):
    """데이터 처리 도구"""
    
    def __init__(self):
        super().__init__(
            name="data_processing",
            category=ToolCategory.DATA_PROCESSING,
            description="데이터 전처리 및 변환 도구"
        )
        self.capabilities = [
            "data_cleaning", "missing_value_handling", "outlier_detection",
            "data_transformation", "feature_engineering"
        ]
        self.data_preprocessor = DataPreprocessor()
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 처리 실행"""
        start_time = datetime.now()
        
        try:
            operation = parameters.get('operation', 'clean')
            data = parameters.get('data')
            
            if data is None:
                return {'success': False, 'error': 'No data provided'}
            
            result = {}
            
            if operation == 'clean':
                result = await self._clean_data(data, parameters)
            elif operation == 'handle_missing':
                result = await self._handle_missing_values(data, parameters)
            elif operation == 'detect_outliers':
                result = await self._detect_outliers(data, parameters)
            elif operation == 'transform':
                result = await self._transform_data(data, parameters)
            else:
                result = {'success': False, 'error': f'Unknown operation: {operation}'}
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.update_metrics(execution_time, result.get('success', False))
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.update_metrics(execution_time, False)
            self.logger.error(f"데이터 처리 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _clean_data(self, data, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 정리"""
        try:
            cleaned_data = self.data_preprocessor.clean_data(data)
            return {'success': True, 'result': cleaned_data}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _handle_missing_values(self, data, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """결측값 처리"""
        try:
            strategy = parameters.get('strategy', 'mean')
            processed_data = self.data_preprocessor.handle_missing_values(data, strategy)
            return {'success': True, 'result': processed_data}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _detect_outliers(self, data, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """이상값 탐지"""
        try:
            method = parameters.get('method', 'iqr')
            outliers = self.data_preprocessor.detect_outliers(data, method)
            return {'success': True, 'result': outliers}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _transform_data(self, data, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 변환"""
        try:
            transformation = parameters.get('transformation', 'standardize')
            transformed_data = self.data_preprocessor.transform_data(data, transformation)
            return {'success': True, 'result': transformed_data}
        except Exception as e:
            return {'success': False, 'error': str(e)}


class CodeExecutionTool(BaseTool):
    """코드 실행 도구"""
    
    def __init__(self):
        super().__init__(
            name="code_execution",
            category=ToolCategory.CODE_EXECUTION,
            description="안전한 코드 실행 도구"
        )
        self.capabilities = [
            "python_execution", "statistical_code", "data_analysis_code",
            "visualization_code"
        ]
        self.code_runner = SafeCodeRunner()
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """코드 실행"""
        start_time = datetime.now()
        
        try:
            code = parameters.get('code')
            context = parameters.get('context', {})
            
            if not code:
                return {'success': False, 'error': 'No code provided'}
            
            result = self.code_runner.execute_code(code, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.update_metrics(execution_time, result.success)
            
            return {
                'success': result.success,
                'result': result.output if result.success else None,
                'error': result.error_message if not result.success else None,
                'execution_time': result.execution_time
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.update_metrics(execution_time, False)
            self.logger.error(f"코드 실행 오류: {e}")
            return {'success': False, 'error': str(e)}


class VisualizationTool(BaseTool):
    """시각화 도구"""
    
    def __init__(self):
        super().__init__(
            name="visualization",
            category=ToolCategory.VISUALIZATION,
            description="데이터 시각화 도구"
        )
        self.capabilities = [
            "statistical_plots", "distribution_plots", "correlation_plots",
            "regression_plots", "interactive_plots"
        ]
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """시각화 실행"""
        start_time = datetime.now()
        
        try:
            plot_type = parameters.get('plot_type', 'histogram')
            data = parameters.get('data')
            
            if data is None:
                return {'success': False, 'error': 'No data provided'}
            
            # 시각화 코드 생성 및 실행
            viz_code = self._generate_visualization_code(plot_type, parameters)
            
            # 코드 실행 도구 사용
            code_tool = CodeExecutionTool()
            result = await code_tool.execute({
                'code': viz_code,
                'context': {'data': data}
            })
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.update_metrics(execution_time, result.get('success', False))
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.update_metrics(execution_time, False)
            self.logger.error(f"시각화 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_visualization_code(self, plot_type: str, parameters: Dict[str, Any]) -> str:
        """시각화 코드 생성"""
        if plot_type == 'histogram':
            return """
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(data=data, kde=True)
plt.title('Histogram with KDE')
plt.show()
"""
        elif plot_type == 'boxplot':
            return """
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(data=data)
plt.title('Box Plot')
plt.show()
"""
        elif plot_type == 'scatter':
            x_col = parameters.get('x_column', 'x')
            y_col = parameters.get('y_column', 'y')
            return f"""
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='{x_col}', y='{y_col}')
plt.title('Scatter Plot')
plt.show()
"""
        else:
            return """
import matplotlib.pyplot as plt
print("기본 플롯 생성")
plt.figure(figsize=(8, 6))
plt.plot(data)
plt.title('Default Plot')
plt.show()
"""


class DataLoadingTool(BaseTool):
    """데이터 로딩 도구"""
    
    def __init__(self):
        super().__init__(
            name="data_loading",
            category=ToolCategory.DATA_LOADING,
            description="다양한 형식의 데이터 파일 로딩 및 메타데이터 생성 도구"
        )
        self.capabilities = [
            "file_loading", "metadata_generation", "data_validation",
            "file_discovery", "cache_management"
        ]
        self.data_loader = DataLoader()
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 로딩 실행"""
        start_time = datetime.now()
        
        try:
            operation = parameters.get('operation', 'load_file')
            
            result = {}
            
            if operation == 'load_file':
                result = await self._load_file(parameters)
            elif operation == 'discover_files':
                result = await self._discover_files(parameters)
            elif operation == 'validate_file':
                result = await self._validate_file(parameters)
            elif operation == 'get_cached_data':
                result = await self._get_cached_data(parameters)
            elif operation == 'clear_cache':
                result = await self._clear_cache(parameters)
            else:
                result = {'success': False, 'error': f'Unknown operation: {operation}'}
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.update_metrics(execution_time, result.get('success', False))
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.update_metrics(execution_time, False)
            self.logger.error(f"데이터 로딩 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _load_file(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """파일 로딩"""
        try:
            file_path = parameters.get('file_path')
            if not file_path:
                return {'success': False, 'error': 'file_path parameter required'}
            
            # 비동기 처리를 위해 별도 스레드에서 실행
            import asyncio
            loop = asyncio.get_event_loop()
            
            df, metadata = await loop.run_in_executor(
                None, 
                self.data_loader.load_file, 
                file_path
            )
            
            if "error" in metadata:
                return {'success': False, 'error': metadata['error']}
            
            return {
                'success': True,
                'data': df,
                'metadata': metadata,
                'file_path': file_path
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _discover_files(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """파일 발견"""
        try:
            data_dir = parameters.get('data_dir', 'input_data/data_files')
            
            files = get_available_data_files(data_dir)
            
            return {
                'success': True,
                'files': files,
                'count': len(files)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _validate_file(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """파일 검증"""
        try:
            file_path = parameters.get('file_path')
            if not file_path:
                return {'success': False, 'error': 'file_path parameter required'}
            
            validation = validate_file_access(file_path)
            
            return {
                'success': True,
                'validation': validation
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _get_cached_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """캐시된 데이터 조회"""
        try:
            file_path = parameters.get('file_path')
            if not file_path:
                return {'success': False, 'error': 'file_path parameter required'}
            
            cached_data = self.data_loader.get_cached_data(file_path)
            
            return {
                'success': True,
                'has_cache': cached_data is not None,
                'data': cached_data[0] if cached_data else None,
                'metadata': cached_data[1] if cached_data else None
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _clear_cache(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """캐시 정리"""
        try:
            self.data_loader.clear_cache()
            
            return {
                'success': True,
                'message': 'Cache cleared successfully'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


class ToolRegistry:
    """도구 레지스트리 (Agent가 사용할 수 있는 도구들 관리)"""
    
    def __init__(self):
        """ToolRegistry 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 도구 저장소
        self.tools: Dict[str, BaseTool] = {}
        self.tool_info: Dict[str, ToolInfo] = {}
        
        # 카테고리별 도구 인덱스
        self.category_index: Dict[ToolCategory, List[str]] = {}
        
        # 액션 타입별 도구 매핑
        self.action_tool_mapping: Dict[ActionType, List[str]] = {}
        
        # 설정
        self.max_concurrent_tools = 10
        self.tool_timeout = 300  # 5분
        
        # 실행 중인 도구 추적
        self.running_tools: Dict[str, asyncio.Task] = {}
        
        # 기본 도구들 등록
        self._register_default_tools()
        
        self.logger.info("ToolRegistry 초기화 완료")
    
    async def initialize(self):
        """도구 레지스트리 초기화"""
        self.logger.info("도구 레지스트리 초기화 중...")
        
        try:
            # 모든 도구 상태 확인
            await self._check_all_tools_status()
            
            # 액션 매핑 설정
            self._setup_action_mappings()
            
            self.logger.info("도구 레지스트리 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"도구 레지스트리 초기화 오류: {e}")
            raise
    
    def register_tool(self, tool: BaseTool, info: ToolInfo = None):
        """도구 등록"""
        tool_name = tool.name
        
        # 도구 등록
        self.tools[tool_name] = tool
        
        # 도구 정보 등록
        if info is None:
            info = ToolInfo(
                name=tool_name,
                category=tool.category,
                description=tool.description,
                version=tool.version,
                capabilities=tool.capabilities,
                requirements=tool.requirements
            )
        
        self.tool_info[tool_name] = info
        
        # 카테고리 인덱스 업데이트
        category = tool.category
        if category not in self.category_index:
            self.category_index[category] = []
        self.category_index[category].append(tool_name)
        
        self.logger.info(f"도구 등록 완료: {tool_name}")
    
    def unregister_tool(self, tool_name: str):
        """도구 등록 해제"""
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            category = tool.category
            
            # 도구 제거
            del self.tools[tool_name]
            del self.tool_info[tool_name]
            
            # 카테고리 인덱스에서 제거
            if category in self.category_index:
                self.category_index[category].remove(tool_name)
            
            self.logger.info(f"도구 등록 해제 완료: {tool_name}")
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """도구 실행"""
        if tool_name not in self.tools:
            return {'success': False, 'error': f'Tool not found: {tool_name}'}
        
        tool = self.tools[tool_name]
        
        # 도구 상태 확인
        if tool.status != ToolStatus.AVAILABLE:
            return {'success': False, 'error': f'Tool not available: {tool_name}'}
        
        # 동시 실행 제한 확인
        if len(self.running_tools) >= self.max_concurrent_tools:
            return {'success': False, 'error': 'Maximum concurrent tools limit reached'}
        
        try:
            # 도구 상태 변경
            tool.status = ToolStatus.BUSY
            
            # 파라미터 검증
            if not await tool.validate_parameters(parameters):
                tool.status = ToolStatus.AVAILABLE
                return {'success': False, 'error': 'Invalid parameters'}
            
            # 비동기 실행
            task = asyncio.create_task(
                asyncio.wait_for(tool.execute(parameters), timeout=self.tool_timeout)
            )
            
            self.running_tools[tool_name] = task
            
            try:
                result = await task
                tool.status = ToolStatus.AVAILABLE
                return result
            
            except asyncio.TimeoutError:
                tool.status = ToolStatus.ERROR
                return {'success': False, 'error': 'Tool execution timeout'}
            
            finally:
                if tool_name in self.running_tools:
                    del self.running_tools[tool_name]
        
        except Exception as e:
            tool.status = ToolStatus.ERROR
            self.logger.error(f"도구 실행 오류 ({tool_name}): {e}")
            return {'success': False, 'error': str(e)}
    
    def get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """이름으로 도구 검색"""
        return self.tools.get(tool_name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """카테고리별 도구 검색"""
        tool_names = self.category_index.get(category, [])
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def get_tool_for_action(self, action_type: ActionType) -> Optional[BaseTool]:
        """액션 타입에 맞는 도구 검색"""
        tool_names = self.action_tool_mapping.get(action_type, [])
        
        for tool_name in tool_names:
            tool = self.tools.get(tool_name)
            if tool and tool.status == ToolStatus.AVAILABLE:
                return tool
        
        return None
    
    def get_tools_by_capability(self, capability: str) -> List[BaseTool]:
        """기능별 도구 검색"""
        matching_tools = []
        
        for tool in self.tools.values():
            if capability in tool.capabilities:
                matching_tools.append(tool)
        
        return matching_tools
    
    def get_tool_info(self, tool_name: str) -> Optional[ToolInfo]:
        """도구 정보 반환"""
        return self.tool_info.get(tool_name)
    
    def get_all_tools_info(self) -> Dict[str, ToolInfo]:
        """모든 도구 정보 반환"""
        return self.tool_info.copy()
    
    def get_tool_metrics(self, tool_name: str) -> Optional[ToolMetrics]:
        """도구 메트릭 반환"""
        tool = self.tools.get(tool_name)
        return tool.metrics if tool else None
    
    async def get_tool_status(self, tool_name: str) -> Optional[ToolStatus]:
        """도구 상태 반환"""
        tool = self.tools.get(tool_name)
        return await tool.get_status() if tool else None
    
    def list_available_tools(self) -> List[str]:
        """사용 가능한 도구 목록"""
        return [
            name for name, tool in self.tools.items()
            if tool.status == ToolStatus.AVAILABLE
        ]
    
    def list_tools_by_category(self) -> Dict[str, List[str]]:
        """카테고리별 도구 목록"""
        result = {}
        for category, tool_names in self.category_index.items():
            result[category.value] = tool_names
        return result
    
    async def health_check(self) -> Dict[str, Any]:
        """도구 상태 점검"""
        health_status = {
            'total_tools': len(self.tools),
            'available_tools': 0,
            'busy_tools': 0,
            'error_tools': 0,
            'disabled_tools': 0,
            'tool_details': {}
        }
        
        for tool_name, tool in self.tools.items():
            status = await tool.get_status()
            
            if status == ToolStatus.AVAILABLE:
                health_status['available_tools'] += 1
            elif status == ToolStatus.BUSY:
                health_status['busy_tools'] += 1
            elif status == ToolStatus.ERROR:
                health_status['error_tools'] += 1
            elif status == ToolStatus.DISABLED:
                health_status['disabled_tools'] += 1
            
            health_status['tool_details'][tool_name] = {
                'status': status.value,
                'metrics': {
                    'usage_count': tool.metrics.usage_count,
                    'success_rate': (tool.metrics.success_count / tool.metrics.usage_count) 
                                  if tool.metrics.usage_count > 0 else 0,
                    'average_execution_time': tool.metrics.average_execution_time,
                    'last_used': tool.metrics.last_used.isoformat() if tool.metrics.last_used else None
                }
            }
        
        return health_status
    
    def _register_default_tools(self):
        """기본 도구들 등록"""
        # 데이터 로딩 도구 (새로 추가)
        data_loading_tool = DataLoadingTool()
        self.register_tool(data_loading_tool)
        
        # 통계 분석 도구
        stats_tool = StatisticalAnalysisTool()
        self.register_tool(stats_tool)
        
        # 데이터 처리 도구
        data_tool = DataProcessingTool()
        self.register_tool(data_tool)
        
        # 코드 실행 도구
        code_tool = CodeExecutionTool()
        self.register_tool(code_tool)
        
        # 시각화 도구
        viz_tool = VisualizationTool()
        self.register_tool(viz_tool)
    
    def _setup_action_mappings(self):
        """액션 타입별 도구 매핑 설정"""
        self.action_tool_mapping = {
            ActionType.DATA_ANALYSIS: ['statistical_analysis', 'data_processing'],
            ActionType.STATISTICAL_TEST: ['statistical_analysis'],
            ActionType.DATA_PREPROCESSING: ['data_processing'],
            ActionType.VISUALIZATION: ['visualization'],
            ActionType.CODE_EXECUTION: ['code_execution'],
            ActionType.ASSUMPTION_CHECK: ['statistical_analysis'],
            ActionType.REPORT_GENERATION: ['statistical_analysis', 'visualization'],
            # 새로운 액션 타입 추가 (필요시)
            # ActionType.DATA_LOADING: ['data_loading']
        }
    
    async def _check_all_tools_status(self):
        """모든 도구 상태 확인"""
        for tool_name, tool in self.tools.items():
            try:
                status = await tool.get_status()
                self.tool_info[tool_name].status = status
            except Exception as e:
                self.logger.warning(f"도구 상태 확인 실패 ({tool_name}): {e}")
                tool.status = ToolStatus.ERROR
                self.tool_info[tool_name].status = ToolStatus.ERROR
    
    # 고급 기능들
    async def optimize_tool_performance(self):
        """도구 성능 최적화"""
        self.logger.info("도구 성능 최적화 시작")
        
        for tool_name, tool in self.tools.items():
            metrics = tool.metrics
            
            # 성능이 낮은 도구 식별
            if metrics.error_rate > 0.5 and metrics.usage_count > 10:
                self.logger.warning(f"도구 성능 저하 감지: {tool_name} (오류율: {metrics.error_rate:.2%})")
                
                # 임시 비활성화
                tool.status = ToolStatus.MAINTENANCE
                
                # 성능 개선 시도 (실제 구현에서는 더 복잡한 로직)
                await asyncio.sleep(1)  # 시뮬레이션
                
                # 다시 활성화
                tool.status = ToolStatus.AVAILABLE
    
    async def auto_scale_tools(self):
        """도구 자동 스케일링"""
        # 사용량이 많은 도구의 인스턴스 추가 생성 등
        # 실제 구현에서는 더 복잡한 로직 필요
        pass
    
    def get_tool_recommendations(self, context: Dict[str, Any]) -> List[str]:
        """컨텍스트 기반 도구 추천"""
        recommendations = []
        
        # 데이터 타입 기반 추천
        if 'data' in context:
            recommendations.extend(['statistical_analysis', 'data_processing'])
        
        # 분석 목표 기반 추천
        analysis_goal = context.get('analysis_goal', '')
        if 'visualization' in analysis_goal.lower():
            recommendations.append('visualization')
        if 'code' in analysis_goal.lower():
            recommendations.append('code_execution')
        
        # 중복 제거 및 사용 가능한 도구만 반환
        recommendations = list(set(recommendations))
        return [tool for tool in recommendations if tool in self.tools and 
                self.tools[tool].status == ToolStatus.AVAILABLE]
    
    def export_tool_registry(self) -> Dict[str, Any]:
        """도구 레지스트리 내보내기"""
        return {
            'tools': {
                name: {
                    'category': tool.category.value,
                    'description': tool.description,
                    'version': tool.version,
                    'capabilities': tool.capabilities,
                    'status': tool.status.value,
                    'metrics': {
                        'usage_count': tool.metrics.usage_count,
                        'success_count': tool.metrics.success_count,
                        'failure_count': tool.metrics.failure_count,
                        'error_rate': tool.metrics.error_rate,
                        'average_execution_time': tool.metrics.average_execution_time
                    }
                }
                for name, tool in self.tools.items()
            },
            'category_index': {
                category.value: tools for category, tools in self.category_index.items()
            },
            'action_mappings': {
                action.value: tools for action, tools in self.action_tool_mapping.items()
            }
        } 