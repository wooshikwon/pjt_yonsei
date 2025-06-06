"""
통계 분석 실행 엔진
다양한 통계 분석을 실행하고 결과를 관리하는 중앙 실행기
"""

import logging
import time
import traceback
import psutil
import threading
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, Future
import pandas as pd
import numpy as np
from enum import Enum

from utils.error_handler import ErrorHandler
from utils.global_cache import GlobalCache
from services.statistics.data_preprocessor import DataPreprocessor
from services.statistics.descriptive_stats import DescriptiveStats
from services.statistics.inferential_tests.parametric_tests import ParametricTests
from services.statistics.inferential_tests.nonparametric_tests import NonParametricTests
from services.statistics.inferential_tests.regression_tests import RegressionTests
from services.statistics.inferential_tests.assumption_checks import AssumptionChecks

class AnalysisStatus(Enum):
    """분석 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class AnalysisTask:
    """분석 작업 정의"""
    task_id: str
    analysis_type: str
    data: pd.DataFrame
    parameters: Dict[str, Any]
    status: AnalysisStatus = AnalysisStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionMetrics:
    """실행 메트릭"""
    total_analyses: int = 0
    successful_analyses: int = 0
    failed_analyses: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    memory_usage_peak: float = 0.0
    cpu_usage_peak: float = 0.0

class StatsExecutor:
    """통계 분석 실행 엔진"""
    
    def __init__(self, 
                 max_workers: int = 4,
                 max_memory_usage: float = 0.8,
                 timeout_seconds: int = 300):
        """
        Args:
            max_workers: 최대 병렬 작업 수
            max_memory_usage: 최대 메모리 사용률 (0-1)
            timeout_seconds: 작업 타임아웃 시간
        """
        self.logger = logging.getLogger(__name__)
        self.error_handler = ErrorHandler()
        self.cache = GlobalCache()
        
        # 실행 설정
        self.max_workers = max_workers
        self.max_memory_usage = max_memory_usage
        self.timeout_seconds = timeout_seconds
        
        # 서비스 초기화
        self.data_preprocessor = DataPreprocessor()
        self.descriptive_stats = DescriptiveStats()
        self.parametric_tests = ParametricTests()
        self.nonparametric_tests = NonParametricTests()
        self.regression_tests = RegressionTests()
        self.assumption_checks = AssumptionChecks()
        
        # 실행 관리
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running_tasks: Dict[str, AnalysisTask] = {}
        self.completed_tasks: Dict[str, AnalysisTask] = {}
        self.metrics = ExecutionMetrics()
        
        # 리소스 모니터링
        self.resource_monitor_active = False
        self.resource_stats: List[Dict[str, Any]] = []
        
        # 분석 메서드 매핑
        self.analysis_methods = {
            'descriptive_stats': self._execute_descriptive_stats,
            'ttest_independent': self._execute_ttest_independent,
            'ttest_paired': self._execute_ttest_paired,
            'anova_oneway': self._execute_anova_oneway,
            'anova_twoway': self._execute_anova_twoway,
            'mannwhitney': self._execute_mannwhitney,
            'kruskal_wallis': self._execute_kruskal_wallis,
            'wilcoxon': self._execute_wilcoxon,
            'linear_regression': self._execute_linear_regression,
            'multiple_regression': self._execute_multiple_regression,
            'logistic_regression': self._execute_logistic_regression,
            'chi_square': self._execute_chi_square,
            'fisher_exact': self._execute_fisher_exact,
            'correlation_pearson': self._execute_correlation_pearson,
            'correlation_spearman': self._execute_correlation_spearman,
            'normality_test': self._execute_normality_test,
            'homogeneity_test': self._execute_homogeneity_test
        }
        
        self.logger.info("StatsExecutor 초기화 완료")
    
    def execute_analysis(self,
                        analysis_type: str,
                        data: pd.DataFrame,
                        parameters: Optional[Dict[str, Any]] = None,
                        task_id: Optional[str] = None,
                        async_execution: bool = False) -> Union[Dict[str, Any], str]:
        """
        통계 분석 실행
        
        Args:
            analysis_type: 분석 유형
            data: 분석할 데이터
            parameters: 분석 파라미터
            task_id: 작업 ID (미지정시 자동 생성)
            async_execution: 비동기 실행 여부
            
        Returns:
            동기 실행시 분석 결과, 비동기 실행시 작업 ID
        """
        try:
            if task_id is None:
                task_id = f"task_{int(time.time() * 1000)}"
            
            # 리소스 체크
            if not self._check_resource_availability():
                raise RuntimeError("시스템 리소스 부족")
            
            # 분석 방법 검증
            if analysis_type not in self.analysis_methods:
                raise ValueError(f"지원하지 않는 분석 유형: {analysis_type}")
            
            # 작업 생성
            task = AnalysisTask(
                task_id=task_id,
                analysis_type=analysis_type,
                data=data.copy(),
                parameters=parameters or {}
            )
            
            if async_execution:
                # 비동기 실행
                future = self.executor.submit(self._execute_task, task)
                self.running_tasks[task_id] = task
                return task_id
            else:
                # 동기 실행
                return self._execute_task(task)
            
        except Exception as e:
            self.logger.error(f"분석 실행 오류: {e}")
            return self.error_handler.handle_error(e, default_return={})
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """작업 상태 조회"""
        try:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                return {
                    'task_id': task.task_id,
                    'status': task.status.value,
                    'analysis_type': task.analysis_type,
                    'created_at': task.created_at.isoformat(),
                    'started_at': task.started_at.isoformat() if task.started_at else None
                }
            elif task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                return {
                    'task_id': task.task_id,
                    'status': task.status.value,
                    'analysis_type': task.analysis_type,
                    'created_at': task.created_at.isoformat(),
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                    'result': task.result,
                    'error': task.error
                }
            else:
                return {'error': f'작업 ID {task_id}를 찾을 수 없습니다'}
                
        except Exception as e:
            self.logger.error(f"작업 상태 조회 오류: {e}")
            return {'error': str(e)}
    
    def cancel_task(self, task_id: str) -> bool:
        """작업 취소"""
        try:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.status = AnalysisStatus.CANCELLED
                return True
            return False
        except Exception as e:
            self.logger.error(f"작업 취소 오류: {e}")
            return False
    
    def get_available_analyses(self) -> List[Dict[str, Any]]:
        """사용 가능한 분석 목록 반환"""
        analyses = []
        
        for analysis_type in self.analysis_methods.keys():
            info = self._get_analysis_info(analysis_type)
            analyses.append({
                'type': analysis_type,
                'name': info.get('name', analysis_type),
                'description': info.get('description', ''),
                'required_parameters': info.get('required_parameters', []),
                'optional_parameters': info.get('optional_parameters', [])
            })
        
        return analyses
    
    def _execute_task(self, task: AnalysisTask) -> Dict[str, Any]:
        """단일 작업 실행"""
        start_time = time.time()
        
        try:
            # 상태 업데이트
            task.status = AnalysisStatus.RUNNING
            task.started_at = datetime.now()
            
            # 리소스 모니터링 시작
            self._start_resource_monitoring()
            
            # 분석 실행
            analysis_method = self.analysis_methods[task.analysis_type]
            result = analysis_method(task.data, task.parameters)
            
            # 작업 완료
            task.status = AnalysisStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            # 실행 시간 기록
            execution_time = time.time() - start_time
            task.resource_usage['execution_time'] = execution_time
            
            # 메트릭 업데이트
            self._update_metrics(task, execution_time, success=True)
            
            # 완료된 작업으로 이동
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            self.completed_tasks[task.task_id] = task
            
            self.logger.info(f"분석 완료: {task.analysis_type} ({execution_time:.2f}초)")
            return result
            
        except Exception as e:
            # 오류 처리
            task.status = AnalysisStatus.FAILED
            task.completed_at = datetime.now()
            task.error = str(e)
            
            execution_time = time.time() - start_time
            self._update_metrics(task, execution_time, success=False)
            
            # 완료된 작업으로 이동
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            self.completed_tasks[task.task_id] = task
            
            self.logger.error(f"분석 실패: {task.analysis_type} - {e}")
            raise e
        finally:
            self._stop_resource_monitoring()
    
    def _execute_descriptive_stats(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """기술통계 실행"""
        return self.descriptive_stats.calculate_all_statistics(
            data, 
            columns=params.get('columns'),
            include_distribution=params.get('include_distribution', True)
        )
    
    def _execute_ttest_independent(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """독립표본 t-검정 실행"""
        dependent_var = params['dependent_var']
        group_var = params['group_var']
        
        # 그룹별로 데이터 분리
        groups = data.groupby(group_var)[dependent_var]
        group_names = list(groups.groups.keys())
        
        if len(group_names) != 2:
            raise ValueError("독립표본 t-검정은 정확히 2개 그룹이 필요합니다")
        
        group1 = groups.get_group(group_names[0])
        group2 = groups.get_group(group_names[1])
        
        result = self.parametric_tests.independent_t_test(
            group1=group1,
            group2=group2,
            equal_var=params.get('equal_var', True),
            alternative=params.get('alternative', 'two-sided')
        )
        
        # TestResult를 Dict로 변환
        return {
            'test_name': 'Independent samples t-test',
            'test_type': result.test_type.value,
            'statistic': result.statistic,
            'p_value': result.p_value,
            'degrees_of_freedom': result.degrees_of_freedom,
            'effect_size': result.effect_size,
            'confidence_interval': result.confidence_interval,
            'is_significant': result.is_significant,
            'interpretation': result.interpretation,
            'assumptions_met': result.assumptions_met,
            'metadata': result.metadata
        }
    
    def _execute_ttest_paired(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """대응표본 t-검정 실행"""
        var1 = params['var1']
        var2 = params['var2']
        
        result = self.parametric_tests.paired_t_test(
            before=data[var1],
            after=data[var2],
            alternative=params.get('alternative', 'two-sided')
        )
        
        return {
            'test_name': 'Paired samples t-test',
            'test_type': result.test_type.value,
            'statistic': result.statistic,
            'p_value': result.p_value,
            'degrees_of_freedom': result.degrees_of_freedom,
            'effect_size': result.effect_size,
            'confidence_interval': result.confidence_interval,
            'is_significant': result.is_significant,
            'interpretation': result.interpretation,
            'assumptions_met': result.assumptions_met,
            'metadata': result.metadata
        }
    
    def _execute_anova_oneway(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """일원분산분석 실행"""
        dependent_var = params['dependent_var']
        group_var = params['group_var']
        
        # 그룹별로 데이터 분리
        groups = data.groupby(group_var)[dependent_var]
        group_data = [group.dropna() for name, group in groups]
        group_names = [str(name) for name in groups.groups.keys()]
        
        result = self.parametric_tests.one_way_anova(
            *group_data,
            group_names=group_names
        )
        
        response = {
            'test_name': 'One-way ANOVA',
            'test_type': result.test_type.value,
            'statistic': result.statistic,
            'p_value': result.p_value,
            'degrees_of_freedom': result.degrees_of_freedom,
            'effect_size': result.effect_size,
            'is_significant': result.is_significant,
            'interpretation': result.interpretation,
            'assumptions_met': result.assumptions_met,
            'metadata': result.metadata
        }
        
        # 사후 검정 수행
        if params.get('posthoc', True) and result.is_significant:
            try:
                posthoc_result = self.parametric_tests.post_hoc_test(
                    *group_data,
                    group_names=group_names
                )
                response['posthoc'] = {
                    'method': posthoc_result.method.value,
                    'comparisons': posthoc_result.comparisons,
                    'overall_significant': posthoc_result.overall_significant,
                    'metadata': posthoc_result.metadata
                }
            except Exception as e:
                self.logger.warning(f"사후 검정 실패: {e}")
        
        return response
    
    def _execute_anova_twoway(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """이원분산분석 실행"""
        result = self.parametric_tests.two_way_anova(
            data=data,
            dependent_var=params['dependent_var'],
            factor1=params['factor1'],
            factor2=params['factor2']
        )
        
        # 다중 결과를 하나의 Dict로 통합
        response = {
            'test_name': 'Two-way ANOVA',
            'results': {}
        }
        
        for effect_name, test_result in result.items():
            response['results'][effect_name] = {
                'test_type': test_result.test_type.value,
                'statistic': test_result.statistic,
                'p_value': test_result.p_value,
                'degrees_of_freedom': test_result.degrees_of_freedom,
                'effect_size': test_result.effect_size,
                'is_significant': test_result.is_significant,
                'interpretation': test_result.interpretation,
                'metadata': test_result.metadata
            }
        
        return response
    
    def _execute_mannwhitney(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mann-Whitney U 검정 실행"""
        dependent_var = params['dependent_var']
        group_var = params['group_var']
        
        # 그룹별로 데이터 분리
        groups = data.groupby(group_var)[dependent_var]
        group_names = list(groups.groups.keys())
        
        if len(group_names) != 2:
            raise ValueError("Mann-Whitney U 검정은 정확히 2개 그룹이 필요합니다")
        
        group1 = groups.get_group(group_names[0])
        group2 = groups.get_group(group_names[1])
        
        result = self.nonparametric_tests.mann_whitney_u_test(
            group1=group1,
            group2=group2,
            alternative=params.get('alternative', 'two-sided')
        )
        
        return {
            'test_name': 'Mann-Whitney U test',
            'test_type': result.test_type.value,
            'statistic': result.statistic,
            'p_value': result.p_value,
            'effect_size': result.effect_size,
            'is_significant': result.is_significant,
            'interpretation': result.interpretation,
            'metadata': result.metadata
        }
    
    def _execute_kruskal_wallis(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Kruskal-Wallis 검정 실행"""
        dependent_var = params['dependent_var']
        group_var = params['group_var']
        
        # 그룹별로 데이터 분리
        groups = data.groupby(group_var)[dependent_var]
        group_data = [group.dropna() for name, group in groups]
        group_names = [str(name) for name in groups.groups.keys()]
        
        result = self.nonparametric_tests.kruskal_wallis_test(
            *group_data,
            group_names=group_names
        )
        
        return {
            'test_name': 'Kruskal-Wallis test',
            'test_type': result.test_type.value,
            'statistic': result.statistic,
            'p_value': result.p_value,
            'effect_size': result.effect_size,
            'is_significant': result.is_significant,
            'interpretation': result.interpretation,
            'metadata': result.metadata
        }
    
    def _execute_wilcoxon(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Wilcoxon 부호순위 검정 실행"""
        var1 = params['var1']
        var2 = params['var2']
        
        result = self.nonparametric_tests.wilcoxon_signed_rank_test(
            before=data[var1],
            after=data[var2],
            alternative=params.get('alternative', 'two-sided')
        )
        
        return {
            'test_name': 'Wilcoxon signed-rank test',
            'test_type': result.test_type.value,
            'statistic': result.statistic,
            'p_value': result.p_value,
            'effect_size': result.effect_size,
            'is_significant': result.is_significant,
            'interpretation': result.interpretation,
            'metadata': result.metadata
        }
    
    def _execute_linear_regression(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """단순선형회귀 실행"""
        result = self.regression_tests.simple_linear_regression(
            x=data[params['independent_var']],
            y=data[params['dependent_var']],
            x_name=params['independent_var'],
            y_name=params['dependent_var']
        )
        
        return {
            'test_name': 'Simple Linear Regression',
            'regression_type': result.regression_type.value,
            'coefficients': result.coefficients,
            'p_values': result.p_values,
            'confidence_intervals': result.confidence_intervals,
            'r_squared': result.r_squared,
            'adjusted_r_squared': result.adjusted_r_squared,
            'f_statistic': result.f_statistic,
            'f_p_value': result.f_p_value,
            'interpretation': result.interpretation,
            'diagnostics': result.diagnostics,
            'metadata': result.metadata
        }
    
    def _execute_multiple_regression(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """다중선형회귀 실행"""
        result = self.regression_tests.multiple_linear_regression(
            X=data[params['independent_vars']],
            y=data[params['dependent_var']],
            y_name=params['dependent_var']
        )
        
        return {
            'test_name': 'Multiple Linear Regression',
            'regression_type': result.regression_type.value,
            'coefficients': result.coefficients,
            'p_values': result.p_values,
            'confidence_intervals': result.confidence_intervals,
            'r_squared': result.r_squared,
            'adjusted_r_squared': result.adjusted_r_squared,
            'f_statistic': result.f_statistic,
            'f_p_value': result.f_p_value,
            'interpretation': result.interpretation,
            'diagnostics': result.diagnostics,
            'metadata': result.metadata
        }
    
    def _execute_logistic_regression(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """로지스틱 회귀 실행"""
        result = self.regression_tests.logistic_regression(
            X=data[params['independent_vars']],
            y=data[params['dependent_var']]
        )
        
        return {
            'test_name': 'Logistic Regression',
            'regression_type': result.regression_type.value,
            'coefficients': result.coefficients,
            'p_values': result.p_values,
            'confidence_intervals': result.confidence_intervals,
            'interpretation': result.interpretation,
            'diagnostics': result.diagnostics,
            'metadata': result.metadata
        }
    
    def _execute_chi_square(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """카이제곱 검정 실행"""
        # 임시로 기본 구현 (categorical_analysis 모듈 생성 후 수정 예정)
        from scipy.stats import chi2_contingency
        
        contingency_table = pd.crosstab(data[params['var1']], data[params['var2']])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        return {
            'test_name': 'Chi-square test of independence',
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'contingency_table': contingency_table.to_dict(),
            'expected_frequencies': expected.tolist(),
            'significant': p_value < params.get('alpha', 0.05)
        }
    
    def _execute_fisher_exact(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fisher 정확검정 실행"""
        # 임시로 기본 구현 (categorical_analysis 모듈 생성 후 수정 예정)
        from scipy.stats import fisher_exact
        
        contingency_table = pd.crosstab(data[params['var1']], data[params['var2']])
        
        if contingency_table.shape != (2, 2):
            raise ValueError("Fisher's exact test requires a 2x2 contingency table")
        
        odds_ratio, p_value = fisher_exact(contingency_table)
        
        return {
            'test_name': "Fisher's exact test",
            'odds_ratio': float(odds_ratio),
            'p_value': float(p_value),
            'contingency_table': contingency_table.to_dict(),
            'significant': p_value < params.get('alpha', 0.05)
        }
    
    def _execute_correlation_pearson(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """피어슨 상관분석 실행"""
        if 'var2' in params:
            return self.descriptive_stats.correlation_analysis(
                data, params['var1'], params['var2'], method='pearson'
            )
        else:
            return self.descriptive_stats.correlation_matrix(
                data, params.get('variables'), method='pearson'
            )
    
    def _execute_correlation_spearman(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """스피어만 상관분석 실행"""
        if 'var2' in params:
            return self.descriptive_stats.correlation_analysis(
                data, params['var1'], params['var2'], method='spearman'
            )
        else:
            return self.descriptive_stats.correlation_matrix(
                data, params.get('variables'), method='spearman'
            )
    
    def _execute_normality_test(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """정규성 검정 실행"""
        return self.assumption_checks.test_normality(
            data,
            variables=params.get('variables'),
            method=params.get('method', 'shapiro')
        )
    
    def _execute_homogeneity_test(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """등분산성 검정 실행"""
        return self.assumption_checks.test_homogeneity(
            data,
            dependent_var=params['dependent_var'],
            group_var=params['group_var'],
            method=params.get('method', 'levene')
        )
    
    def _check_resource_availability(self) -> bool:
        """시스템 리소스 가용성 확인"""
        try:
            # 메모리 사용률 확인
            memory = psutil.virtual_memory()
            if memory.percent / 100 > self.max_memory_usage:
                return False
            
            # CPU 사용률 확인 (간단한 체크)
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"리소스 확인 오류: {e}")
            return True  # 오류시 실행 허용
    
    def _start_resource_monitoring(self):
        """리소스 모니터링 시작"""
        if not self.resource_monitor_active:
            self.resource_monitor_active = True
            threading.Thread(target=self._monitor_resources, daemon=True).start()
    
    def _stop_resource_monitoring(self):
        """리소스 모니터링 중지"""
        self.resource_monitor_active = False
    
    def _monitor_resources(self):
        """리소스 모니터링 실행"""
        while self.resource_monitor_active:
            try:
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent()
                
                resource_info = {
                    'timestamp': datetime.now().isoformat(),
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'cpu_percent': cpu_percent
                }
                
                self.resource_stats.append(resource_info)
                
                # 최근 100개 기록만 유지
                if len(self.resource_stats) > 100:
                    self.resource_stats = self.resource_stats[-100:]
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"리소스 모니터링 오류: {e}")
                break
    
    def _update_metrics(self, task: AnalysisTask, execution_time: float, success: bool):
        """메트릭 업데이트"""
        self.metrics.total_analyses += 1
        self.metrics.total_execution_time += execution_time
        
        if success:
            self.metrics.successful_analyses += 1
        else:
            self.metrics.failed_analyses += 1
        
        self.metrics.average_execution_time = (
            self.metrics.total_execution_time / self.metrics.total_analyses
        )
        
        # 최대 리소스 사용량 업데이트
        if self.resource_stats:
            recent_stats = self.resource_stats[-10:]  # 최근 10개 기록
            max_memory = max(stat['memory_percent'] for stat in recent_stats)
            max_cpu = max(stat['cpu_percent'] for stat in recent_stats)
            
            self.metrics.memory_usage_peak = max(self.metrics.memory_usage_peak, max_memory)
            self.metrics.cpu_usage_peak = max(self.metrics.cpu_usage_peak, max_cpu)
    
    def _get_analysis_info(self, analysis_type: str) -> Dict[str, Any]:
        """분석 유형별 정보 반환"""
        info_map = {
            'descriptive_stats': {
                'name': '기술통계',
                'description': '데이터의 기본적인 통계량 계산',
                'required_parameters': [],
                'optional_parameters': ['columns', 'include_distribution']
            },
            'ttest_independent': {
                'name': '독립표본 t-검정',
                'description': '두 독립 그룹 간 평균 비교',
                'required_parameters': ['dependent_var', 'group_var'],
                'optional_parameters': ['alpha', 'equal_var']
            },
            'ttest_paired': {
                'name': '대응표본 t-검정',
                'description': '대응하는 두 측정값 간 평균 비교',
                'required_parameters': ['var1', 'var2'],
                'optional_parameters': ['alpha']
            },
            'anova_oneway': {
                'name': '일원분산분석',
                'description': '세 개 이상 그룹 간 평균 비교',
                'required_parameters': ['dependent_var', 'group_var'],
                'optional_parameters': ['alpha', 'posthoc']
            },
            'linear_regression': {
                'name': '단순선형회귀',
                'description': '두 연속변수 간 선형관계 분석',
                'required_parameters': ['dependent_var', 'independent_var'],
                'optional_parameters': ['alpha']
            },
            'chi_square': {
                'name': '카이제곱 검정',
                'description': '두 범주형 변수 간 독립성 검정',
                'required_parameters': ['var1', 'var2'],
                'optional_parameters': ['alpha']
            }
            # 다른 분석들도 필요시 추가
        }
        
        return info_map.get(analysis_type, {})
    
    def get_runtime_statistics(self) -> Dict[str, Any]:
        """실행 통계 반환"""
        return {
            'metrics': {
                'total_analyses': self.metrics.total_analyses,
                'successful_analyses': self.metrics.successful_analyses,
                'failed_analyses': self.metrics.failed_analyses,
                'success_rate': (
                    self.metrics.successful_analyses / max(self.metrics.total_analyses, 1)
                ) * 100,
                'total_execution_time': self.metrics.total_execution_time,
                'average_execution_time': self.metrics.average_execution_time
            },
            'resource_usage': {
                'memory_usage_peak': self.metrics.memory_usage_peak,
                'cpu_usage_peak': self.metrics.cpu_usage_peak
            },
            'active_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks)
        }
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """현재 리소스 사용량 반환"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            
            return {
                'current': {
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_available_gb': memory.available / (1024**3),
                    'cpu_percent': cpu_percent
                },
                'recent_history': self.resource_stats[-20:] if self.resource_stats else []
            }
            
        except Exception as e:
            self.logger.error(f"리소스 사용량 조회 오류: {e}")
            return {'error': str(e)}
    
    def get_error_logs(self, last_n: int = 50) -> List[Dict[str, Any]]:
        """오류 로그 반환"""
        error_tasks = []
        
        for task in self.completed_tasks.values():
            if task.status == AnalysisStatus.FAILED:
                error_tasks.append({
                    'task_id': task.task_id,
                    'analysis_type': task.analysis_type,
                    'error': task.error,
                    'created_at': task.created_at.isoformat(),
                    'failed_at': task.completed_at.isoformat() if task.completed_at else None
                })
        
        # 최신 순으로 정렬
        error_tasks.sort(key=lambda x: x['failed_at'] or x['created_at'], reverse=True)
        
        return error_tasks[:last_n]
    
    def cleanup_completed_tasks(self, older_than_hours: int = 24) -> int:
        """완료된 작업 정리"""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        tasks_to_remove = []
        
        for task_id, task in self.completed_tasks.items():
            if task.completed_at and task.completed_at < cutoff_time:
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.completed_tasks[task_id]
        
        self.logger.info(f"정리된 완료 작업 수: {len(tasks_to_remove)}")
        return len(tasks_to_remove)
    
    def shutdown(self):
        """실행기 종료"""
        self.logger.info("StatsExecutor 종료 중...")
        
        # 리소스 모니터링 중지
        self._stop_resource_monitoring()
        
        # 실행기 종료
        self.executor.shutdown(wait=True)
        
        self.logger.info("StatsExecutor 종료 완료") 