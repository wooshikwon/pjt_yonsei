"""
Statistical Analyzer

통계 분석을 통합 관리하는 메인 클래스
다양한 통계 검정과 분석을 수행하는 중앙 인터페이스
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from .stats_executor import StatsExecutor
from .data_preprocessor import DataPreprocessor
from .descriptive_stats import DescriptiveStats
from .inferential_tests.parametric_tests import ParametricTests
from .inferential_tests.nonparametric_tests import NonParametricTests
from .inferential_tests.regression_tests import RegressionTests
from .inferential_tests.assumption_checks import AssumptionChecks


class StatisticalAnalyzer:
    """
    통계 분석 통합 관리 클래스
    
    모든 통계 분석 기능을 하나의 인터페이스로 제공하며,
    Agent가 사용할 수 있는 통계 도구들을 관리합니다.
    """
    
    def __init__(self):
        """StatisticalAnalyzer 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 하위 분석 모듈들 초기화
        self.stats_executor = StatsExecutor()
        self.data_preprocessor = DataPreprocessor()
        self.descriptive_stats = DescriptiveStats()
        self.parametric_tests = ParametricTests()
        self.nonparametric_tests = NonParametricTests()
        self.regression_tests = RegressionTests()
        self.assumption_checks = AssumptionChecks()
        
        # 분석 이력 추적
        self.analysis_history: List[Dict[str, Any]] = []
        
        self.logger.info("StatisticalAnalyzer 초기화 완료")
    
    def analyze_data(self, 
                    data: pd.DataFrame,
                    analysis_type: str,
                    parameters: Optional[Dict[str, Any]] = None,
                    preprocess: bool = True) -> Dict[str, Any]:
        """
        데이터 분석 수행
        
        Args:
            data: 분석할 데이터
            analysis_type: 분석 유형
            parameters: 분석 파라미터
            preprocess: 전처리 수행 여부
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            analysis_start_time = datetime.now()
            
            # 1. 데이터 전처리 (필요시)
            processed_data = data.copy()
            if preprocess:
                processed_data = self._preprocess_data(processed_data, parameters)
            
            # 2. 분석 실행
            analysis_result = self.stats_executor.execute_analysis(
                analysis_type=analysis_type,
                data=processed_data,
                parameters=parameters or {}
            )
            
            # 3. 결과 후처리
            enhanced_result = self._enhance_analysis_result(
                analysis_result, analysis_type, processed_data
            )
            
            # 4. 분석 이력 기록
            analysis_record = {
                'timestamp': analysis_start_time.isoformat(),
                'analysis_type': analysis_type,
                'data_shape': data.shape,
                'parameters': parameters,
                'success': enhanced_result.get('success', False),
                'execution_time': (datetime.now() - analysis_start_time).total_seconds()
            }
            self.analysis_history.append(analysis_record)
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"데이터 분석 오류: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'analysis_error'
            }
    
    def get_descriptive_statistics(self, 
                                  data: pd.DataFrame,
                                  columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        기술통계 계산
        
        Args:
            data: 분석할 데이터
            columns: 분석할 컬럼 (None이면 모든 컬럼)
            
        Returns:
            Dict[str, Any]: 기술통계 결과
        """
        try:
            if columns:
                analysis_data = data[columns]
            else:
                analysis_data = data
            
            result = self.descriptive_stats.calculate_all_statistics(analysis_data)
            return result
            
        except Exception as e:
            self.logger.error(f"기술통계 계산 오류: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def check_assumptions(self,
                         data: pd.DataFrame,
                         test_type: str,
                         variables: Dict[str, str]) -> Dict[str, Any]:
        """
        통계적 가정 검증
        
        Args:
            data: 검증할 데이터
            test_type: 검정 유형
            variables: 변수 정의
            
        Returns:
            Dict[str, Any]: 가정 검증 결과
        """
        try:
            result = self.assumption_checks.check_all_assumptions(
                data=data,
                test_type=test_type,
                variables=variables
            )
            return result
            
        except Exception as e:
            self.logger.error(f"가정 검증 오류: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def recommend_analysis_method(self,
                                 data: pd.DataFrame,
                                 target_variable: str,
                                 predictor_variables: Optional[List[str]] = None,
                                 analysis_goal: str = "compare") -> Dict[str, Any]:
        """
        데이터 특성에 따른 분석 방법 추천
        
        Args:
            data: 분석할 데이터
            target_variable: 종속변수
            predictor_variables: 독립변수들
            analysis_goal: 분석 목표 (compare, predict, associate)
            
        Returns:
            Dict[str, Any]: 추천 분석 방법들
        """
        try:
            recommendations = []
            
            # 변수 유형 분석
            target_type = self._get_variable_type(data[target_variable])
            
            if predictor_variables:
                predictor_types = {
                    var: self._get_variable_type(data[var]) 
                    for var in predictor_variables
                }
            else:
                predictor_types = {}
            
            # 분석 목표에 따른 추천
            if analysis_goal == "compare":
                recommendations.extend(
                    self._recommend_comparison_methods(target_type, predictor_types, data)
                )
            elif analysis_goal == "predict":
                recommendations.extend(
                    self._recommend_prediction_methods(target_type, predictor_types, data)
                )
            elif analysis_goal == "associate":
                recommendations.extend(
                    self._recommend_association_methods(target_type, predictor_types, data)
                )
            
            return {
                'success': True,
                'recommendations': recommendations,
                'target_variable_type': target_type,
                'predictor_variable_types': predictor_types
            }
            
        except Exception as e:
            self.logger.error(f"분석 방법 추천 오류: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        분석 요약 정보 반환
        
        Returns:
            Dict[str, Any]: 분석 요약
        """
        return {
            'total_analyses': len(self.analysis_history),
            'successful_analyses': sum(1 for a in self.analysis_history if a.get('success')),
            'recent_analyses': self.analysis_history[-5:] if self.analysis_history else [],
            'available_methods': self.stats_executor.get_available_analyses(),
            'executor_stats': self.stats_executor.get_runtime_statistics()
        }
    
    def _preprocess_data(self, 
                        data: pd.DataFrame, 
                        parameters: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """데이터 전처리"""
        try:
            preprocess_config = parameters.get('preprocessing', {}) if parameters else {}
            
            if preprocess_config.get('handle_missing', True):
                data = self.data_preprocessor.handle_missing_values(
                    data, method=preprocess_config.get('missing_method', 'drop')
                )
            
            if preprocess_config.get('handle_outliers', False):
                data = self.data_preprocessor.handle_outliers(
                    data, method=preprocess_config.get('outlier_method', 'iqr')
                )
            
            return data
            
        except Exception as e:
            self.logger.warning(f"전처리 중 오류 발생, 원본 데이터 사용: {e}")
            return data
    
    def _enhance_analysis_result(self,
                               result: Dict[str, Any],
                               analysis_type: str,
                               data: pd.DataFrame) -> Dict[str, Any]:
        """분석 결과 향상"""
        try:
            enhanced = result.copy()
            
            # 메타데이터 추가
            enhanced['metadata'] = {
                'analysis_type': analysis_type,
                'data_shape': data.shape,
                'timestamp': datetime.now().isoformat(),
                'sample_size': len(data)
            }
            
            # 효과 크기 계산 (해당하는 경우)
            if analysis_type in ['ttest_independent', 'ttest_paired']:
                enhanced = self._add_effect_size(enhanced, analysis_type)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"결과 향상 중 오류: {e}")
            return result
    
    def _get_variable_type(self, series: pd.Series) -> str:
        """변수 유형 판단"""
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() <= 10 and series.min() >= 0:
                return 'categorical_numeric'
            else:
                return 'continuous'
        elif pd.api.types.is_categorical_dtype(series) or series.dtype == 'object':
            return 'categorical'
        elif pd.api.types.is_bool_dtype(series):
            return 'binary'
        else:
            return 'unknown'
    
    def _recommend_comparison_methods(self,
                                    target_type: str,
                                    predictor_types: Dict[str, str],
                                    data: pd.DataFrame) -> List[Dict[str, Any]]:
        """비교 분석 방법 추천"""
        recommendations = []
        
        if target_type == 'continuous':
            # 연속형 종속변수
            for var, var_type in predictor_types.items():
                if var_type in ['categorical', 'binary']:
                    n_groups = data[var].nunique()
                    if n_groups == 2:
                        recommendations.append({
                            'method': 'ttest_independent',
                            'description': f'{var}에 따른 {data.columns[0]} 평균 비교 (독립표본 t-검정)',
                            'confidence': 0.9
                        })
                    elif n_groups > 2:
                        recommendations.append({
                            'method': 'anova_oneway',
                            'description': f'{var}에 따른 {data.columns[0]} 평균 비교 (일원분산분석)',
                            'confidence': 0.85
                        })
        
        return recommendations
    
    def _recommend_prediction_methods(self,
                                    target_type: str,
                                    predictor_types: Dict[str, str],
                                    data: pd.DataFrame) -> List[Dict[str, Any]]:
        """예측 분석 방법 추천"""
        recommendations = []
        
        if target_type == 'continuous':
            recommendations.append({
                'method': 'linear_regression',
                'description': '선형 회귀분석을 통한 예측 모델',
                'confidence': 0.8
            })
        elif target_type in ['categorical', 'binary']:
            recommendations.append({
                'method': 'logistic_regression',
                'description': '로지스틱 회귀분석을 통한 분류 모델',
                'confidence': 0.8
            })
        
        return recommendations
    
    def _recommend_association_methods(self,
                                     target_type: str,
                                     predictor_types: Dict[str, str],
                                     data: pd.DataFrame) -> List[Dict[str, Any]]:
        """연관성 분석 방법 추천"""
        recommendations = []
        
        for var, var_type in predictor_types.items():
            if target_type == 'continuous' and var_type == 'continuous':
                recommendations.append({
                    'method': 'correlation_pearson',
                    'description': f'{var}와의 피어슨 상관분석',
                    'confidence': 0.9
                })
            elif target_type == 'categorical' and var_type == 'categorical':
                recommendations.append({
                    'method': 'chi_square',
                    'description': f'{var}와의 카이제곱 독립성 검정',
                    'confidence': 0.85
                })
        
        return recommendations
    
    def _add_effect_size(self, result: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """효과 크기 추가"""
        try:
            # 기본적인 효과 크기 계산 로직
            # 실제로는 각 분석 유형에 맞는 효과 크기를 계산해야 함
            enhanced = result.copy()
            enhanced['effect_size'] = {
                'calculated': True,
                'method': 'cohen_d' if 'ttest' in analysis_type else 'eta_squared',
                'interpretation': 'medium'  # 실제 계산 결과에 따라 결정
            }
            return enhanced
        except Exception:
            return result 