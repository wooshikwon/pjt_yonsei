"""
Data Summary Pipeline

3단계: 데이터 심층 분석 및 요약
선택된 데이터에 대한 기술 통계, 변수 분포, 잠재적 이슈 (결측치, 이상치 등)를 
심층적으로 분석하고 요약하여 사용자에게 제공합니다.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path

from .base_pipeline_step import BasePipelineStep, PipelineStepRegistry
from utils.data_loader import DataLoader
from services.statistics.descriptive_stats import DescriptiveStats
from services.statistics.data_preprocessor import DataPreprocessor


class DataSummaryStep(BasePipelineStep):
    """3단계: 데이터 심층 분석 및 요약"""
    
    def __init__(self):
        """DataSummaryStep 초기화"""
        super().__init__("데이터 심층 분석 및 요약", 3)
        self.data_loader = DataLoader()
        self.stats_calculator = DescriptiveStats()
        self.preprocessor = DataPreprocessor()
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        입력 데이터 유효성 검증
        
        Args:
            input_data: 2단계에서 전달받은 데이터
            
        Returns:
            bool: 유효성 검증 결과
        """
        required_fields = ['selected_file', 'file_info', 'user_request', 'refined_objectives']
        return all(field in input_data for field in required_fields)
    
    def get_expected_output_schema(self) -> Dict[str, Any]:
        """
        예상 출력 스키마 반환
        
        Returns:
            Dict[str, Any]: 출력 데이터 스키마
        """
        return {
            'data_overview': {
                'basic_info': dict,
                'shape': dict,
                'data_types': dict,
                'memory_usage': dict
            },
            'descriptive_statistics': {
                'numerical_summary': dict,
                'categorical_summary': dict,
                'correlation_matrix': dict
            },
            'data_quality_assessment': {
                'missing_values': dict,
                'outliers': dict,
                'duplicates': dict,
                'data_issues': list
            },
            'variable_analysis': {
                'numerical_variables': list,
                'categorical_variables': list,
                'variable_relationships': dict,
                'feature_importance': dict
            },
            'analysis_recommendations': {
                'preprocessing_needed': list,
                'suitable_analyses': list,
                'potential_challenges': list
            },
            'summary_insights': {
                'key_findings': list,
                'data_characteristics': list,
                'analysis_readiness': str
            }
        }
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        데이터 심층 분석 및 요약 파이프라인 실행
        
        Args:
            input_data: 파이프라인 실행 컨텍스트
                - selected_file: 선택된 파일 경로
                - file_info: 파일 기본 정보
                - user_request: 사용자 요청
                - refined_objectives: 분석 목표
                - request_metadata: 요청 메타데이터
            
        Returns:
            Dict: 실행 결과
        """
        self.logger.info("3단계: 데이터 심층 분석 및 요약 시작")
        
        try:
            # 데이터 로딩
            data = self.data_loader.load_data(input_data['selected_file'])
            if data is None:
                return {
                    "success": False,
                    "error": "데이터 로딩 실패",
                    "file_path": input_data['selected_file']
                }
            
            # 1. 데이터 개요 분석
            data_overview = self._analyze_data_overview(data, input_data)
            
            # 2. 기술 통계 계산
            descriptive_stats = self._calculate_descriptive_statistics(data)
            
            # 3. 데이터 품질 평가
            quality_assessment = self._assess_data_quality(data)
            
            # 4. 변수 분석
            variable_analysis = self._analyze_variables(data, input_data)
            
            # 5. 분석 추천사항 생성
            recommendations = self._generate_analysis_recommendations(
                data, input_data, quality_assessment, variable_analysis
            )
            
            # 6. 요약 인사이트 생성
            summary_insights = self._generate_summary_insights(
                data_overview, descriptive_stats, quality_assessment, 
                variable_analysis, recommendations
            )
            
            self.logger.info("데이터 심층 분석 및 요약 완료")
            
            return {
                'data_overview': data_overview,
                'descriptive_statistics': descriptive_stats,
                'data_quality_assessment': quality_assessment,
                'variable_analysis': variable_analysis,
                'analysis_recommendations': recommendations,
                'summary_insights': summary_insights,
                'data_object': data,  # 다음 단계에서 사용할 데이터 객체
                'success_message': f"📊 데이터 심층 분석이 완료되었습니다."
            }
                
        except Exception as e:
            self.logger.error(f"데이터 심층 분석 파이프라인 오류: {e}")
            return {
                'error': True,
                'error_message': str(e),
                'error_type': 'analysis_error'
            }
    
    def _load_and_validate_data(self, file_path: str) -> Any:
        """데이터 로딩 및 기본 검증"""
        try:
            data = self.data_loader.load_data(file_path)
            
            if data.empty:
                return {
                    'error': True,
                    'error_message': '데이터가 비어있습니다.',
                    'error_type': 'empty_data'
                }
            
            return data
            
        except Exception as e:
            self.logger.error(f"데이터 로딩 오류: {e}")
            return {
                'error': True,
                'error_message': f'데이터 로딩 실패: {str(e)}',
                'error_type': 'loading_error'
            }
    
    def _analyze_data_overview(self, data: pd.DataFrame, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 개요 분석"""
        basic_info = {
            'file_name': Path(input_data['selected_file']).name,
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'column_names': list(data.columns),
            'index_type': str(type(data.index).__name__)
        }
        
        shape_info = {
            'dimensions': f"{data.shape[0]} rows × {data.shape[1]} columns",
            'size': data.size,
            'memory_usage_mb': round(data.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
        
        # 데이터 타입 분석
        data_types = self._analyze_data_types(data)
        
        memory_info = {
            'total_memory_mb': round(data.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            'memory_per_column': {
                col: round(data.memory_usage(deep=True)[col] / 1024 / 1024, 2) 
                for col in data.columns
            }
        }
        
        return {
            'basic_info': basic_info,
            'shape': shape_info,
            'data_types': data_types,
            'memory_usage': memory_info
        }
    
    def _analyze_data_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """데이터 타입 분석"""
        type_summary = {
            'numerical': [],
            'categorical': [],
            'datetime': [],
            'boolean': [],
            'object': []
        }
        
        type_counts = {
            'numerical': 0,
            'categorical': 0,
            'datetime': 0,
            'boolean': 0,
            'object': 0
        }
        
        for col in data.columns:
            dtype = data[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                if data[col].nunique() <= 10 and data[col].nunique() < len(data) * 0.05:
                    # 수치형이지만 범주형으로 보이는 경우
                    type_summary['categorical'].append(col)
                    type_counts['categorical'] += 1
                else:
                    type_summary['numerical'].append(col)
                    type_counts['numerical'] += 1
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                type_summary['datetime'].append(col)
                type_counts['datetime'] += 1
            elif pd.api.types.is_bool_dtype(dtype):
                type_summary['boolean'].append(col)
                type_counts['boolean'] += 1
            else:
                # 텍스트나 범주형
                if data[col].nunique() <= 50:  # 범주형으로 간주
                    type_summary['categorical'].append(col)
                    type_counts['categorical'] += 1
                else:
                    type_summary['object'].append(col)
                    type_counts['object'] += 1
        
        return {
            'type_summary': type_summary,
            'type_counts': type_counts,
            'detailed_types': {col: str(data[col].dtype) for col in data.columns}
        }
    
    def _calculate_descriptive_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """기술 통계 계산"""
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 수치형 변수 기술 통계
        numerical_summary = {}
        if numerical_cols:
            numerical_summary = self.stats_calculator.calculate_numerical_stats(data[numerical_cols])
        
        # 범주형 변수 기술 통계
        categorical_summary = {}
        if categorical_cols:
            categorical_summary = self.stats_calculator.calculate_categorical_stats(data[categorical_cols])
        
        # 상관관계 행렬 (수치형 변수들에 대해서만)
        correlation_matrix = {}
        if len(numerical_cols) > 1:
            correlation_matrix = self.stats_calculator.calculate_correlation_matrix(data[numerical_cols])
        
        return {
            'numerical_summary': numerical_summary,
            'categorical_summary': categorical_summary,
            'correlation_matrix': correlation_matrix
        }
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """데이터 품질 평가"""
        # 결측치 분석
        missing_analysis = self._analyze_missing_values(data)
        
        # 이상치 분석 (수치형 변수들에 대해서만)
        outlier_analysis = self._analyze_outliers(data)
        
        # 중복 데이터 분석
        duplicate_analysis = self._analyze_duplicates(data)
        
        # 전반적인 데이터 이슈 식별
        data_issues = self._identify_data_issues(data, missing_analysis, outlier_analysis, duplicate_analysis)
        
        return {
            'missing_values': missing_analysis,
            'outliers': outlier_analysis,
            'duplicates': duplicate_analysis,
            'data_issues': data_issues
        }
    
    def _analyze_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """결측치 분석"""
        missing_count = data.isnull().sum()
        missing_percent = (missing_count / len(data)) * 100
        
        missing_summary = {
            'total_missing': missing_count.sum(),
            'columns_with_missing': missing_count[missing_count > 0].to_dict(),
            'missing_percentages': missing_percent[missing_percent > 0].to_dict(),
            'complete_rows': len(data.dropna()),
            'missing_patterns': self._analyze_missing_patterns(data)
        }
        
        return missing_summary
    
    def _analyze_missing_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """결측치 패턴 분석"""
        # 결측치가 있는 컬럼들만 분석
        missing_cols = data.columns[data.isnull().any()].tolist()
        
        if not missing_cols:
            return {'pattern_analysis': 'No missing values found'}
        
        # 결측치 패턴 조합 분석
        missing_patterns = data[missing_cols].isnull().value_counts().head(10)
        
        return {
            'top_patterns': missing_patterns.to_dict(),
            'pattern_description': 'Most common combinations of missing values across columns'
        }
    
    def _analyze_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """이상치 분석"""
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_cols:
            return {'analysis': 'No numerical columns for outlier analysis'}
        
        outlier_summary = {}
        
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            
            outlier_summary[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': round((len(outliers) / len(data)) * 100, 2),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'extreme_values': {
                    'min': data[col].min(),
                    'max': data[col].max()
                }
            }
        
        return outlier_summary
    
    def _analyze_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """중복 데이터 분석"""
        duplicate_rows = data.duplicated()
        
        return {
            'total_duplicates': duplicate_rows.sum(),
            'duplicate_percentage': round((duplicate_rows.sum() / len(data)) * 100, 2),
            'unique_rows': len(data) - duplicate_rows.sum(),
            'duplicate_subset_analysis': self._analyze_partial_duplicates(data)
        }
    
    def _analyze_partial_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """부분 중복 분석"""
        # 주요 컬럼들에 대한 부분 중복 검사
        important_cols = data.columns[:5].tolist()  # 처음 5개 컬럼만 분석
        
        partial_duplicates = {}
        for col in important_cols:
            duplicate_values = data[col].duplicated()
            partial_duplicates[col] = {
                'duplicate_count': duplicate_values.sum(),
                'unique_values': data[col].nunique(),
                'most_common': data[col].value_counts().head(3).to_dict()
            }
        
        return partial_duplicates
    
    def _identify_data_issues(self, data: pd.DataFrame, missing_analysis: Dict, 
                            outlier_analysis: Dict, duplicate_analysis: Dict) -> List[str]:
        """전반적인 데이터 이슈 식별"""
        issues = []
        
        # 결측치 관련 이슈
        if missing_analysis['total_missing'] > 0:
            high_missing_cols = [
                col for col, pct in missing_analysis['missing_percentages'].items() 
                if pct > 20
            ]
            if high_missing_cols:
                issues.append(f"높은 결측치 비율 컬럼: {', '.join(high_missing_cols)}")
        
        # 이상치 관련 이슈
        if isinstance(outlier_analysis, dict) and outlier_analysis.get('analysis') != 'No numerical columns for outlier analysis':
            high_outlier_cols = [
                col for col, info in outlier_analysis.items() 
                if info.get('outlier_percentage', 0) > 5
            ]
            if high_outlier_cols:
                issues.append(f"이상치가 많은 컬럼: {', '.join(high_outlier_cols)}")
        
        # 중복 데이터 이슈
        if duplicate_analysis['duplicate_percentage'] > 5:
            issues.append(f"중복 데이터 비율이 높음: {duplicate_analysis['duplicate_percentage']}%")
        
        # 데이터 크기 관련 이슈
        if len(data) < 30:
            issues.append("표본 크기가 작음 (통계적 검정에 제한이 있을 수 있음)")
        
        # 변수 수 관련 이슈
        if len(data.columns) > len(data):
            issues.append("변수 수가 관측치 수보다 많음 (차원의 저주 가능성)")
        
        return issues
    
    def _analyze_variables(self, data: pd.DataFrame, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """변수 분석"""
        numerical_vars = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_vars = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 사용자가 언급한 변수들과 매칭
        request_metadata = input_data.get('request_metadata', {})
        target_variables = request_metadata.get('target_variables', [])
        group_variables = request_metadata.get('group_variables', [])
        
        # 변수 간 관계 분석
        relationships = self._analyze_variable_relationships(data, numerical_vars, categorical_vars)
        
        # 특성 중요도 분석 (간단한 버전)
        feature_importance = self._analyze_feature_importance(data, target_variables)
        
        return {
            'numerical_variables': numerical_vars,
            'categorical_variables': categorical_vars,
            'target_variables': target_variables,
            'group_variables': group_variables,
            'variable_relationships': relationships,
            'feature_importance': feature_importance
        }
    
    def _analyze_variable_relationships(self, data: pd.DataFrame, 
                                      numerical_vars: List[str], 
                                      categorical_vars: List[str]) -> Dict[str, Any]:
        """변수 간 관계 분석"""
        relationships = {}
        
        # 수치형 변수 간 상관관계
        if len(numerical_vars) > 1:
            corr_matrix = data[numerical_vars].corr()
            high_correlations = []
            
            for i in range(len(numerical_vars)):
                for j in range(i+1, len(numerical_vars)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # 높은 상관관계
                        high_correlations.append({
                            'var1': numerical_vars[i],
                            'var2': numerical_vars[j],
                            'correlation': round(corr_value, 3)
                        })
            
            relationships['high_correlations'] = high_correlations
        
        # 범주형 변수와 수치형 변수 간 관계 (간단한 분석)
        cat_num_relationships = []
        for cat_var in categorical_vars[:3]:  # 처음 3개만 분석
            for num_var in numerical_vars[:3]:  # 처음 3개만 분석
                try:
                    grouped = data.groupby(cat_var)[num_var].agg(['mean', 'std', 'count'])
                    if len(grouped) > 1:  # 그룹이 여러 개 있는 경우
                        cat_num_relationships.append({
                            'categorical_var': cat_var,
                            'numerical_var': num_var,
                            'group_stats': grouped.to_dict('index')
                        })
                except:
                    continue
        
        relationships['categorical_numerical'] = cat_num_relationships[:5]  # 최대 5개까지
        
        return relationships
    
    def _analyze_feature_importance(self, data: pd.DataFrame, target_variables: List[str]) -> Dict[str, Any]:
        """특성 중요도 분석 (간단한 버전)"""
        if not target_variables:
            return {'analysis': 'No target variables specified'}
        
        importance_analysis = {}
        
        for target_var in target_variables:
            if target_var in data.columns:
                # 간단한 상관관계 기반 중요도
                numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if target_var in numerical_cols and len(numerical_cols) > 1:
                    correlations = data[numerical_cols].corr()[target_var].abs().sort_values(ascending=False)
                    importance_analysis[target_var] = correlations.head(5).to_dict()
        
        return importance_analysis
    
    def _generate_analysis_recommendations(self, data: pd.DataFrame, input_data: Dict[str, Any],
                                         quality_assessment: Dict[str, Any],
                                         variable_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """분석 추천사항 생성"""
        preprocessing_needed = []
        suitable_analyses = []
        potential_challenges = []
        
        # 전처리 추천
        if quality_assessment['missing_values']['total_missing'] > 0:
            preprocessing_needed.append("결측치 처리 (제거 또는 대체)")
        
        if quality_assessment['duplicates']['duplicate_percentage'] > 1:
            preprocessing_needed.append("중복 데이터 제거")
        
        # 이상치가 많은 경우
        outlier_analysis = quality_assessment['outliers']
        if isinstance(outlier_analysis, dict):
            high_outlier_cols = [
                col for col, info in outlier_analysis.items() 
                if isinstance(info, dict) and info.get('outlier_percentage', 0) > 10
            ]
            if high_outlier_cols:
                preprocessing_needed.append("이상치 처리 고려")
        
        # 적합한 분석 방법 추천
        request_metadata = input_data.get('request_metadata', {})
        analysis_type = request_metadata.get('analysis_type', 'unknown')
        
        num_vars = len(variable_analysis['numerical_variables'])
        cat_vars = len(variable_analysis['categorical_variables'])
        
        if analysis_type == 'group_comparison':
            if cat_vars > 0 and num_vars > 0:
                suitable_analyses.append("그룹 간 평균 비교 (t-검정, ANOVA)")
            if cat_vars > 1:
                suitable_analyses.append("범주형 변수 간 연관성 분석 (카이제곱 검정)")
        
        elif analysis_type == 'relationship':
            if num_vars > 1:
                suitable_analyses.append("상관관계 분석")
                suitable_analyses.append("회귀분석")
        
        elif analysis_type == 'categorical':
            if cat_vars > 1:
                suitable_analyses.append("카이제곱 독립성 검정")
                suitable_analyses.append("Fisher의 정확검정")
        
        # 잠재적 도전과제
        if len(data) < 30:
            potential_challenges.append("작은 표본 크기로 인한 검정력 제한")
        
        if quality_assessment['missing_values']['total_missing'] > len(data) * 0.1:
            potential_challenges.append("높은 결측치 비율로 인한 편향 가능성")
        
        if num_vars > 0:
            # 정규성 간단 체크
            numerical_data = data.select_dtypes(include=[np.number])
            for col in numerical_data.columns:
                if len(numerical_data[col].dropna()) > 0:
                    # 간단한 정규성 체크 (왜도, 첨도)
                    skewness = numerical_data[col].skew()
                    if abs(skewness) > 2:
                        potential_challenges.append(f"{col} 변수의 비정규성 (왜도: {round(skewness, 2)})")
                        break
        
        return {
            'preprocessing_needed': preprocessing_needed,
            'suitable_analyses': suitable_analyses,
            'potential_challenges': potential_challenges
        }
    
    def _generate_summary_insights(self, data_overview: Dict, descriptive_stats: Dict,
                                 quality_assessment: Dict, variable_analysis: Dict,
                                 recommendations: Dict) -> Dict[str, Any]:
        """요약 인사이트 생성"""
        key_findings = []
        data_characteristics = []
        
        # 주요 발견사항
        total_rows = data_overview['basic_info']['total_rows']
        total_cols = data_overview['basic_info']['total_columns']
        key_findings.append(f"데이터셋 크기: {total_rows:,}행 × {total_cols}열")
        
        num_vars = len(variable_analysis['numerical_variables'])
        cat_vars = len(variable_analysis['categorical_variables'])
        key_findings.append(f"변수 구성: 수치형 {num_vars}개, 범주형 {cat_vars}개")
        
        missing_total = quality_assessment['missing_values']['total_missing']
        if missing_total > 0:
            missing_pct = round((missing_total / (total_rows * total_cols)) * 100, 1)
            key_findings.append(f"결측치: 전체 데이터의 {missing_pct}%")
        
        # 데이터 특성
        if total_rows >= 1000:
            data_characteristics.append("대용량 데이터셋")
        elif total_rows < 100:
            data_characteristics.append("소규모 데이터셋")
        else:
            data_characteristics.append("중간 규모 데이터셋")
        
        if quality_assessment['duplicates']['duplicate_percentage'] < 1:
            data_characteristics.append("중복 데이터 거의 없음")
        
        if missing_total == 0:
            data_characteristics.append("결측치 없는 완전한 데이터")
        
        # 분석 준비도 평가
        readiness_score = 0
        
        # 긍정적 요소
        if missing_total == 0:
            readiness_score += 30
        elif missing_total < total_rows * total_cols * 0.05:
            readiness_score += 20
        
        if quality_assessment['duplicates']['duplicate_percentage'] < 5:
            readiness_score += 20
        
        if total_rows >= 30:
            readiness_score += 20
        
        if len(recommendations['preprocessing_needed']) <= 2:
            readiness_score += 15
        
        if num_vars > 0 and cat_vars > 0:
            readiness_score += 15  # 다양한 변수 타입
        
        # 분석 준비도 결정
        if readiness_score >= 80:
            analysis_readiness = "excellent"
        elif readiness_score >= 60:
            analysis_readiness = "good"
        elif readiness_score >= 40:
            analysis_readiness = "fair"
        else:
            analysis_readiness = "poor"
        
        return {
            'key_findings': key_findings,
            'data_characteristics': data_characteristics,
            'analysis_readiness': analysis_readiness,
            'readiness_score': readiness_score
        }
    
    def get_step_info(self) -> Dict[str, Any]:
        """단계 정보 반환 (부모 클래스 메서드 확장)"""
        base_info = super().get_step_info()
        base_info.update({
            'description': '데이터 심층 분석 및 요약',
            'input_requirements': ['selected_file', 'file_info', 'user_request', 'refined_objectives'],
            'output_provides': [
                'data_overview', 'descriptive_statistics', 'data_quality_assessment',
                'variable_analysis', 'analysis_recommendations', 'summary_insights'
            ],
            'capabilities': [
                '기술 통계 계산', '데이터 품질 평가', '변수 관계 분석',
                '결측치/이상치 탐지', '분석 추천사항 제공'
            ]
        })
        return base_info


# 단계 등록
PipelineStepRegistry.register_step(3, DataSummaryStep) 