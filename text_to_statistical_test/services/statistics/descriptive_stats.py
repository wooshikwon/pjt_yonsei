"""
Descriptive Statistics

기술통계 계산 서비스
- 중심경향성 (평균, 중앙값, 최빈값)
- 분산성 (표준편차, 분산, 범위)
- 분포 특성 (왜도, 첨도, 정규성)
- 상관관계 분석
- 그룹별 통계
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

from utils.error_handler import ErrorHandler, StatisticsException
from utils.helpers import safe_divide

logger = logging.getLogger(__name__)

class StatisticType(Enum):
    """통계량 타입"""
    CENTRAL_TENDENCY = "central_tendency"
    VARIABILITY = "variability"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    FREQUENCY = "frequency"

@dataclass
class DescriptiveResult:
    """기술통계 결과"""
    statistics: Dict[str, Any]
    data_info: Dict[str, Any]
    warnings: List[str]
    metadata: Dict[str, Any]

class DescriptiveStats:
    """기술통계 메인 클래스"""
    
    def __init__(self):
        """기술통계 계산기 초기화"""
        self.error_handler = ErrorHandler()
        logger.info("기술통계 계산기 초기화 완료")
    
    def calculate_all_statistics(self, 
                                data: pd.DataFrame,
                                target_column: Optional[str] = None,
                                group_by: Optional[str] = None,
                                include_correlations: bool = True) -> DescriptiveResult:
        """
        모든 기술통계 계산
        
        Args:
            data: 분석할 데이터
            target_column: 타겟 컬럼 (있으면 특별 분석)
            group_by: 그룹별 분석할 컬럼
            include_correlations: 상관관계 분석 포함 여부
            
        Returns:
            DescriptiveResult: 기술통계 결과
        """
        try:
            warnings_list = []
            
            # 데이터 기본 정보
            data_info = self._get_data_info(data)
            
            # 수치형/범주형 컬럼 분리
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            statistics = {}
            
            # 1. 수치형 변수 기술통계
            if numeric_columns:
                statistics['numeric'] = self._calculate_numeric_statistics(
                    data[numeric_columns], target_column
                )
            
            # 2. 범주형 변수 기술통계
            if categorical_columns:
                statistics['categorical'] = self._calculate_categorical_statistics(
                    data[categorical_columns]
                )
            
            # 3. 상관관계 분석
            if include_correlations and len(numeric_columns) > 1:
                statistics['correlations'] = self._calculate_correlations(
                    data[numeric_columns]
                )
            
            # 4. 그룹별 분석
            if group_by and group_by in data.columns:
                statistics['group_analysis'] = self._calculate_group_statistics(
                    data, group_by, numeric_columns
                )
            
            # 5. 타겟 변수 특별 분석
            if target_column and target_column in data.columns:
                statistics['target_analysis'] = self._analyze_target_variable(
                    data, target_column, numeric_columns, categorical_columns
                )
            
            # 6. 데이터 품질 지표
            statistics['data_quality'] = self._calculate_data_quality_metrics(data)
            
            result = DescriptiveResult(
                statistics=statistics,
                data_info=data_info,
                warnings=warnings_list,
                metadata={
                    'numeric_columns': numeric_columns,
                    'categorical_columns': categorical_columns,
                    'target_column': target_column,
                    'group_by': group_by
                }
            )
            
            logger.info(f"기술통계 계산 완료 - 수치형: {len(numeric_columns)}, 범주형: {len(categorical_columns)}")
            return result
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'data_shape': data.shape})
            raise StatisticsException(f"기술통계 계산 실패: {error_info['message']}")
    
    def _get_data_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """데이터 기본 정보"""
        return {
            'shape': data.shape,
            'columns': data.columns.tolist(),
            'dtypes': data.dtypes.to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'missing_values': data.isnull().sum().to_dict(),
            'duplicate_rows': data.duplicated().sum()
        }
    
    def _calculate_numeric_statistics(self, 
                                    data: pd.DataFrame, 
                                    target_column: Optional[str] = None) -> Dict[str, Any]:
        """수치형 변수 기술통계"""
        stats = {}
        
        for column in data.columns:
            series = data[column].dropna()
            
            if len(series) == 0:
                stats[column] = {'error': 'No valid data'}
                continue
            
            col_stats = {}
            
            # 중심경향성
            col_stats['central_tendency'] = {
                'mean': float(series.mean()),
                'median': float(series.median()),
                'mode': float(series.mode().iloc[0]) if not series.mode().empty else None,
                'trimmed_mean_5': float(series.quantile([0.05, 0.95]).mean()),
                'geometric_mean': float(np.exp(np.log(series[series > 0]).mean())) if (series > 0).all() else None
            }
            
            # 분산성
            col_stats['variability'] = {
                'std': float(series.std()),
                'variance': float(series.var()),
                'range': float(series.max() - series.min()),
                'iqr': float(series.quantile(0.75) - series.quantile(0.25)),
                'mad': float(np.median(np.abs(series - series.median()))),  # Median Absolute Deviation
                'cv': safe_divide(series.std(), series.mean())  # Coefficient of Variation
            }
            
            # 분포 특성
            col_stats['distribution'] = {
                'skewness': float(series.skew()),
                'kurtosis': float(series.kurtosis()),
                'min': float(series.min()),
                'max': float(series.max()),
                'q25': float(series.quantile(0.25)),
                'q75': float(series.quantile(0.75)),
                'q10': float(series.quantile(0.10)),
                'q90': float(series.quantile(0.90))
            }
            
            # 정규성 검정
            normality_result = self._test_normality(series)
            col_stats['normality'] = normality_result
            
            # 이상치 정보
            outlier_info = self._detect_outliers_info(series)
            col_stats['outliers'] = outlier_info
            
            stats[column] = col_stats
        
        return stats
    
    def _calculate_categorical_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """범주형 변수 기술통계"""
        stats = {}
        
        for column in data.columns:
            series = data[column].dropna()
            
            if len(series) == 0:
                stats[column] = {'error': 'No valid data'}
                continue
            
            # 빈도 분석
            value_counts = series.value_counts()
            proportions = series.value_counts(normalize=True)
            
            col_stats = {
                'frequency': {
                    'unique_count': len(value_counts),
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                    'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                    'value_counts': value_counts.head(10).to_dict(),  # 상위 10개만
                    'proportions': proportions.head(10).to_dict()
                },
                'diversity': {
                    'entropy': self._calculate_entropy(proportions),
                    'gini_coefficient': self._calculate_gini_coefficient(proportions),
                    'concentration_ratio': proportions.iloc[0] if len(proportions) > 0 else 0  # 최빈값 비율
                }
            }
            
            stats[column] = col_stats
        
        return stats
    
    def _calculate_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """상관관계 분석"""
        correlations = {}
        
        # Pearson 상관계수
        pearson_corr = data.corr(method='pearson')
        correlations['pearson'] = pearson_corr.to_dict()
        
        # Spearman 상관계수
        spearman_corr = data.corr(method='spearman')
        correlations['spearman'] = spearman_corr.to_dict()
        
        # Kendall 상관계수
        try:
            kendall_corr = data.corr(method='kendall')
            correlations['kendall'] = kendall_corr.to_dict()
        except Exception:
            logger.warning("Kendall 상관계수 계산 실패")
        
        # 강한 상관관계 찾기
        strong_correlations = self._find_strong_correlations(pearson_corr)
        correlations['strong_correlations'] = strong_correlations
        
        return correlations
    
    def _calculate_group_statistics(self, 
                                  data: pd.DataFrame, 
                                  group_by: str, 
                                  numeric_columns: List[str]) -> Dict[str, Any]:
        """그룹별 통계"""
        group_stats = {}
        
        try:
            grouped = data.groupby(group_by)
            
            # 그룹 기본 정보
            group_stats['group_info'] = {
                'group_count': len(grouped),
                'group_sizes': grouped.size().to_dict(),
                'group_names': list(grouped.groups.keys())
            }
            
            # 수치형 변수별 그룹 통계
            for column in numeric_columns:
                if column in data.columns:
                    group_column_stats = {}
                    
                    # 기본 통계
                    group_desc = grouped[column].describe()
                    group_column_stats['descriptive'] = group_desc.to_dict()
                    
                    # 그룹간 차이 검정 (ANOVA)
                    anova_result = self._perform_anova(data, column, group_by)
                    group_column_stats['anova'] = anova_result
                    
                    group_stats[column] = group_column_stats
            
        except Exception as e:
            logger.error(f"그룹별 통계 계산 오류: {str(e)}")
            group_stats['error'] = str(e)
        
        return group_stats
    
    def _analyze_target_variable(self, 
                               data: pd.DataFrame, 
                               target_column: str,
                               numeric_columns: List[str],
                               categorical_columns: List[str]) -> Dict[str, Any]:
        """타겟 변수 특별 분석"""
        target_analysis = {}
        
        target_series = data[target_column]
        
        # 타겟 변수 타입 확인
        is_numeric_target = target_column in numeric_columns
        
        if is_numeric_target:
            # 수치형 타겟
            target_analysis['type'] = 'numeric'
            
            # 다른 수치형 변수와의 관계
            other_numeric = [col for col in numeric_columns if col != target_column]
            if other_numeric:
                correlations_with_target = data[other_numeric + [target_column]].corr()[target_column].drop(target_column)
                target_analysis['correlations_with_features'] = correlations_with_target.to_dict()
            
            # 범주형 변수별 타겟 분포
            for cat_col in categorical_columns:
                if cat_col in data.columns:
                    group_stats = data.groupby(cat_col)[target_column].describe()
                    target_analysis[f'by_{cat_col}'] = group_stats.to_dict()
        
        else:
            # 범주형 타겟
            target_analysis['type'] = 'categorical'
            
            # 클래스 분포
            class_distribution = target_series.value_counts()
            target_analysis['class_distribution'] = class_distribution.to_dict()
            target_analysis['class_proportions'] = target_series.value_counts(normalize=True).to_dict()
            
            # 수치형 변수별 타겟 클래스 분포
            for num_col in numeric_columns:
                if num_col in data.columns:
                    group_stats = data.groupby(target_column)[num_col].describe()
                    target_analysis[f'feature_{num_col}'] = group_stats.to_dict()
        
        return target_analysis
    
    def _calculate_data_quality_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """데이터 품질 지표"""
        quality_metrics = {}
        
        # 완성도 (Completeness)
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells)
        
        # 일관성 (Consistency) - 데이터 타입 일관성
        type_consistency = {}
        for column in data.columns:
            if data[column].dtype == 'object':
                # 문자열 컬럼의 일관성 체크
                non_null_values = data[column].dropna()
                if len(non_null_values) > 0:
                    # 숫자로 변환 가능한 비율
                    numeric_convertible = 0
                    for value in non_null_values.sample(min(100, len(non_null_values))):
                        try:
                            float(value)
                            numeric_convertible += 1
                        except:
                            pass
                    type_consistency[column] = numeric_convertible / min(100, len(non_null_values))
        
        # 유일성 (Uniqueness)
        uniqueness = {}
        for column in data.columns:
            total_values = len(data[column].dropna())
            unique_values = data[column].nunique()
            uniqueness[column] = unique_values / total_values if total_values > 0 else 0
        
        quality_metrics = {
            'completeness': completeness,
            'missing_percentage': (missing_cells / total_cells) * 100,
            'type_consistency': type_consistency,
            'uniqueness': uniqueness,
            'duplicate_percentage': (data.duplicated().sum() / len(data)) * 100,
            'overall_quality_score': self._calculate_overall_quality_score(
                completeness, type_consistency, uniqueness, data
            )
        }
        
        return quality_metrics
    
    def _test_normality(self, series: pd.Series) -> Dict[str, Any]:
        """정규성 검정"""
        result = {
            'is_normal': False,
            'tests': {}
        }
        
        try:
            from scipy import stats
            
            # Shapiro-Wilk 검정 (샘플 크기 < 5000)
            if len(series) < 5000:
                shapiro_stat, shapiro_p = stats.shapiro(series)
                result['tests']['shapiro'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }
            
            # Kolmogorov-Smirnov 검정
            ks_stat, ks_p = stats.kstest(series, 'norm', args=(series.mean(), series.std()))
            result['tests']['kolmogorov_smirnov'] = {
                'statistic': float(ks_stat),
                'p_value': float(ks_p),
                'is_normal': ks_p > 0.05
            }
            
            # Anderson-Darling 검정
            ad_result = stats.anderson(series, dist='norm')
            result['tests']['anderson_darling'] = {
                'statistic': float(ad_result.statistic),
                'critical_values': ad_result.critical_values.tolist(),
                'significance_levels': ad_result.significance_level.tolist()
            }
            
            # 전체 정규성 판단 (여러 검정 결과 종합)
            normal_tests = [test.get('is_normal', False) for test in result['tests'].values() if 'is_normal' in test]
            result['is_normal'] = sum(normal_tests) >= len(normal_tests) / 2
            
        except ImportError:
            logger.warning("scipy가 설치되지 않아 정규성 검정을 스킵합니다")
        except Exception as e:
            logger.warning(f"정규성 검정 오류: {str(e)}")
        
        return result
    
    def _detect_outliers_info(self, series: pd.Series) -> Dict[str, Any]:
        """이상치 정보"""
        outlier_info = {}
        
        # IQR 방법
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        outlier_info['iqr_method'] = {
            'count': len(iqr_outliers),
            'percentage': (len(iqr_outliers) / len(series)) * 100,
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'outlier_values': iqr_outliers.tolist()[:10]  # 최대 10개만
        }
        
        # Z-score 방법
        z_scores = np.abs((series - series.mean()) / series.std())
        z_outliers = series[z_scores > 3]
        
        outlier_info['z_score_method'] = {
            'count': len(z_outliers),
            'percentage': (len(z_outliers) / len(series)) * 100,
            'threshold': 3.0,
            'outlier_values': z_outliers.tolist()[:10]
        }
        
        return outlier_info
    
    def _calculate_entropy(self, proportions: pd.Series) -> float:
        """엔트로피 계산"""
        # Shannon entropy
        entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
        return float(entropy)
    
    def _calculate_gini_coefficient(self, proportions: pd.Series) -> float:
        """지니 계수 계산"""
        sorted_props = np.sort(proportions.values)
        n = len(sorted_props)
        cumsum = np.cumsum(sorted_props)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        return float(gini)
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """강한 상관관계 찾기"""
        strong_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_corrs.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': float(corr_value),
                        'strength': 'very_strong' if abs(corr_value) >= 0.9 else 'strong'
                    })
        
        # 상관계수 절댓값으로 정렬
        strong_corrs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return strong_corrs
    
    def _perform_anova(self, data: pd.DataFrame, numeric_column: str, group_column: str) -> Dict[str, Any]:
        """일원분산분석 (ANOVA)"""
        try:
            from scipy import stats
            
            groups = [group[numeric_column].dropna() for name, group in data.groupby(group_column)]
            
            # 그룹이 2개 이상이어야 ANOVA 수행 가능
            if len(groups) < 2:
                return {'error': 'Need at least 2 groups for ANOVA'}
            
            # 각 그룹에 데이터가 있어야 함
            groups = [group for group in groups if len(group) > 0]
            if len(groups) < 2:
                return {'error': 'Need at least 2 non-empty groups for ANOVA'}
            
            f_stat, p_value = stats.f_oneway(*groups)
            
            return {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'groups_count': len(groups),
                'total_observations': sum(len(group) for group in groups)
            }
            
        except ImportError:
            return {'error': 'scipy not available for ANOVA'}
        except Exception as e:
            return {'error': f'ANOVA calculation failed: {str(e)}'}
    
    def _calculate_overall_quality_score(self, 
                                       completeness: float,
                                       type_consistency: Dict[str, float],
                                       uniqueness: Dict[str, float],
                                       data: pd.DataFrame) -> float:
        """전체 데이터 품질 점수 계산"""
        score = 0.0
        
        # 완성도 (40%)
        score += completeness * 0.4
        
        # 타입 일관성 (20%)
        if type_consistency:
            avg_consistency = np.mean(list(type_consistency.values()))
            score += avg_consistency * 0.2
        else:
            score += 0.2  # 타입 불일치가 없으면 만점
        
        # 유일성 적절성 (20%) - 너무 높거나 낮으면 감점
        if uniqueness:
            avg_uniqueness = np.mean(list(uniqueness.values()))
            # 0.1 ~ 0.9 범위가 적절하다고 가정
            uniqueness_score = 1.0 - abs(avg_uniqueness - 0.5) * 2
            score += max(0, uniqueness_score) * 0.2
        
        # 중복 없음 (20%)
        duplicate_ratio = data.duplicated().sum() / len(data)
        score += (1 - duplicate_ratio) * 0.2
        
        return min(1.0, max(0.0, score))
    
    def generate_summary_report(self, result: DescriptiveResult) -> str:
        """기술통계 요약 보고서 생성"""
        report = []
        
        report.append("=== 기술통계 분석 보고서 ===\n")
        
        # 데이터 개요
        data_info = result.data_info
        report.append(f"데이터 크기: {data_info['shape'][0]}행 x {data_info['shape'][1]}열")
        report.append(f"메모리 사용량: {data_info['memory_usage'] / 1024**2:.2f} MB")
        
        missing_total = sum(data_info['missing_values'].values())
        if missing_total > 0:
            report.append(f"결측값: {missing_total}개")
        
        if data_info['duplicate_rows'] > 0:
            report.append(f"중복 행: {data_info['duplicate_rows']}개")
        
        report.append("")
        
        # 수치형 변수 요약
        if 'numeric' in result.statistics:
            report.append("=== 수치형 변수 요약 ===")
            numeric_stats = result.statistics['numeric']
            
            for column, stats in numeric_stats.items():
                if 'error' in stats:
                    continue
                    
                central = stats['central_tendency']
                variability = stats['variability']
                distribution = stats['distribution']
                
                report.append(f"\n[{column}]")
                report.append(f"  평균: {central['mean']:.3f}, 중앙값: {central['median']:.3f}")
                report.append(f"  표준편차: {variability['std']:.3f}, 범위: {variability['range']:.3f}")
                report.append(f"  왜도: {distribution['skewness']:.3f}, 첨도: {distribution['kurtosis']:.3f}")
                
                if stats.get('outliers', {}).get('iqr_method', {}).get('count', 0) > 0:
                    outlier_count = stats['outliers']['iqr_method']['count']
                    outlier_pct = stats['outliers']['iqr_method']['percentage']
                    report.append(f"  이상치: {outlier_count}개 ({outlier_pct:.1f}%)")
        
        # 범주형 변수 요약
        if 'categorical' in result.statistics:
            report.append("\n=== 범주형 변수 요약 ===")
            categorical_stats = result.statistics['categorical']
            
            for column, stats in categorical_stats.items():
                if 'error' in stats:
                    continue
                    
                freq = stats['frequency']
                diversity = stats['diversity']
                
                report.append(f"\n[{column}]")
                report.append(f"  고유값 수: {freq['unique_count']}")
                report.append(f"  최빈값: {freq['most_frequent']} ({freq['most_frequent_count']}회)")
                report.append(f"  엔트로피: {diversity['entropy']:.3f}")
                report.append(f"  집중도: {diversity['concentration_ratio']:.3f}")
        
        # 상관관계 요약
        if 'correlations' in result.statistics:
            strong_corrs = result.statistics['correlations'].get('strong_correlations', [])
            if strong_corrs:
                report.append("\n=== 강한 상관관계 ===")
                for corr in strong_corrs[:5]:  # 상위 5개만
                    report.append(f"  {corr['variable1']} - {corr['variable2']}: {corr['correlation']:.3f}")
        
        # 데이터 품질
        if 'data_quality' in result.statistics:
            quality = result.statistics['data_quality']
            report.append(f"\n=== 데이터 품질 ===")
            report.append(f"전체 품질 점수: {quality['overall_quality_score']:.3f}")
            report.append(f"완성도: {quality['completeness']:.3f}")
            report.append(f"결측률: {quality['missing_percentage']:.1f}%")
            report.append(f"중복률: {quality['duplicate_percentage']:.1f}%")
        
        # 경고사항
        if result.warnings:
            report.append("\n=== 주의사항 ===")
            for warning in result.warnings:
                report.append(f"  - {warning}")
        
        return "\n".join(report) 