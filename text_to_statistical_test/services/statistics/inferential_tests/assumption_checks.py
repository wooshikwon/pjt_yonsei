"""
Assumption Checks

통계적 가정 검정 모듈
- 정규성 검정 (Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling)
- 등분산성 검정 (Levene, Bartlett, Brown-Forsythe)
- 독립성 검정
- 선형성 검정
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

class AssumptionType(Enum):
    """가정 검정 타입"""
    NORMALITY = "normality"
    HOMOSCEDASTICITY = "homoscedasticity"
    INDEPENDENCE = "independence"
    LINEARITY = "linearity"

@dataclass
class AssumptionResult:
    """가정 검정 결과"""
    assumption_type: AssumptionType
    is_satisfied: bool
    test_results: Dict[str, Any]
    recommendations: List[str]
    confidence_level: float
    metadata: Dict[str, Any]

class AssumptionChecks:
    """통계적 가정 검정 메인 클래스"""
    
    def __init__(self, alpha: float = 0.05):
        """
        가정 검정 클래스 초기화
        
        Args:
            alpha: 유의수준 (기본값: 0.05)
        """
        self.alpha = alpha
        self.error_handler = ErrorHandler()
        logger.info("가정 검정 클래스 초기화 완료")
    
    def check_normality(self, 
                       data: Union[pd.Series, np.ndarray, List],
                       method: str = "auto") -> AssumptionResult:
        """
        정규성 검정
        
        Args:
            data: 검정할 데이터
            method: 검정 방법 ("shapiro", "ks", "anderson", "auto")
            
        Returns:
            AssumptionResult: 정규성 검정 결과
        """
        try:
            if isinstance(data, (list, np.ndarray)):
                data = pd.Series(data)
            
            # 결측값 제거
            clean_data = data.dropna()
            
            if len(clean_data) < 3:
                raise StatisticsException("정규성 검정을 위한 데이터가 부족합니다 (최소 3개 필요)")
            
            test_results = {}
            recommendations = []
            
            # 자동 방법 선택
            if method == "auto":
                if len(clean_data) <= 50:
                    method = "shapiro"
                elif len(clean_data) <= 5000:
                    method = "ks"
                else:
                    method = "anderson"
            
            # Shapiro-Wilk 검정
            if method in ["shapiro", "all"] and len(clean_data) <= 5000:
                shapiro_result = self._shapiro_test(clean_data)
                test_results["shapiro"] = shapiro_result
            
            # Kolmogorov-Smirnov 검정
            if method in ["ks", "all"]:
                ks_result = self._kolmogorov_smirnov_test(clean_data)
                test_results["kolmogorov_smirnov"] = ks_result
            
            # Anderson-Darling 검정
            if method in ["anderson", "all"]:
                anderson_result = self._anderson_darling_test(clean_data)
                test_results["anderson_darling"] = anderson_result
            
            # 전체 정규성 판단
            is_normal = self._evaluate_normality(test_results)
            
            # 권장사항 생성
            if not is_normal:
                recommendations.extend([
                    "데이터가 정규분포를 따르지 않습니다.",
                    "비모수 검정 사용을 고려하세요.",
                    "데이터 변환(로그, 제곱근 등)을 시도해보세요."
                ])
            
            return AssumptionResult(
                assumption_type=AssumptionType.NORMALITY,
                is_satisfied=is_normal,
                test_results=test_results,
                recommendations=recommendations,
                confidence_level=1 - self.alpha,
                metadata={
                    "sample_size": len(clean_data),
                    "method_used": method,
                    "missing_values": len(data) - len(clean_data)
                }
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'data_size': len(data)})
            raise StatisticsException(f"정규성 검정 실패: {error_info['message']}")
    
    def check_homoscedasticity(self, 
                              *groups: Union[pd.Series, np.ndarray, List],
                              method: str = "levene") -> AssumptionResult:
        """
        등분산성 검정
        
        Args:
            *groups: 비교할 그룹들
            method: 검정 방법 ("levene", "bartlett", "brown_forsythe")
            
        Returns:
            AssumptionResult: 등분산성 검정 결과
        """
        try:
            if len(groups) < 2:
                raise StatisticsException("등분산성 검정을 위해서는 최소 2개 그룹이 필요합니다")
            
            # 데이터 정리
            clean_groups = []
            for group in groups:
                if isinstance(group, (list, np.ndarray)):
                    group = pd.Series(group)
                clean_group = group.dropna()
                if len(clean_group) < 2:
                    raise StatisticsException("각 그룹은 최소 2개의 관측값이 필요합니다")
                clean_groups.append(clean_group)
            
            test_results = {}
            recommendations = []
            
            # Levene 검정
            if method in ["levene", "all"]:
                levene_result = self._levene_test(clean_groups)
                test_results["levene"] = levene_result
            
            # Bartlett 검정
            if method in ["bartlett", "all"]:
                bartlett_result = self._bartlett_test(clean_groups)
                test_results["bartlett"] = bartlett_result
            
            # Brown-Forsythe 검정
            if method in ["brown_forsythe", "all"]:
                bf_result = self._brown_forsythe_test(clean_groups)
                test_results["brown_forsythe"] = bf_result
            
            # 전체 등분산성 판단
            is_homoscedastic = self._evaluate_homoscedasticity(test_results)
            
            # 권장사항 생성
            if not is_homoscedastic:
                recommendations.extend([
                    "그룹 간 분산이 동일하지 않습니다.",
                    "Welch's t-test 또는 비모수 검정을 고려하세요.",
                    "데이터 변환을 통해 분산을 안정화해보세요."
                ])
            
            return AssumptionResult(
                assumption_type=AssumptionType.HOMOSCEDASTICITY,
                is_satisfied=is_homoscedastic,
                test_results=test_results,
                recommendations=recommendations,
                confidence_level=1 - self.alpha,
                metadata={
                    "num_groups": len(clean_groups),
                    "group_sizes": [len(group) for group in clean_groups],
                    "method_used": method
                }
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'num_groups': len(groups)})
            raise StatisticsException(f"등분산성 검정 실패: {error_info['message']}")
    
    def check_independence(self, 
                          data: pd.DataFrame,
                          time_column: Optional[str] = None,
                          method: str = "durbin_watson") -> AssumptionResult:
        """
        독립성 검정
        
        Args:
            data: 검정할 데이터
            time_column: 시간 컬럼명
            method: 검정 방법 ("durbin_watson", "ljung_box")
            
        Returns:
            AssumptionResult: 독립성 검정 결과
        """
        try:
            test_results = {}
            recommendations = []
            
            if method == "durbin_watson":
                dw_result = self._durbin_watson_test(data, time_column)
                test_results["durbin_watson"] = dw_result
            
            elif method == "ljung_box":
                lb_result = self._ljung_box_test(data, time_column)
                test_results["ljung_box"] = lb_result
            
            # 독립성 판단
            is_independent = self._evaluate_independence(test_results)
            
            if not is_independent:
                recommendations.extend([
                    "데이터에 자기상관이 존재할 수 있습니다.",
                    "시계열 분석 방법을 고려하세요.",
                    "잔차 분석을 통해 패턴을 확인하세요."
                ])
            
            return AssumptionResult(
                assumption_type=AssumptionType.INDEPENDENCE,
                is_satisfied=is_independent,
                test_results=test_results,
                recommendations=recommendations,
                confidence_level=1 - self.alpha,
                metadata={
                    "method_used": method,
                    "data_shape": data.shape
                }
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'method': method})
            raise StatisticsException(f"독립성 검정 실패: {error_info['message']}")
    
    def check_linearity(self, 
                       x: Union[pd.Series, np.ndarray],
                       y: Union[pd.Series, np.ndarray],
                       method: str = "rainbow") -> AssumptionResult:
        """
        선형성 검정
        
        Args:
            x: 독립변수
            y: 종속변수
            method: 검정 방법 ("rainbow", "harvey_collier")
            
        Returns:
            AssumptionResult: 선형성 검정 결과
        """
        try:
            if isinstance(x, (list, np.ndarray)):
                x = pd.Series(x)
            if isinstance(y, (list, np.ndarray)):
                y = pd.Series(y)
            
            # 결측값 제거
            valid_idx = ~(x.isna() | y.isna())
            x_clean = x[valid_idx]
            y_clean = y[valid_idx]
            
            if len(x_clean) < 10:
                raise StatisticsException("선형성 검정을 위해서는 최소 10개의 관측값이 필요합니다")
            
            test_results = {}
            recommendations = []
            
            if method == "rainbow":
                rainbow_result = self._rainbow_test(x_clean, y_clean)
                test_results["rainbow"] = rainbow_result
            
            elif method == "harvey_collier":
                hc_result = self._harvey_collier_test(x_clean, y_clean)
                test_results["harvey_collier"] = hc_result
            
            # 선형성 판단
            is_linear = self._evaluate_linearity(test_results)
            
            if not is_linear:
                recommendations.extend([
                    "변수 간 선형 관계가 명확하지 않습니다.",
                    "비선형 모델을 고려하세요.",
                    "변수 변환을 시도해보세요."
                ])
            
            return AssumptionResult(
                assumption_type=AssumptionType.LINEARITY,
                is_satisfied=is_linear,
                test_results=test_results,
                recommendations=recommendations,
                confidence_level=1 - self.alpha,
                metadata={
                    "sample_size": len(x_clean),
                    "method_used": method,
                    "correlation": float(x_clean.corr(y_clean))
                }
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'method': method})
            raise StatisticsException(f"선형성 검정 실패: {error_info['message']}")
    
    def _shapiro_test(self, data: pd.Series) -> Dict[str, Any]:
        """Shapiro-Wilk 검정"""
        from scipy import stats
        
        statistic, p_value = stats.shapiro(data)
        
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_normal": p_value > self.alpha,
            "interpretation": "정규분포를 따름" if p_value > self.alpha else "정규분포를 따르지 않음"
        }
    
    def _kolmogorov_smirnov_test(self, data: pd.Series) -> Dict[str, Any]:
        """Kolmogorov-Smirnov 검정"""
        from scipy import stats
        
        # 표준화된 데이터로 정규분포와 비교
        standardized = (data - data.mean()) / data.std()
        statistic, p_value = stats.kstest(standardized, 'norm')
        
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_normal": p_value > self.alpha,
            "interpretation": "정규분포를 따름" if p_value > self.alpha else "정규분포를 따르지 않음"
        }
    
    def _anderson_darling_test(self, data: pd.Series) -> Dict[str, Any]:
        """Anderson-Darling 검정"""
        from scipy import stats
        
        result = stats.anderson(data, dist='norm')
        
        # 5% 유의수준에서의 임계값 (일반적으로 인덱스 2)
        critical_value = result.critical_values[2] if len(result.critical_values) > 2 else result.critical_values[-1]
        is_normal = result.statistic < critical_value
        
        return {
            "statistic": float(result.statistic),
            "critical_values": result.critical_values.tolist(),
            "significance_levels": result.significance_level.tolist(),
            "is_normal": is_normal,
            "interpretation": "정규분포를 따름" if is_normal else "정규분포를 따르지 않음"
        }
    
    def _levene_test(self, groups: List[pd.Series]) -> Dict[str, Any]:
        """Levene 검정"""
        from scipy import stats
        
        statistic, p_value = stats.levene(*groups)
        
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_homoscedastic": p_value > self.alpha,
            "interpretation": "등분산성 만족" if p_value > self.alpha else "등분산성 위배"
        }
    
    def _bartlett_test(self, groups: List[pd.Series]) -> Dict[str, Any]:
        """Bartlett 검정"""
        from scipy import stats
        
        statistic, p_value = stats.bartlett(*groups)
        
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_homoscedastic": p_value > self.alpha,
            "interpretation": "등분산성 만족" if p_value > self.alpha else "등분산성 위배"
        }
    
    def _brown_forsythe_test(self, groups: List[pd.Series]) -> Dict[str, Any]:
        """Brown-Forsythe 검정"""
        from scipy import stats
        
        # Levene 검정의 변형 (중앙값 사용)
        statistic, p_value = stats.levene(*groups, center='median')
        
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_homoscedastic": p_value > self.alpha,
            "interpretation": "등분산성 만족" if p_value > self.alpha else "등분산성 위배"
        }
    
    def _durbin_watson_test(self, data: pd.DataFrame, time_column: Optional[str]) -> Dict[str, Any]:
        """Durbin-Watson 검정"""
        try:
            from statsmodels.stats.diagnostic import durbin_watson
            from sklearn.linear_model import LinearRegression
            
            # 시간 순서대로 정렬
            if time_column and time_column in data.columns:
                data_sorted = data.sort_values(time_column)
            else:
                data_sorted = data
            
            # 첫 번째 수치형 컬럼을 종속변수로 사용
            numeric_cols = data_sorted.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                raise StatisticsException("Durbin-Watson 검정을 위해서는 최소 2개의 수치형 변수가 필요합니다")
            
            y = data_sorted[numeric_cols[0]]
            X = data_sorted[numeric_cols[1:]]
            
            # 선형 회귀 모델 적합
            model = LinearRegression()
            model.fit(X, y)
            residuals = y - model.predict(X)
            
            # Durbin-Watson 통계량 계산
            dw_stat = durbin_watson(residuals)
            
            # 해석 (일반적인 기준)
            if 1.5 <= dw_stat <= 2.5:
                interpretation = "자기상관 없음"
                is_independent = True
            elif dw_stat < 1.5:
                interpretation = "양의 자기상관 존재"
                is_independent = False
            else:
                interpretation = "음의 자기상관 존재"
                is_independent = False
            
            return {
                "statistic": float(dw_stat),
                "is_independent": is_independent,
                "interpretation": interpretation
            }
            
        except ImportError:
            # statsmodels가 없는 경우 간단한 구현
            return self._simple_durbin_watson(data, time_column)
    
    def _simple_durbin_watson(self, data: pd.DataFrame, time_column: Optional[str]) -> Dict[str, Any]:
        """간단한 Durbin-Watson 구현"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise StatisticsException("수치형 데이터가 없습니다")
        
        # 첫 번째 수치형 컬럼 사용
        series = data[numeric_cols[0]].dropna()
        
        if len(series) < 3:
            raise StatisticsException("Durbin-Watson 검정을 위해서는 최소 3개의 관측값이 필요합니다")
        
        # 차분 계산
        diff = series.diff().dropna()
        
        # DW 통계량 계산
        dw_stat = (diff ** 2).sum() / (series ** 2).sum()
        
        # 해석
        if 1.5 <= dw_stat <= 2.5:
            interpretation = "자기상관 없음"
            is_independent = True
        else:
            interpretation = "자기상관 존재 가능"
            is_independent = False
        
        return {
            "statistic": float(dw_stat),
            "is_independent": is_independent,
            "interpretation": interpretation
        }
    
    def _ljung_box_test(self, data: pd.DataFrame, time_column: Optional[str]) -> Dict[str, Any]:
        """Ljung-Box 검정"""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise StatisticsException("수치형 데이터가 없습니다")
            
            # 첫 번째 수치형 컬럼 사용
            series = data[numeric_cols[0]].dropna()
            
            if len(series) < 10:
                raise StatisticsException("Ljung-Box 검정을 위해서는 최소 10개의 관측값이 필요합니다")
            
            # Ljung-Box 검정
            result = acorr_ljungbox(series, lags=min(10, len(series)//4), return_df=True)
            
            # 가장 낮은 p-value 사용
            min_p_value = result['lb_pvalue'].min()
            
            return {
                "statistic": float(result['lb_stat'].iloc[-1]),
                "p_value": float(min_p_value),
                "is_independent": min_p_value > self.alpha,
                "interpretation": "독립성 만족" if min_p_value > self.alpha else "자기상관 존재"
            }
            
        except ImportError:
            return {"error": "statsmodels 패키지가 필요합니다"}
    
    def _rainbow_test(self, x: pd.Series, y: pd.Series) -> Dict[str, Any]:
        """Rainbow 선형성 검정"""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import PolynomialFeatures
            from scipy import stats
            
            # 선형 모델
            linear_model = LinearRegression()
            linear_model.fit(x.values.reshape(-1, 1), y)
            linear_pred = linear_model.predict(x.values.reshape(-1, 1))
            linear_rss = ((y - linear_pred) ** 2).sum()
            
            # 2차 다항식 모델
            poly_features = PolynomialFeatures(degree=2)
            x_poly = poly_features.fit_transform(x.values.reshape(-1, 1))
            poly_model = LinearRegression()
            poly_model.fit(x_poly, y)
            poly_pred = poly_model.predict(x_poly)
            poly_rss = ((y - poly_pred) ** 2).sum()
            
            # F 통계량 계산
            n = len(x)
            f_stat = ((linear_rss - poly_rss) / 1) / (poly_rss / (n - 3))
            p_value = 1 - stats.f.cdf(f_stat, 1, n - 3)
            
            return {
                "statistic": float(f_stat),
                "p_value": float(p_value),
                "is_linear": p_value > self.alpha,
                "interpretation": "선형 관계" if p_value > self.alpha else "비선형 관계 가능"
            }
            
        except Exception as e:
            return {"error": f"Rainbow 검정 실패: {str(e)}"}
    
    def _harvey_collier_test(self, x: pd.Series, y: pd.Series) -> Dict[str, Any]:
        """Harvey-Collier 선형성 검정"""
        try:
            from sklearn.linear_model import LinearRegression
            from scipy import stats
            
            # 데이터 정렬
            sorted_idx = x.argsort()
            x_sorted = x.iloc[sorted_idx]
            y_sorted = y.iloc[sorted_idx]
            
            # 선형 회귀
            model = LinearRegression()
            model.fit(x_sorted.values.reshape(-1, 1), y_sorted)
            residuals = y_sorted - model.predict(x_sorted.values.reshape(-1, 1))
            
            # 재귀적 잔차 계산 (간단한 근사)
            n = len(residuals)
            mid = n // 2
            
            first_half_mean = residuals[:mid].mean()
            second_half_mean = residuals[mid:].mean()
            
            # t 통계량 계산
            pooled_std = residuals.std()
            t_stat = (first_half_mean - second_half_mean) / (pooled_std * np.sqrt(2/n))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            
            return {
                "statistic": float(t_stat),
                "p_value": float(p_value),
                "is_linear": p_value > self.alpha,
                "interpretation": "선형 관계" if p_value > self.alpha else "비선형 관계 가능"
            }
            
        except Exception as e:
            return {"error": f"Harvey-Collier 검정 실패: {str(e)}"}
    
    def _evaluate_normality(self, test_results: Dict[str, Any]) -> bool:
        """정규성 검정 결과 종합 평가"""
        normal_count = 0
        total_count = 0
        
        for test_name, result in test_results.items():
            if 'is_normal' in result:
                total_count += 1
                if result['is_normal']:
                    normal_count += 1
        
        # 과반수 이상의 검정에서 정규성을 만족하면 정규분포로 판단
        return normal_count > total_count / 2 if total_count > 0 else False
    
    def _evaluate_homoscedasticity(self, test_results: Dict[str, Any]) -> bool:
        """등분산성 검정 결과 종합 평가"""
        homoscedastic_count = 0
        total_count = 0
        
        for test_name, result in test_results.items():
            if 'is_homoscedastic' in result:
                total_count += 1
                if result['is_homoscedastic']:
                    homoscedastic_count += 1
        
        return homoscedastic_count > total_count / 2 if total_count > 0 else False
    
    def _evaluate_independence(self, test_results: Dict[str, Any]) -> bool:
        """독립성 검정 결과 종합 평가"""
        independent_count = 0
        total_count = 0
        
        for test_name, result in test_results.items():
            if 'is_independent' in result:
                total_count += 1
                if result['is_independent']:
                    independent_count += 1
        
        return independent_count > total_count / 2 if total_count > 0 else False
    
    def _evaluate_linearity(self, test_results: Dict[str, Any]) -> bool:
        """선형성 검정 결과 종합 평가"""
        linear_count = 0
        total_count = 0
        
        for test_name, result in test_results.items():
            if 'is_linear' in result:
                total_count += 1
                if result['is_linear']:
                    linear_count += 1
        
        return linear_count > total_count / 2 if total_count > 0 else False
    
    def check_all_assumptions(self, 
                             data: Union[pd.DataFrame, Dict[str, pd.Series]],
                             test_type: str = "parametric") -> Dict[str, AssumptionResult]:
        """
        모든 가정 검정 수행
        
        Args:
            data: 검정할 데이터
            test_type: 검정 유형 ("parametric", "anova", "regression")
            
        Returns:
            Dict[str, AssumptionResult]: 가정별 검정 결과
        """
        results = {}
        
        try:
            if isinstance(data, dict):
                # 그룹별 데이터인 경우
                groups = list(data.values())
                
                # 각 그룹의 정규성 검정
                for group_name, group_data in data.items():
                    results[f"normality_{group_name}"] = self.check_normality(group_data)
                
                # 등분산성 검정
                if len(groups) > 1:
                    results["homoscedasticity"] = self.check_homoscedasticity(*groups)
            
            elif isinstance(data, pd.DataFrame):
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                
                if test_type == "regression" and len(numeric_cols) >= 2:
                    # 회귀분석용 가정 검정
                    y = data[numeric_cols[0]]
                    
                    # 종속변수 정규성
                    results["normality_dependent"] = self.check_normality(y)
                    
                    # 독립성 검정
                    results["independence"] = self.check_independence(data)
                    
                    # 선형성 검정 (첫 번째 독립변수와)
                    if len(numeric_cols) > 1:
                        x = data[numeric_cols[1]]
                        results["linearity"] = self.check_linearity(x, y)
                
                else:
                    # 일반적인 가정 검정
                    for col in numeric_cols:
                        results[f"normality_{col}"] = self.check_normality(data[col])
            
            logger.info(f"전체 가정 검정 완료 - {len(results)}개 검정 수행")
            return results
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'test_type': test_type})
            raise StatisticsException(f"전체 가정 검정 실패: {error_info['message']}") 