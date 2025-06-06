"""
Regression Tests

회귀분석 모듈
- 선형 회귀분석 (단순, 다중)
- 로지스틱 회귀분석
- 다항 회귀분석
- 회귀 진단 및 가정 검정
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

class RegressionType(Enum):
    """회귀분석 타입"""
    LINEAR = "linear"
    LOGISTIC = "logistic"
    POLYNOMIAL = "polynomial"
    MULTIPLE = "multiple"

@dataclass
class RegressionResult:
    """회귀분석 결과"""
    regression_type: RegressionType
    coefficients: Dict[str, float]
    p_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    r_squared: Optional[float]
    adjusted_r_squared: Optional[float]
    f_statistic: Optional[float]
    f_p_value: Optional[float]
    residuals: np.ndarray
    fitted_values: np.ndarray
    interpretation: str
    diagnostics: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class RegressionDiagnostics:
    """회귀 진단 결과"""
    normality_test: Dict[str, Any]
    homoscedasticity_test: Dict[str, Any]
    independence_test: Dict[str, Any]
    linearity_test: Dict[str, Any]
    outliers: List[int]
    influential_points: List[int]
    multicollinearity: Dict[str, float]

class RegressionTests:
    """회귀분석 메인 클래스"""
    
    def __init__(self, alpha: float = 0.05):
        """
        회귀분석 클래스 초기화
        
        Args:
            alpha: 유의수준 (기본값: 0.05)
        """
        self.alpha = alpha
        self.error_handler = ErrorHandler()
        logger.info("회귀분석 클래스 초기화 완료")
    
    def simple_linear_regression(self, 
                                x: Union[pd.Series, np.ndarray, List],
                                y: Union[pd.Series, np.ndarray, List],
                                x_name: str = "X",
                                y_name: str = "Y") -> RegressionResult:
        """
        단순 선형 회귀분석
        
        Args:
            x: 독립변수
            y: 종속변수
            x_name: 독립변수 이름
            y_name: 종속변수 이름
            
        Returns:
            RegressionResult: 회귀분석 결과
        """
        try:
            from scipy import stats
            
            if isinstance(x, (list, np.ndarray)):
                x = pd.Series(x)
            if isinstance(y, (list, np.ndarray)):
                y = pd.Series(y)
            
            # 길이 확인
            if len(x) != len(y):
                raise StatisticsException("독립변수와 종속변수의 길이가 같아야 합니다")
            
            # 결측값 제거 (쌍으로)
            valid_idx = ~(x.isna() | y.isna())
            clean_x = x[valid_idx]
            clean_y = y[valid_idx]
            
            if len(clean_x) < 3:
                raise StatisticsException("단순 선형 회귀분석을 위해서는 최소 3쌍의 관측값이 필요합니다")
            
            # 회귀분석 수행
            slope, intercept, r_value, p_value, std_err = stats.linregress(clean_x, clean_y)
            
            # 예측값과 잔차
            fitted_values = intercept + slope * clean_x
            residuals = clean_y - fitted_values
            
            # R-squared
            r_squared = r_value ** 2
            
            # 조정된 R-squared
            n = len(clean_x)
            adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2) if n > 2 else r_squared
            
            # F 통계량
            mse_regression = np.sum((fitted_values - clean_y.mean()) ** 2)
            mse_residual = np.sum(residuals ** 2) / (n - 2) if n > 2 else 1
            f_statistic = mse_regression / mse_residual if mse_residual > 0 else 0
            f_p_value = 1 - stats.f.cdf(f_statistic, 1, n - 2) if n > 2 else 1
            
            # 계수의 표준오차와 신뢰구간
            slope_se = std_err
            intercept_se = slope_se * np.sqrt(np.sum(clean_x ** 2) / n)
            
            t_critical = stats.t.ppf(1 - self.alpha/2, n - 2) if n > 2 else 0
            
            slope_ci = (slope - t_critical * slope_se, slope + t_critical * slope_se)
            intercept_ci = (intercept - t_critical * intercept_se, intercept + t_critical * intercept_se)
            
            # 계수 p-value
            slope_t = slope / slope_se if slope_se > 0 else 0
            intercept_t = intercept / intercept_se if intercept_se > 0 else 0
            
            slope_p = 2 * (1 - stats.t.cdf(abs(slope_t), n - 2)) if n > 2 else 1
            intercept_p = 2 * (1 - stats.t.cdf(abs(intercept_t), n - 2)) if n > 2 else 1
            
            # 회귀 진단
            diagnostics = self._perform_regression_diagnostics(
                clean_x.values.reshape(-1, 1), clean_y.values, residuals, fitted_values
            )
            
            # 해석
            interpretation = self._interpret_linear_regression(
                slope, intercept, r_squared, slope_p, x_name, y_name
            )
            
            return RegressionResult(
                regression_type=RegressionType.LINEAR,
                coefficients={
                    "intercept": float(intercept),
                    x_name: float(slope)
                },
                p_values={
                    "intercept": float(intercept_p),
                    x_name: float(slope_p)
                },
                confidence_intervals={
                    "intercept": intercept_ci,
                    x_name: slope_ci
                },
                r_squared=float(r_squared),
                adjusted_r_squared=float(adjusted_r_squared),
                f_statistic=float(f_statistic),
                f_p_value=float(f_p_value),
                residuals=residuals.values,
                fitted_values=fitted_values.values,
                interpretation=interpretation,
                diagnostics=diagnostics,
                metadata={
                    "sample_size": n,
                    "x_name": x_name,
                    "y_name": y_name,
                    "correlation": float(r_value)
                }
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'regression_type': 'simple_linear'})
            raise StatisticsException(f"단순 선형 회귀분석 실패: {error_info['message']}")
    
    def multiple_linear_regression(self, 
                                  X: pd.DataFrame,
                                  y: Union[pd.Series, np.ndarray, List],
                                  y_name: str = "Y") -> RegressionResult:
        """
        다중 선형 회귀분석
        
        Args:
            X: 독립변수들 (DataFrame)
            y: 종속변수
            y_name: 종속변수 이름
            
        Returns:
            RegressionResult: 회귀분석 결과
        """
        try:
            from scipy import stats
            
            if isinstance(y, (list, np.ndarray)):
                y = pd.Series(y)
            
            # 길이 확인
            if len(X) != len(y):
                raise StatisticsException("독립변수와 종속변수의 길이가 같아야 합니다")
            
            # 결측값 제거
            data = pd.concat([X, y], axis=1)
            clean_data = data.dropna()
            
            if len(clean_data) < X.shape[1] + 2:
                raise StatisticsException(f"다중 회귀분석을 위해서는 최소 {X.shape[1] + 2}개의 관측값이 필요합니다")
            
            clean_X = clean_data.iloc[:, :-1]
            clean_y = clean_data.iloc[:, -1]
            
            # 상수항 추가
            X_with_const = np.column_stack([np.ones(len(clean_X)), clean_X])
            
            # 회귀계수 계산 (최소제곱법)
            try:
                XtX_inv = np.linalg.inv(X_with_const.T @ X_with_const)
                coefficients = XtX_inv @ X_with_const.T @ clean_y
            except np.linalg.LinAlgError:
                raise StatisticsException("다중공선성으로 인해 회귀분석을 수행할 수 없습니다")
            
            # 예측값과 잔차
            fitted_values = X_with_const @ coefficients
            residuals = clean_y - fitted_values
            
            # R-squared
            ss_total = np.sum((clean_y - clean_y.mean()) ** 2)
            ss_residual = np.sum(residuals ** 2)
            r_squared = 1 - ss_residual / ss_total if ss_total > 0 else 0
            
            # 조정된 R-squared
            n = len(clean_y)
            p = X.shape[1]
            adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared
            
            # F 통계량
            mse_regression = (ss_total - ss_residual) / p if p > 0 else 0
            mse_residual = ss_residual / (n - p - 1) if n > p + 1 else 1
            f_statistic = mse_regression / mse_residual if mse_residual > 0 else 0
            f_p_value = 1 - stats.f.cdf(f_statistic, p, n - p - 1) if n > p + 1 else 1
            
            # 계수의 표준오차
            mse = ss_residual / (n - p - 1) if n > p + 1 else 1
            var_coef = mse * np.diag(XtX_inv)
            se_coef = np.sqrt(var_coef)
            
            # t 통계량과 p-value
            t_stats = coefficients / se_coef
            t_p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1)) if n > p + 1 else np.ones_like(t_stats)
            
            # 신뢰구간
            t_critical = stats.t.ppf(1 - self.alpha/2, n - p - 1) if n > p + 1 else 0
            ci_lower = coefficients - t_critical * se_coef
            ci_upper = coefficients + t_critical * se_coef
            
            # 결과 정리
            var_names = ["intercept"] + list(clean_X.columns)
            coef_dict = {name: float(coef) for name, coef in zip(var_names, coefficients)}
            p_dict = {name: float(p_val) for name, p_val in zip(var_names, t_p_values)}
            ci_dict = {name: (float(ci_lower[i]), float(ci_upper[i])) 
                      for i, name in enumerate(var_names)}
            
            # 회귀 진단
            diagnostics = self._perform_regression_diagnostics(
                clean_X.values, clean_y.values, residuals, fitted_values
            )
            
            # 해석
            interpretation = self._interpret_multiple_regression(
                coef_dict, p_dict, r_squared, f_p_value, y_name
            )
            
            return RegressionResult(
                regression_type=RegressionType.MULTIPLE,
                coefficients=coef_dict,
                p_values=p_dict,
                confidence_intervals=ci_dict,
                r_squared=float(r_squared),
                adjusted_r_squared=float(adjusted_r_squared),
                f_statistic=float(f_statistic),
                f_p_value=float(f_p_value),
                residuals=residuals.values,
                fitted_values=fitted_values,
                interpretation=interpretation,
                diagnostics=diagnostics,
                metadata={
                    "sample_size": n,
                    "num_predictors": p,
                    "y_name": y_name,
                    "predictor_names": list(clean_X.columns)
                }
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'regression_type': 'multiple_linear'})
            raise StatisticsException(f"다중 선형 회귀분석 실패: {error_info['message']}")
    
    def logistic_regression(self, 
                           X: Union[pd.DataFrame, pd.Series, np.ndarray],
                           y: Union[pd.Series, np.ndarray, List],
                           max_iter: int = 100) -> RegressionResult:
        """
        로지스틱 회귀분석
        
        Args:
            X: 독립변수(들)
            y: 종속변수 (이진)
            max_iter: 최대 반복 횟수
            
        Returns:
            RegressionResult: 회귀분석 결과
        """
        try:
            from scipy import stats
            from scipy.optimize import minimize
            
            if isinstance(X, pd.Series):
                X = X.to_frame()
            elif isinstance(X, np.ndarray):
                if X.ndim == 1:
                    X = pd.DataFrame(X, columns=['X'])
                else:
                    X = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(X.shape[1])])
            
            if isinstance(y, (list, np.ndarray)):
                y = pd.Series(y)
            
            # 길이 확인
            if len(X) != len(y):
                raise StatisticsException("독립변수와 종속변수의 길이가 같아야 합니다")
            
            # 결측값 제거
            data = pd.concat([X, y], axis=1)
            clean_data = data.dropna()
            
            if len(clean_data) < X.shape[1] + 2:
                raise StatisticsException(f"로지스틱 회귀분석을 위해서는 최소 {X.shape[1] + 2}개의 관측값이 필요합니다")
            
            clean_X = clean_data.iloc[:, :-1]
            clean_y = clean_data.iloc[:, -1]
            
            # 이진 변수 확인
            unique_y = clean_y.unique()
            if len(unique_y) != 2:
                raise StatisticsException("로지스틱 회귀분석은 이진 종속변수만 지원합니다")
            
            # 0, 1로 변환
            y_binary = (clean_y == unique_y[1]).astype(int)
            
            # 상수항 추가
            X_with_const = np.column_stack([np.ones(len(clean_X)), clean_X])
            
            # 로지스틱 함수
            def sigmoid(z):
                z = np.clip(z, -500, 500)  # 오버플로우 방지
                return 1 / (1 + np.exp(-z))
            
            # 로그 우도 함수
            def log_likelihood(beta, X, y):
                z = X @ beta
                p = sigmoid(z)
                p = np.clip(p, 1e-15, 1 - 1e-15)  # 로그(0) 방지
                return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            
            # 초기값
            initial_beta = np.zeros(X_with_const.shape[1])
            
            # 최적화
            result = minimize(log_likelihood, initial_beta, args=(X_with_const, y_binary), 
                            method='BFGS', options={'maxiter': max_iter})
            
            if not result.success:
                logger.warning("로지스틱 회귀분석 최적화가 수렴하지 않았습니다")
            
            coefficients = result.x
            
            # 예측 확률
            fitted_probs = sigmoid(X_with_const @ coefficients)
            fitted_values = (fitted_probs > 0.5).astype(int)
            
            # 잔차 (피어슨 잔차)
            residuals = (y_binary - fitted_probs) / np.sqrt(fitted_probs * (1 - fitted_probs))
            
            # Pseudo R-squared (McFadden's R-squared)
            null_ll = -np.sum(y_binary * np.log(y_binary.mean()) + 
                             (1 - y_binary) * np.log(1 - y_binary.mean()))
            model_ll = -log_likelihood(coefficients, X_with_const, y_binary)
            pseudo_r_squared = 1 - model_ll / null_ll if null_ll != 0 else 0
            
            # Wald 검정을 위한 표준오차 (근사치)
            # 헤시안 행렬의 역행렬로부터 계산
            try:
                W = np.diag(fitted_probs * (1 - fitted_probs))
                hessian = X_with_const.T @ W @ X_with_const
                var_coef = np.linalg.inv(hessian)
                se_coef = np.sqrt(np.diag(var_coef))
            except:
                se_coef = np.ones_like(coefficients) * 0.1  # 기본값
            
            # Wald 통계량과 p-value
            z_stats = coefficients / se_coef
            p_values = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))
            
            # 신뢰구간
            z_critical = stats.norm.ppf(1 - self.alpha/2)
            ci_lower = coefficients - z_critical * se_coef
            ci_upper = coefficients + z_critical * se_coef
            
            # 결과 정리
            var_names = ["intercept"] + list(clean_X.columns)
            coef_dict = {name: float(coef) for name, coef in zip(var_names, coefficients)}
            p_dict = {name: float(p_val) for name, p_val in zip(var_names, p_values)}
            ci_dict = {name: (float(ci_lower[i]), float(ci_upper[i])) 
                      for i, name in enumerate(var_names)}
            
            # 해석
            interpretation = self._interpret_logistic_regression(
                coef_dict, p_dict, pseudo_r_squared
            )
            
            return RegressionResult(
                regression_type=RegressionType.LOGISTIC,
                coefficients=coef_dict,
                p_values=p_dict,
                confidence_intervals=ci_dict,
                r_squared=float(pseudo_r_squared),
                adjusted_r_squared=None,
                f_statistic=None,
                f_p_value=None,
                residuals=residuals,
                fitted_values=fitted_probs,
                interpretation=interpretation,
                diagnostics={},  # 로지스틱 회귀는 별도 진단
                metadata={
                    "sample_size": len(clean_y),
                    "num_predictors": X.shape[1],
                    "convergence": result.success,
                    "iterations": result.nit,
                    "unique_y_values": list(unique_y)
                }
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'regression_type': 'logistic'})
            raise StatisticsException(f"로지스틱 회귀분석 실패: {error_info['message']}")
    
    def polynomial_regression(self, 
                             x: Union[pd.Series, np.ndarray, List],
                             y: Union[pd.Series, np.ndarray, List],
                             degree: int = 2,
                             x_name: str = "X",
                             y_name: str = "Y") -> RegressionResult:
        """
        다항 회귀분석
        
        Args:
            x: 독립변수
            y: 종속변수
            degree: 다항식 차수
            x_name: 독립변수 이름
            y_name: 종속변수 이름
            
        Returns:
            RegressionResult: 회귀분석 결과
        """
        try:
            if isinstance(x, (list, np.ndarray)):
                x = pd.Series(x)
            if isinstance(y, (list, np.ndarray)):
                y = pd.Series(y)
            
            # 길이 확인
            if len(x) != len(y):
                raise StatisticsException("독립변수와 종속변수의 길이가 같아야 합니다")
            
            # 결측값 제거 (쌍으로)
            valid_idx = ~(x.isna() | y.isna())
            clean_x = x[valid_idx]
            clean_y = y[valid_idx]
            
            if len(clean_x) < degree + 2:
                raise StatisticsException(f"{degree}차 다항 회귀분석을 위해서는 최소 {degree + 2}개의 관측값이 필요합니다")
            
            # 다항식 특성 생성
            X_poly = np.column_stack([clean_x ** i for i in range(degree + 1)])
            
            # 다중 회귀분석으로 처리
            poly_df = pd.DataFrame(X_poly, columns=[f"{x_name}^{i}" if i > 0 else "intercept" 
                                                   for i in range(degree + 1)])
            
            # 상수항 제거 (이미 포함됨)
            X_features = poly_df.iloc[:, 1:]
            
            result = self.multiple_linear_regression(X_features, clean_y, y_name)
            result.regression_type = RegressionType.POLYNOMIAL
            
            # 메타데이터 업데이트
            result.metadata.update({
                "degree": degree,
                "x_name": x_name,
                "polynomial_terms": list(X_features.columns)
            })
            
            return result
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'regression_type': 'polynomial'})
            raise StatisticsException(f"다항 회귀분석 실패: {error_info['message']}")
    
    def _perform_regression_diagnostics(self, 
                                       X: np.ndarray, 
                                       y: np.ndarray, 
                                       residuals: np.ndarray, 
                                       fitted_values: np.ndarray) -> Dict[str, Any]:
        """회귀 진단 수행"""
        try:
            from scipy import stats
            
            diagnostics = {}
            
            # 1. 정규성 검정 (잔차)
            if len(residuals) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                diagnostics['normality'] = {
                    'test': 'Shapiro-Wilk',
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'assumption_met': shapiro_p > self.alpha
                }
            else:
                diagnostics['normality'] = {'assumption_met': True}
            
            # 2. 등분산성 검정 (Breusch-Pagan 근사)
            if len(fitted_values) >= 3:
                # 잔차 제곱과 예측값의 상관관계
                corr, p_val = stats.pearsonr(fitted_values, residuals ** 2)
                diagnostics['homoscedasticity'] = {
                    'test': 'Breusch-Pagan (approximate)',
                    'correlation': float(corr),
                    'p_value': float(p_val),
                    'assumption_met': p_val > self.alpha
                }
            else:
                diagnostics['homoscedasticity'] = {'assumption_met': True}
            
            # 3. 독립성 검정 (Durbin-Watson 근사)
            if len(residuals) >= 2:
                dw_stat = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
                # 2에 가까우면 독립성 만족
                diagnostics['independence'] = {
                    'test': 'Durbin-Watson (approximate)',
                    'statistic': float(dw_stat),
                    'assumption_met': 1.5 < dw_stat < 2.5
                }
            else:
                diagnostics['independence'] = {'assumption_met': True}
            
            # 4. 이상치 탐지 (표준화 잔차)
            if len(residuals) > 0:
                std_residuals = residuals / np.std(residuals)
                outliers = np.where(np.abs(std_residuals) > 2.5)[0].tolist()
                diagnostics['outliers'] = {
                    'indices': outliers,
                    'count': len(outliers),
                    'threshold': 2.5
                }
            else:
                diagnostics['outliers'] = {'indices': [], 'count': 0}
            
            # 5. 다중공선성 (VIF 근사) - 다중회귀인 경우
            if X.shape[1] > 1:
                vif_values = {}
                for i in range(X.shape[1]):
                    # 각 변수를 다른 변수들로 회귀
                    X_i = X[:, i]
                    X_others = np.delete(X, i, axis=1)
                    
                    if X_others.shape[1] > 0:
                        try:
                            # 단순 상관계수로 근사
                            corr_matrix = np.corrcoef(X.T)
                            r_squared = corr_matrix[i, :] ** 2
                            r_squared = np.delete(r_squared, i)
                            max_r_squared = np.max(r_squared) if len(r_squared) > 0 else 0
                            vif = 1 / (1 - max_r_squared) if max_r_squared < 0.99 else 10
                        except:
                            vif = 1
                    else:
                        vif = 1
                    
                    vif_values[f'X{i+1}'] = float(vif)
                
                diagnostics['multicollinearity'] = {
                    'vif_values': vif_values,
                    'max_vif': max(vif_values.values()) if vif_values else 1,
                    'assumption_met': max(vif_values.values()) < 5 if vif_values else True
                }
            else:
                diagnostics['multicollinearity'] = {'assumption_met': True}
            
            return diagnostics
            
        except Exception as e:
            logger.warning(f"회귀 진단 중 오류 발생: {e}")
            return {'error': str(e)}
    
    def _interpret_linear_regression(self, slope: float, intercept: float, 
                                   r_squared: float, slope_p: float,
                                   x_name: str, y_name: str) -> str:
        """선형 회귀분석 해석"""
        interpretation = []
        
        # 회귀식
        interpretation.append(f"회귀식: {y_name} = {intercept:.3f} + {slope:.3f} × {x_name}")
        
        # 기울기 유의성
        if slope_p < self.alpha:
            direction = "양의" if slope > 0 else "음의"
            interpretation.append(f"{x_name}이 {y_name}에 미치는 {direction} 영향이 유의합니다 (p = {slope_p:.3f})")
        else:
            interpretation.append(f"{x_name}이 {y_name}에 미치는 영향이 유의하지 않습니다 (p = {slope_p:.3f})")
        
        # 설명력
        interpretation.append(f"모델의 설명력(R²)은 {r_squared:.3f}로, {y_name} 변동의 {r_squared*100:.1f}%를 설명합니다")
        
        return " ".join(interpretation)
    
    def _interpret_multiple_regression(self, coefficients: Dict[str, float], 
                                     p_values: Dict[str, float],
                                     r_squared: float, f_p_value: float,
                                     y_name: str) -> str:
        """다중 회귀분석 해석"""
        interpretation = []
        
        # 모델 전체 유의성
        if f_p_value < self.alpha:
            interpretation.append(f"회귀모델이 전체적으로 유의합니다 (F p-value = {f_p_value:.3f})")
        else:
            interpretation.append(f"회귀모델이 전체적으로 유의하지 않습니다 (F p-value = {f_p_value:.3f})")
        
        # 개별 계수 유의성
        significant_vars = []
        for var, p_val in p_values.items():
            if var != "intercept" and p_val < self.alpha:
                coef = coefficients[var]
                direction = "양의" if coef > 0 else "음의"
                significant_vars.append(f"{var}({direction}, p={p_val:.3f})")
        
        if significant_vars:
            interpretation.append(f"유의한 예측변수: {', '.join(significant_vars)}")
        else:
            interpretation.append("유의한 예측변수가 없습니다")
        
        # 설명력
        interpretation.append(f"모델의 설명력(R²)은 {r_squared:.3f}입니다")
        
        return " ".join(interpretation)
    
    def _interpret_logistic_regression(self, coefficients: Dict[str, float], 
                                     p_values: Dict[str, float],
                                     pseudo_r_squared: float) -> str:
        """로지스틱 회귀분석 해석"""
        interpretation = []
        
        # 개별 계수 해석 (오즈비)
        significant_vars = []
        for var, p_val in p_values.items():
            if var != "intercept" and p_val < self.alpha:
                coef = coefficients[var]
                odds_ratio = np.exp(coef)
                if odds_ratio > 1:
                    effect = f"확률을 {odds_ratio:.2f}배 증가"
                else:
                    effect = f"확률을 {1/odds_ratio:.2f}배 감소"
                significant_vars.append(f"{var}({effect}, p={p_val:.3f})")
        
        if significant_vars:
            interpretation.append(f"유의한 예측변수: {', '.join(significant_vars)}")
        else:
            interpretation.append("유의한 예측변수가 없습니다")
        
        # 모델 적합도
        interpretation.append(f"Pseudo R²는 {pseudo_r_squared:.3f}입니다")
        
        return " ".join(interpretation)
    
    def predict(self, result: RegressionResult, 
               X_new: Union[pd.DataFrame, pd.Series, np.ndarray, List]) -> np.ndarray:
        """새로운 데이터에 대한 예측"""
        try:
            if result.regression_type == RegressionType.LINEAR:
                # 단순 선형 회귀
                if isinstance(X_new, (list, np.ndarray)):
                    X_new = np.array(X_new)
                
                intercept = result.coefficients['intercept']
                slope = list(result.coefficients.values())[1]  # 첫 번째 변수의 계수
                
                return intercept + slope * X_new
                
            elif result.regression_type == RegressionType.MULTIPLE:
                # 다중 회귀
                if isinstance(X_new, (list, np.ndarray)):
                    if np.array(X_new).ndim == 1:
                        X_new = np.array(X_new).reshape(1, -1)
                    X_new = pd.DataFrame(X_new)
                
                intercept = result.coefficients['intercept']
                predictions = np.full(len(X_new), intercept)
                
                for var, coef in result.coefficients.items():
                    if var != 'intercept':
                        if var in X_new.columns:
                            predictions += coef * X_new[var]
                
                return predictions
                
            else:
                raise StatisticsException(f"예측이 지원되지 않는 회귀 타입: {result.regression_type}")
                
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'operation': 'prediction'})
            raise StatisticsException(f"예측 실패: {error_info['message']}")
    
    def calculate_model_metrics(self, result: RegressionResult) -> Dict[str, Any]:
        """모델 평가 지표 계산"""
        try:
            metrics = {}
            
            if result.regression_type in [RegressionType.LINEAR, RegressionType.MULTIPLE, RegressionType.POLYNOMIAL]:
                # 선형 회귀 지표
                residuals = result.residuals
                fitted_values = result.fitted_values
                
                # RMSE
                rmse = np.sqrt(np.mean(residuals ** 2))
                metrics['rmse'] = float(rmse)
                
                # MAE
                mae = np.mean(np.abs(residuals))
                metrics['mae'] = float(mae)
                
                # MAPE (평균절대백분율오차)
                actual_values = fitted_values + residuals
                non_zero_actual = actual_values[actual_values != 0]
                if len(non_zero_actual) > 0:
                    mape = np.mean(np.abs(residuals[actual_values != 0] / non_zero_actual)) * 100
                    metrics['mape'] = float(mape)
                
                # AIC (근사치)
                n = len(residuals)
                k = len(result.coefficients)
                aic = n * np.log(np.mean(residuals ** 2)) + 2 * k
                metrics['aic'] = float(aic)
                
                # BIC (근사치)
                bic = n * np.log(np.mean(residuals ** 2)) + k * np.log(n)
                metrics['bic'] = float(bic)
                
            elif result.regression_type == RegressionType.LOGISTIC:
                # 로지스틱 회귀 지표
                fitted_probs = result.fitted_values
                
                # 분류 정확도 (0.5 임계값)
                predicted_classes = (fitted_probs > 0.5).astype(int)
                # 실제 클래스는 잔차로부터 역산 (근사치)
                actual_classes = (fitted_probs + result.residuals > 0.5).astype(int)
                
                accuracy = np.mean(predicted_classes == actual_classes)
                metrics['accuracy'] = float(accuracy)
                
                # 로그 손실 (근사치)
                log_loss = -np.mean(actual_classes * np.log(fitted_probs + 1e-15) + 
                                  (1 - actual_classes) * np.log(1 - fitted_probs + 1e-15))
                metrics['log_loss'] = float(log_loss)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"모델 지표 계산 중 오류 발생: {e}")
            return {'error': str(e)} 