"""
Non-parametric Tests

비모수적 통계 검정 모듈
- Mann-Whitney U 검정
- Wilcoxon 부호순위 검정
- Kruskal-Wallis 검정
- Friedman 검정
- 순위 상관분석 (Spearman, Kendall)
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

class NonParametricTestType(Enum):
    """비모수 검정 타입"""
    MANN_WHITNEY_U = "mann_whitney_u"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    KRUSKAL_WALLIS = "kruskal_wallis"
    FRIEDMAN = "friedman"
    SPEARMAN_CORRELATION = "spearman_correlation"
    KENDALL_TAU = "kendall_tau"
    SIGN_TEST = "sign_test"
    RUNS_TEST = "runs_test"

@dataclass
class NonParametricTestResult:
    """비모수 검정 결과"""
    test_type: NonParametricTestType
    statistic: float
    p_value: float
    effect_size: Optional[float]
    is_significant: bool
    interpretation: str
    metadata: Dict[str, Any]

class NonParametricTests:
    """비모수적 검정 메인 클래스"""
    
    def __init__(self, alpha: float = 0.05):
        """
        비모수적 검정 클래스 초기화
        
        Args:
            alpha: 유의수준 (기본값: 0.05)
        """
        self.alpha = alpha
        self.error_handler = ErrorHandler()
        logger.info("비모수적 검정 클래스 초기화 완료")
    
    def mann_whitney_u_test(self, 
                           group1: Union[pd.Series, np.ndarray, List],
                           group2: Union[pd.Series, np.ndarray, List],
                           alternative: str = "two-sided") -> NonParametricTestResult:
        """
        Mann-Whitney U 검정 (독립표본)
        
        Args:
            group1: 첫 번째 그룹 데이터
            group2: 두 번째 그룹 데이터
            alternative: 대립가설 ("two-sided", "greater", "less")
            
        Returns:
            NonParametricTestResult: 검정 결과
        """
        try:
            from scipy import stats
            
            if isinstance(group1, (list, np.ndarray)):
                group1 = pd.Series(group1)
            if isinstance(group2, (list, np.ndarray)):
                group2 = pd.Series(group2)
            
            # 결측값 제거
            clean_group1 = group1.dropna()
            clean_group2 = group2.dropna()
            
            if len(clean_group1) < 1 or len(clean_group2) < 1:
                raise StatisticsException("각 그룹은 최소 1개의 관측값이 필요합니다")
            
            # Mann-Whitney U 검정 수행
            u_stat, p_value = stats.mannwhitneyu(
                clean_group1, clean_group2, 
                alternative=alternative
            )
            
            # 효과 크기 (r = Z / sqrt(N))
            n1, n2 = len(clean_group1), len(clean_group2)
            n_total = n1 + n2
            
            # Z 점수 계산 (근사치)
            u_expected = n1 * n2 / 2
            u_variance = n1 * n2 * (n1 + n2 + 1) / 12
            z_score = (u_stat - u_expected) / np.sqrt(u_variance) if u_variance > 0 else 0
            effect_size = abs(z_score) / np.sqrt(n_total)
            
            # 해석
            interpretation = self._interpret_mann_whitney(
                u_stat, p_value, clean_group1.median(), clean_group2.median(), alternative
            )
            
            return NonParametricTestResult(
                test_type=NonParametricTestType.MANN_WHITNEY_U,
                statistic=float(u_stat),
                p_value=float(p_value),
                effect_size=float(effect_size),
                is_significant=p_value < self.alpha,
                interpretation=interpretation,
                metadata={
                    "group1_size": n1,
                    "group2_size": n2,
                    "group1_median": float(clean_group1.median()),
                    "group2_median": float(clean_group2.median()),
                    "alternative": alternative,
                    "z_score": float(z_score)
                }
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'test_type': 'mann_whitney_u'})
            raise StatisticsException(f"Mann-Whitney U 검정 실패: {error_info['message']}")
    
    def wilcoxon_signed_rank_test(self, 
                                 before: Union[pd.Series, np.ndarray, List],
                                 after: Union[pd.Series, np.ndarray, List],
                                 alternative: str = "two-sided") -> NonParametricTestResult:
        """
        Wilcoxon 부호순위 검정 (대응표본)
        
        Args:
            before: 처치 전 데이터
            after: 처치 후 데이터
            alternative: 대립가설 ("two-sided", "greater", "less")
            
        Returns:
            NonParametricTestResult: 검정 결과
        """
        try:
            from scipy import stats
            
            if isinstance(before, (list, np.ndarray)):
                before = pd.Series(before)
            if isinstance(after, (list, np.ndarray)):
                after = pd.Series(after)
            
            # 길이 확인
            if len(before) != len(after):
                raise StatisticsException("Wilcoxon 부호순위 검정을 위해서는 두 그룹의 크기가 같아야 합니다")
            
            # 결측값 제거 (쌍으로)
            valid_idx = ~(before.isna() | after.isna())
            clean_before = before[valid_idx]
            clean_after = after[valid_idx]
            
            if len(clean_before) < 1:
                raise StatisticsException("Wilcoxon 부호순위 검정을 위해서는 최소 1쌍의 관측값이 필요합니다")
            
            # Wilcoxon 부호순위 검정 수행
            w_stat, p_value = stats.wilcoxon(
                clean_before, clean_after, 
                alternative=alternative
            )
            
            # 차이 계산
            differences = clean_after - clean_before
            
            # 효과 크기 (r = Z / sqrt(N))
            n = len(differences)
            
            # Z 점수 계산 (근사치)
            w_expected = n * (n + 1) / 4
            w_variance = n * (n + 1) * (2 * n + 1) / 24
            z_score = (w_stat - w_expected) / np.sqrt(w_variance) if w_variance > 0 else 0
            effect_size = abs(z_score) / np.sqrt(n)
            
            # 해석
            interpretation = self._interpret_wilcoxon(
                w_stat, p_value, differences.median(), alternative
            )
            
            return NonParametricTestResult(
                test_type=NonParametricTestType.WILCOXON_SIGNED_RANK,
                statistic=float(w_stat),
                p_value=float(p_value),
                effect_size=float(effect_size),
                is_significant=p_value < self.alpha,
                interpretation=interpretation,
                metadata={
                    "sample_size": n,
                    "median_difference": float(differences.median()),
                    "before_median": float(clean_before.median()),
                    "after_median": float(clean_after.median()),
                    "alternative": alternative,
                    "z_score": float(z_score)
                }
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'test_type': 'wilcoxon_signed_rank'})
            raise StatisticsException(f"Wilcoxon 부호순위 검정 실패: {error_info['message']}")
    
    def kruskal_wallis_test(self, 
                           *groups: Union[pd.Series, np.ndarray, List],
                           group_names: Optional[List[str]] = None) -> NonParametricTestResult:
        """
        Kruskal-Wallis 검정 (다중 독립표본)
        
        Args:
            *groups: 비교할 그룹들
            group_names: 그룹 이름들
            
        Returns:
            NonParametricTestResult: 검정 결과
        """
        try:
            from scipy import stats
            
            if len(groups) < 2:
                raise StatisticsException("Kruskal-Wallis 검정을 위해서는 최소 2개 그룹이 필요합니다")
            
            # 데이터 정리
            clean_groups = []
            for i, group in enumerate(groups):
                if isinstance(group, (list, np.ndarray)):
                    group = pd.Series(group)
                clean_group = group.dropna()
                if len(clean_group) < 1:
                    raise StatisticsException(f"그룹 {i+1}은 최소 1개의 관측값이 필요합니다")
                clean_groups.append(clean_group)
            
            # Kruskal-Wallis 검정 수행
            h_stat, p_value = stats.kruskal(*clean_groups)
            
            # 효과 크기 (Eta-squared for Kruskal-Wallis)
            n_total = sum(len(group) for group in clean_groups)
            k = len(clean_groups)
            eta_squared = (h_stat - k + 1) / (n_total - k) if (n_total - k) > 0 else 0
            
            # 해석
            interpretation = self._interpret_kruskal_wallis(h_stat, p_value, k)
            
            # 그룹 정보
            if group_names is None:
                group_names = [f"Group_{i+1}" for i in range(len(clean_groups))]
            
            group_stats = {}
            for i, (name, group) in enumerate(zip(group_names, clean_groups)):
                group_stats[name] = {
                    "median": float(group.median()),
                    "size": len(group),
                    "mean_rank": float(stats.rankdata(np.concatenate(clean_groups))[
                        sum(len(clean_groups[j]) for j in range(i)):
                        sum(len(clean_groups[j]) for j in range(i+1))
                    ].mean())
                }
            
            return NonParametricTestResult(
                test_type=NonParametricTestType.KRUSKAL_WALLIS,
                statistic=float(h_stat),
                p_value=float(p_value),
                effect_size=float(eta_squared),
                is_significant=p_value < self.alpha,
                interpretation=interpretation,
                metadata={
                    "num_groups": k,
                    "total_sample_size": n_total,
                    "group_stats": group_stats
                }
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'test_type': 'kruskal_wallis'})
            raise StatisticsException(f"Kruskal-Wallis 검정 실패: {error_info['message']}")
    
    def friedman_test(self, 
                     data: pd.DataFrame,
                     subject_col: str,
                     condition_col: str,
                     value_col: str) -> NonParametricTestResult:
        """
        Friedman 검정 (반복측정 비모수 검정)
        
        Args:
            data: 데이터프레임
            subject_col: 피험자 컬럼명
            condition_col: 조건 컬럼명
            value_col: 값 컬럼명
            
        Returns:
            NonParametricTestResult: 검정 결과
        """
        try:
            from scipy import stats
            
            # 결측값 제거
            clean_data = data[[subject_col, condition_col, value_col]].dropna()
            
            if len(clean_data) < 3:
                raise StatisticsException("Friedman 검정을 위해서는 최소 3개의 관측값이 필요합니다")
            
            # 피벗 테이블 생성
            pivot_data = clean_data.pivot(
                index=subject_col, 
                columns=condition_col, 
                values=value_col
            )
            
            # 결측값이 있는 행 제거
            pivot_data = pivot_data.dropna()
            
            if len(pivot_data) < 2:
                raise StatisticsException("Friedman 검정을 위해서는 최소 2명의 피험자가 필요합니다")
            
            if pivot_data.shape[1] < 2:
                raise StatisticsException("Friedman 검정을 위해서는 최소 2개의 조건이 필요합니다")
            
            # Friedman 검정 수행
            chi2_stat, p_value = stats.friedmanchisquare(*[pivot_data[col] for col in pivot_data.columns])
            
            # 효과 크기 (Kendall's W)
            n = len(pivot_data)  # 피험자 수
            k = len(pivot_data.columns)  # 조건 수
            
            # 순위 계산
            ranks = pivot_data.rank(axis=1)
            rank_sums = ranks.sum(axis=0)
            
            # Kendall's W 계산
            ss_total = ((rank_sums - rank_sums.mean()) ** 2).sum()
            w = 12 * ss_total / (n**2 * (k**3 - k))
            
            # 해석
            interpretation = self._interpret_friedman(chi2_stat, p_value, k)
            
            # 조건별 통계
            condition_stats = {}
            for col in pivot_data.columns:
                condition_stats[str(col)] = {
                    "median": float(pivot_data[col].median()),
                    "mean_rank": float(ranks[col].mean())
                }
            
            return NonParametricTestResult(
                test_type=NonParametricTestType.FRIEDMAN,
                statistic=float(chi2_stat),
                p_value=float(p_value),
                effect_size=float(w),
                is_significant=p_value < self.alpha,
                interpretation=interpretation,
                metadata={
                    "num_subjects": n,
                    "num_conditions": k,
                    "kendalls_w": float(w),
                    "condition_stats": condition_stats
                }
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'test_type': 'friedman'})
            raise StatisticsException(f"Friedman 검정 실패: {error_info['message']}")
    
    def spearman_correlation(self, 
                           x: Union[pd.Series, np.ndarray, List],
                           y: Union[pd.Series, np.ndarray, List]) -> NonParametricTestResult:
        """
        Spearman 순위 상관분석
        
        Args:
            x: 첫 번째 변수
            y: 두 번째 변수
            
        Returns:
            NonParametricTestResult: 검정 결과
        """
        try:
            from scipy import stats
            
            if isinstance(x, (list, np.ndarray)):
                x = pd.Series(x)
            if isinstance(y, (list, np.ndarray)):
                y = pd.Series(y)
            
            # 길이 확인
            if len(x) != len(y):
                raise StatisticsException("Spearman 상관분석을 위해서는 두 변수의 길이가 같아야 합니다")
            
            # 결측값 제거 (쌍으로)
            valid_idx = ~(x.isna() | y.isna())
            clean_x = x[valid_idx]
            clean_y = y[valid_idx]
            
            if len(clean_x) < 3:
                raise StatisticsException("Spearman 상관분석을 위해서는 최소 3쌍의 관측값이 필요합니다")
            
            # Spearman 상관분석 수행
            rho, p_value = stats.spearmanr(clean_x, clean_y)
            
            # 효과 크기는 상관계수 자체
            effect_size = abs(rho)
            
            # 해석
            interpretation = self._interpret_correlation(rho, p_value, "Spearman")
            
            return NonParametricTestResult(
                test_type=NonParametricTestType.SPEARMAN_CORRELATION,
                statistic=float(rho),
                p_value=float(p_value),
                effect_size=float(effect_size),
                is_significant=p_value < self.alpha,
                interpretation=interpretation,
                metadata={
                    "sample_size": len(clean_x),
                    "correlation_coefficient": float(rho),
                    "correlation_strength": self._interpret_correlation_strength(abs(rho))
                }
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'test_type': 'spearman_correlation'})
            raise StatisticsException(f"Spearman 상관분석 실패: {error_info['message']}")
    
    def kendall_tau_correlation(self, 
                               x: Union[pd.Series, np.ndarray, List],
                               y: Union[pd.Series, np.ndarray, List]) -> NonParametricTestResult:
        """
        Kendall's Tau 상관분석
        
        Args:
            x: 첫 번째 변수
            y: 두 번째 변수
            
        Returns:
            NonParametricTestResult: 검정 결과
        """
        try:
            from scipy import stats
            
            if isinstance(x, (list, np.ndarray)):
                x = pd.Series(x)
            if isinstance(y, (list, np.ndarray)):
                y = pd.Series(y)
            
            # 길이 확인
            if len(x) != len(y):
                raise StatisticsException("Kendall's Tau 상관분석을 위해서는 두 변수의 길이가 같아야 합니다")
            
            # 결측값 제거 (쌍으로)
            valid_idx = ~(x.isna() | y.isna())
            clean_x = x[valid_idx]
            clean_y = y[valid_idx]
            
            if len(clean_x) < 3:
                raise StatisticsException("Kendall's Tau 상관분석을 위해서는 최소 3쌍의 관측값이 필요합니다")
            
            # Kendall's Tau 상관분석 수행
            tau, p_value = stats.kendalltau(clean_x, clean_y)
            
            # 효과 크기는 상관계수 자체
            effect_size = abs(tau)
            
            # 해석
            interpretation = self._interpret_correlation(tau, p_value, "Kendall's Tau")
            
            return NonParametricTestResult(
                test_type=NonParametricTestType.KENDALL_TAU,
                statistic=float(tau),
                p_value=float(p_value),
                effect_size=float(effect_size),
                is_significant=p_value < self.alpha,
                interpretation=interpretation,
                metadata={
                    "sample_size": len(clean_x),
                    "correlation_coefficient": float(tau),
                    "correlation_strength": self._interpret_correlation_strength(abs(tau))
                }
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'test_type': 'kendall_tau'})
            raise StatisticsException(f"Kendall's Tau 상관분석 실패: {error_info['message']}")
    
    def sign_test(self, 
                 before: Union[pd.Series, np.ndarray, List],
                 after: Union[pd.Series, np.ndarray, List],
                 alternative: str = "two-sided") -> NonParametricTestResult:
        """
        부호 검정 (Sign Test)
        
        Args:
            before: 처치 전 데이터
            after: 처치 후 데이터
            alternative: 대립가설 ("two-sided", "greater", "less")
            
        Returns:
            NonParametricTestResult: 검정 결과
        """
        try:
            from scipy import stats
            
            if isinstance(before, (list, np.ndarray)):
                before = pd.Series(before)
            if isinstance(after, (list, np.ndarray)):
                after = pd.Series(after)
            
            # 길이 확인
            if len(before) != len(after):
                raise StatisticsException("부호 검정을 위해서는 두 그룹의 크기가 같아야 합니다")
            
            # 결측값 제거 (쌍으로)
            valid_idx = ~(before.isna() | after.isna())
            clean_before = before[valid_idx]
            clean_after = after[valid_idx]
            
            if len(clean_before) < 1:
                raise StatisticsException("부호 검정을 위해서는 최소 1쌍의 관측값이 필요합니다")
            
            # 차이 계산
            differences = clean_after - clean_before
            
            # 0인 차이 제거
            non_zero_diff = differences[differences != 0]
            
            if len(non_zero_diff) == 0:
                raise StatisticsException("모든 차이가 0입니다. 부호 검정을 수행할 수 없습니다")
            
            # 양수 개수
            n_positive = (non_zero_diff > 0).sum()
            n_total = len(non_zero_diff)
            
            # 이항검정 수행
            if alternative == "two-sided":
                p_value = 2 * min(
                    stats.binom.cdf(n_positive, n_total, 0.5),
                    1 - stats.binom.cdf(n_positive - 1, n_total, 0.5)
                )
            elif alternative == "greater":
                p_value = 1 - stats.binom.cdf(n_positive - 1, n_total, 0.5)
            else:  # less
                p_value = stats.binom.cdf(n_positive, n_total, 0.5)
            
            # 검정통계량 (양수 개수)
            statistic = n_positive
            
            # 효과 크기 (비율 차이)
            effect_size = abs(n_positive / n_total - 0.5) * 2
            
            # 해석
            interpretation = self._interpret_sign_test(
                n_positive, n_total, p_value, alternative
            )
            
            return NonParametricTestResult(
                test_type=NonParametricTestType.SIGN_TEST,
                statistic=float(statistic),
                p_value=float(p_value),
                effect_size=float(effect_size),
                is_significant=p_value < self.alpha,
                interpretation=interpretation,
                metadata={
                    "sample_size": n_total,
                    "n_positive": int(n_positive),
                    "n_negative": int(n_total - n_positive),
                    "proportion_positive": float(n_positive / n_total),
                    "alternative": alternative
                }
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'test_type': 'sign_test'})
            raise StatisticsException(f"부호 검정 실패: {error_info['message']}")
    
    def runs_test(self, 
                 data: Union[pd.Series, np.ndarray, List],
                 cutoff: Optional[float] = None) -> NonParametricTestResult:
        """
        연속성 검정 (Runs Test)
        
        Args:
            data: 검정할 데이터
            cutoff: 기준값 (기본값: 중앙값)
            
        Returns:
            NonParametricTestResult: 검정 결과
        """
        try:
            from scipy import stats
            
            if isinstance(data, (list, np.ndarray)):
                data = pd.Series(data)
            
            # 결측값 제거
            clean_data = data.dropna()
            
            if len(clean_data) < 2:
                raise StatisticsException("연속성 검정을 위해서는 최소 2개의 관측값이 필요합니다")
            
            # 기준값 설정
            if cutoff is None:
                cutoff = clean_data.median()
            
            # 이진 시퀀스 생성 (기준값보다 큰지 여부)
            binary_seq = (clean_data > cutoff).astype(int)
            
            # 연속 개수 계산
            runs = []
            current_run = 1
            
            for i in range(1, len(binary_seq)):
                if binary_seq.iloc[i] == binary_seq.iloc[i-1]:
                    current_run += 1
                else:
                    runs.append(current_run)
                    current_run = 1
            runs.append(current_run)
            
            n_runs = len(runs)
            n_total = len(binary_seq)
            n_positive = binary_seq.sum()
            n_negative = n_total - n_positive
            
            # 기대 연속 개수와 분산
            if n_positive > 0 and n_negative > 0:
                expected_runs = (2 * n_positive * n_negative) / n_total + 1
                variance_runs = (2 * n_positive * n_negative * (2 * n_positive * n_negative - n_total)) / \
                               (n_total**2 * (n_total - 1))
                
                # Z 점수 계산
                if variance_runs > 0:
                    z_score = (n_runs - expected_runs) / np.sqrt(variance_runs)
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                else:
                    z_score = 0
                    p_value = 1.0
            else:
                z_score = 0
                p_value = 1.0
            
            # 효과 크기
            effect_size = abs(z_score) / np.sqrt(n_total) if n_total > 0 else 0
            
            # 해석
            interpretation = self._interpret_runs_test(n_runs, expected_runs, p_value)
            
            return NonParametricTestResult(
                test_type=NonParametricTestType.RUNS_TEST,
                statistic=float(n_runs),
                p_value=float(p_value),
                effect_size=float(effect_size),
                is_significant=p_value < self.alpha,
                interpretation=interpretation,
                metadata={
                    "sample_size": n_total,
                    "n_runs": int(n_runs),
                    "expected_runs": float(expected_runs),
                    "n_positive": int(n_positive),
                    "n_negative": int(n_negative),
                    "cutoff": float(cutoff),
                    "z_score": float(z_score)
                }
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'test_type': 'runs_test'})
            raise StatisticsException(f"연속성 검정 실패: {error_info['message']}")
    
    def _interpret_mann_whitney(self, u_stat: float, p_value: float,
                               median1: float, median2: float, alternative: str) -> str:
        """Mann-Whitney U 검정 해석"""
        if p_value < self.alpha:
            if alternative == "two-sided":
                return f"두 그룹 간 분포에 유의한 차이가 있습니다 (중앙값1: {median1:.3f}, 중앙값2: {median2:.3f})."
            elif alternative == "greater":
                return f"그룹1이 그룹2보다 유의하게 큽니다 (중앙값1: {median1:.3f}, 중앙값2: {median2:.3f})."
            else:
                return f"그룹1이 그룹2보다 유의하게 작습니다 (중앙값1: {median1:.3f}, 중앙값2: {median2:.3f})."
        else:
            return f"두 그룹 간 분포에 유의한 차이가 없습니다 (중앙값1: {median1:.3f}, 중앙값2: {median2:.3f})."
    
    def _interpret_wilcoxon(self, w_stat: float, p_value: float,
                           median_diff: float, alternative: str) -> str:
        """Wilcoxon 부호순위 검정 해석"""
        if p_value < self.alpha:
            if alternative == "two-sided":
                return f"처치 전후 간에 유의한 차이가 있습니다 (중앙값 차이: {median_diff:.3f})."
            elif alternative == "greater":
                return f"처치 후가 처치 전보다 유의하게 큽니다 (중앙값 차이: {median_diff:.3f})."
            else:
                return f"처치 후가 처치 전보다 유의하게 작습니다 (중앙값 차이: {median_diff:.3f})."
        else:
            return f"처치 전후 간에 유의한 차이가 없습니다 (중앙값 차이: {median_diff:.3f})."
    
    def _interpret_kruskal_wallis(self, h_stat: float, p_value: float, num_groups: int) -> str:
        """Kruskal-Wallis 검정 해석"""
        if p_value < self.alpha:
            return f"{num_groups}개 그룹 간에 유의한 차이가 있습니다 (H = {h_stat:.3f}, p = {p_value:.3f})."
        else:
            return f"{num_groups}개 그룹 간에 유의한 차이가 없습니다 (H = {h_stat:.3f}, p = {p_value:.3f})."
    
    def _interpret_friedman(self, chi2_stat: float, p_value: float, num_conditions: int) -> str:
        """Friedman 검정 해석"""
        if p_value < self.alpha:
            return f"{num_conditions}개 조건 간에 유의한 차이가 있습니다 (χ² = {chi2_stat:.3f}, p = {p_value:.3f})."
        else:
            return f"{num_conditions}개 조건 간에 유의한 차이가 없습니다 (χ² = {chi2_stat:.3f}, p = {p_value:.3f})."
    
    def _interpret_correlation(self, correlation: float, p_value: float, method: str) -> str:
        """상관분석 해석"""
        strength = self._interpret_correlation_strength(abs(correlation))
        direction = "양의" if correlation > 0 else "음의"
        
        if p_value < self.alpha:
            return f"{method} 상관계수는 {correlation:.3f}로, {direction} {strength} 상관관계가 유의합니다."
        else:
            return f"{method} 상관계수는 {correlation:.3f}로, 유의한 상관관계가 없습니다."
    
    def _interpret_correlation_strength(self, abs_correlation: float) -> str:
        """상관계수 강도 해석"""
        if abs_correlation < 0.1:
            return "매우 약한"
        elif abs_correlation < 0.3:
            return "약한"
        elif abs_correlation < 0.5:
            return "중간"
        elif abs_correlation < 0.7:
            return "강한"
        else:
            return "매우 강한"
    
    def _interpret_sign_test(self, n_positive: int, n_total: int, 
                           p_value: float, alternative: str) -> str:
        """부호 검정 해석"""
        proportion = n_positive / n_total
        
        if p_value < self.alpha:
            if alternative == "two-sided":
                return f"양수와 음수의 비율에 유의한 차이가 있습니다 (양수 비율: {proportion:.3f})."
            elif alternative == "greater":
                return f"양수가 음수보다 유의하게 많습니다 (양수 비율: {proportion:.3f})."
            else:
                return f"음수가 양수보다 유의하게 많습니다 (양수 비율: {proportion:.3f})."
        else:
            return f"양수와 음수의 비율에 유의한 차이가 없습니다 (양수 비율: {proportion:.3f})."
    
    def _interpret_runs_test(self, n_runs: int, expected_runs: float, p_value: float) -> str:
        """연속성 검정 해석"""
        if p_value < self.alpha:
            if n_runs < expected_runs:
                return f"데이터에 유의한 군집성이 있습니다 (관찰된 연속: {n_runs}, 기대 연속: {expected_runs:.1f})."
            else:
                return f"데이터에 유의한 교대성이 있습니다 (관찰된 연속: {n_runs}, 기대 연속: {expected_runs:.1f})."
        else:
            return f"데이터는 무작위적입니다 (관찰된 연속: {n_runs}, 기대 연속: {expected_runs:.1f})."
    
    def calculate_effect_size_interpretation(self, test_result: NonParametricTestResult) -> Dict[str, Any]:
        """효과 크기 해석"""
        effect_size = test_result.effect_size
        
        if test_result.test_type in [NonParametricTestType.MANN_WHITNEY_U, 
                                   NonParametricTestType.WILCOXON_SIGNED_RANK]:
            # r = Z/sqrt(N) 해석
            if effect_size < 0.1:
                magnitude = "작음"
            elif effect_size < 0.3:
                magnitude = "중간"
            elif effect_size < 0.5:
                magnitude = "큼"
            else:
                magnitude = "매우 큼"
        
        elif test_result.test_type == NonParametricTestType.KRUSKAL_WALLIS:
            # Eta-squared 해석
            if effect_size < 0.01:
                magnitude = "작음"
            elif effect_size < 0.06:
                magnitude = "중간"
            elif effect_size < 0.14:
                magnitude = "큼"
            else:
                magnitude = "매우 큼"
        
        elif test_result.test_type in [NonParametricTestType.SPEARMAN_CORRELATION,
                                     NonParametricTestType.KENDALL_TAU]:
            # 상관계수 해석
            magnitude = self._interpret_correlation_strength(effect_size)
        
        else:
            magnitude = "알 수 없음"
        
        return {
            "effect_size": effect_size,
            "magnitude": magnitude,
            "interpretation": f"효과 크기는 {magnitude}입니다 ({effect_size:.3f})"
        } 