"""
Parametric Tests

모수적 통계 검정 모듈
- t-검정 (독립표본, 대응표본, 일표본)
- 분산분석 (일원, 이원 ANOVA)
- 사후 검정 (Tukey HSD, Bonferroni 등)
- 효과 크기 계산
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

class TestType(Enum):
    """검정 타입"""
    ONE_SAMPLE_T = "one_sample_t"
    INDEPENDENT_T = "independent_t"
    PAIRED_T = "paired_t"
    ONE_WAY_ANOVA = "one_way_anova"
    TWO_WAY_ANOVA = "two_way_anova"
    REPEATED_MEASURES_ANOVA = "repeated_measures_anova"

class PostHocMethod(Enum):
    """사후 검정 방법"""
    TUKEY = "tukey"
    BONFERRONI = "bonferroni"
    SCHEFFE = "scheffe"
    DUNCAN = "duncan"

@dataclass
class TestResult:
    """검정 결과"""
    test_type: TestType
    statistic: float
    p_value: float
    degrees_of_freedom: Union[int, Tuple[int, int]]
    effect_size: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    is_significant: bool
    interpretation: str
    assumptions_met: Dict[str, bool]
    metadata: Dict[str, Any]

@dataclass
class PostHocResult:
    """사후 검정 결과"""
    method: PostHocMethod
    comparisons: List[Dict[str, Any]]
    overall_significant: bool
    metadata: Dict[str, Any]

class ParametricTests:
    """모수적 검정 메인 클래스"""
    
    def __init__(self, alpha: float = 0.05):
        """
        모수적 검정 클래스 초기화
        
        Args:
            alpha: 유의수준 (기본값: 0.05)
        """
        self.alpha = alpha
        self.error_handler = ErrorHandler()
        logger.info("모수적 검정 클래스 초기화 완료")
    
    def one_sample_t_test(self, 
                         data: Union[pd.Series, np.ndarray, List],
                         population_mean: float,
                         alternative: str = "two-sided") -> TestResult:
        """
        일표본 t-검정
        
        Args:
            data: 검정할 데이터
            population_mean: 모집단 평균 (귀무가설)
            alternative: 대립가설 ("two-sided", "greater", "less")
            
        Returns:
            TestResult: 검정 결과
        """
        try:
            from scipy import stats
            
            if isinstance(data, (list, np.ndarray)):
                data = pd.Series(data)
            
            # 결측값 제거
            clean_data = data.dropna()
            
            if len(clean_data) < 2:
                raise StatisticsException("일표본 t-검정을 위해서는 최소 2개의 관측값이 필요합니다")
            
            # t-검정 수행
            t_stat, p_value = stats.ttest_1samp(clean_data, population_mean)
            
            # 대립가설에 따른 p-value 조정
            if alternative == "greater":
                p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
            elif alternative == "less":
                p_value = p_value / 2 if t_stat < 0 else 1 - p_value / 2
            
            # 효과 크기 (Cohen's d)
            effect_size = (clean_data.mean() - population_mean) / clean_data.std()
            
            # 신뢰구간
            se = clean_data.std() / np.sqrt(len(clean_data))
            t_critical = stats.t.ppf(1 - self.alpha/2, len(clean_data) - 1)
            margin_error = t_critical * se
            ci_lower = clean_data.mean() - margin_error
            ci_upper = clean_data.mean() + margin_error
            
            # 해석
            interpretation = self._interpret_one_sample_t(
                t_stat, p_value, clean_data.mean(), population_mean, alternative
            )
            
            return TestResult(
                test_type=TestType.ONE_SAMPLE_T,
                statistic=float(t_stat),
                p_value=float(p_value),
                degrees_of_freedom=len(clean_data) - 1,
                effect_size=float(effect_size),
                confidence_interval=(float(ci_lower), float(ci_upper)),
                is_significant=p_value < self.alpha,
                interpretation=interpretation,
                assumptions_met={"normality": True},  # 가정 검정은 별도 수행
                metadata={
                    "sample_size": len(clean_data),
                    "sample_mean": float(clean_data.mean()),
                    "population_mean": population_mean,
                    "alternative": alternative
                }
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'test_type': 'one_sample_t'})
            raise StatisticsException(f"일표본 t-검정 실패: {error_info['message']}")
    
    def independent_t_test(self, 
                          group1: Union[pd.Series, np.ndarray, List],
                          group2: Union[pd.Series, np.ndarray, List],
                          equal_var: bool = True,
                          alternative: str = "two-sided") -> TestResult:
        """
        독립표본 t-검정
        
        Args:
            group1: 첫 번째 그룹 데이터
            group2: 두 번째 그룹 데이터
            equal_var: 등분산 가정 여부
            alternative: 대립가설 ("two-sided", "greater", "less")
            
        Returns:
            TestResult: 검정 결과
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
            
            if len(clean_group1) < 2 or len(clean_group2) < 2:
                raise StatisticsException("각 그룹은 최소 2개의 관측값이 필요합니다")
            
            # t-검정 수행
            t_stat, p_value = stats.ttest_ind(clean_group1, clean_group2, equal_var=equal_var)
            
            # 대립가설에 따른 p-value 조정
            if alternative == "greater":
                p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
            elif alternative == "less":
                p_value = p_value / 2 if t_stat < 0 else 1 - p_value / 2
            
            # 효과 크기 (Cohen's d)
            pooled_std = np.sqrt(((len(clean_group1) - 1) * clean_group1.var() + 
                                 (len(clean_group2) - 1) * clean_group2.var()) / 
                                (len(clean_group1) + len(clean_group2) - 2))
            effect_size = (clean_group1.mean() - clean_group2.mean()) / pooled_std
            
            # 자유도
            if equal_var:
                df = len(clean_group1) + len(clean_group2) - 2
            else:
                # Welch's t-test 자유도
                s1_sq = clean_group1.var()
                s2_sq = clean_group2.var()
                n1, n2 = len(clean_group1), len(clean_group2)
                df = (s1_sq/n1 + s2_sq/n2)**2 / ((s1_sq/n1)**2/(n1-1) + (s2_sq/n2)**2/(n2-1))
            
            # 신뢰구간
            se = np.sqrt(clean_group1.var()/len(clean_group1) + clean_group2.var()/len(clean_group2))
            t_critical = stats.t.ppf(1 - self.alpha/2, df)
            margin_error = t_critical * se
            mean_diff = clean_group1.mean() - clean_group2.mean()
            ci_lower = mean_diff - margin_error
            ci_upper = mean_diff + margin_error
            
            # 해석
            interpretation = self._interpret_independent_t(
                t_stat, p_value, clean_group1.mean(), clean_group2.mean(), alternative
            )
            
            return TestResult(
                test_type=TestType.INDEPENDENT_T,
                statistic=float(t_stat),
                p_value=float(p_value),
                degrees_of_freedom=float(df),
                effect_size=float(effect_size),
                confidence_interval=(float(ci_lower), float(ci_upper)),
                is_significant=p_value < self.alpha,
                interpretation=interpretation,
                assumptions_met={"equal_variance": equal_var},
                metadata={
                    "group1_size": len(clean_group1),
                    "group2_size": len(clean_group2),
                    "group1_mean": float(clean_group1.mean()),
                    "group2_mean": float(clean_group2.mean()),
                    "equal_var": equal_var,
                    "alternative": alternative
                }
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'test_type': 'independent_t'})
            raise StatisticsException(f"독립표본 t-검정 실패: {error_info['message']}")
    
    def paired_t_test(self, 
                     before: Union[pd.Series, np.ndarray, List],
                     after: Union[pd.Series, np.ndarray, List],
                     alternative: str = "two-sided") -> TestResult:
        """
        대응표본 t-검정
        
        Args:
            before: 처치 전 데이터
            after: 처치 후 데이터
            alternative: 대립가설 ("two-sided", "greater", "less")
            
        Returns:
            TestResult: 검정 결과
        """
        try:
            from scipy import stats
            
            if isinstance(before, (list, np.ndarray)):
                before = pd.Series(before)
            if isinstance(after, (list, np.ndarray)):
                after = pd.Series(after)
            
            # 길이 확인
            if len(before) != len(after):
                raise StatisticsException("대응표본 t-검정을 위해서는 두 그룹의 크기가 같아야 합니다")
            
            # 결측값 제거 (쌍으로)
            valid_idx = ~(before.isna() | after.isna())
            clean_before = before[valid_idx]
            clean_after = after[valid_idx]
            
            if len(clean_before) < 2:
                raise StatisticsException("대응표본 t-검정을 위해서는 최소 2쌍의 관측값이 필요합니다")
            
            # t-검정 수행
            t_stat, p_value = stats.ttest_rel(clean_before, clean_after)
            
            # 대립가설에 따른 p-value 조정
            if alternative == "greater":
                p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
            elif alternative == "less":
                p_value = p_value / 2 if t_stat < 0 else 1 - p_value / 2
            
            # 차이 계산
            differences = clean_after - clean_before
            
            # 효과 크기 (Cohen's d for paired samples)
            effect_size = differences.mean() / differences.std()
            
            # 신뢰구간
            se = differences.std() / np.sqrt(len(differences))
            t_critical = stats.t.ppf(1 - self.alpha/2, len(differences) - 1)
            margin_error = t_critical * se
            ci_lower = differences.mean() - margin_error
            ci_upper = differences.mean() + margin_error
            
            # 해석
            interpretation = self._interpret_paired_t(
                t_stat, p_value, differences.mean(), alternative
            )
            
            return TestResult(
                test_type=TestType.PAIRED_T,
                statistic=float(t_stat),
                p_value=float(p_value),
                degrees_of_freedom=len(differences) - 1,
                effect_size=float(effect_size),
                confidence_interval=(float(ci_lower), float(ci_upper)),
                is_significant=p_value < self.alpha,
                interpretation=interpretation,
                assumptions_met={"normality_differences": True},
                metadata={
                    "sample_size": len(differences),
                    "mean_difference": float(differences.mean()),
                    "before_mean": float(clean_before.mean()),
                    "after_mean": float(clean_after.mean()),
                    "alternative": alternative
                }
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'test_type': 'paired_t'})
            raise StatisticsException(f"대응표본 t-검정 실패: {error_info['message']}")
    
    def one_way_anova(self, 
                     *groups: Union[pd.Series, np.ndarray, List],
                     group_names: Optional[List[str]] = None) -> TestResult:
        """
        일원분산분석 (One-way ANOVA)
        
        Args:
            *groups: 비교할 그룹들
            group_names: 그룹 이름들
            
        Returns:
            TestResult: 검정 결과
        """
        try:
            from scipy import stats
            
            if len(groups) < 2:
                raise StatisticsException("일원분산분석을 위해서는 최소 2개 그룹이 필요합니다")
            
            # 데이터 정리
            clean_groups = []
            for i, group in enumerate(groups):
                if isinstance(group, (list, np.ndarray)):
                    group = pd.Series(group)
                clean_group = group.dropna()
                if len(clean_group) < 2:
                    raise StatisticsException(f"그룹 {i+1}은 최소 2개의 관측값이 필요합니다")
                clean_groups.append(clean_group)
            
            # ANOVA 수행
            f_stat, p_value = stats.f_oneway(*clean_groups)
            
            # 자유도 계산
            k = len(clean_groups)  # 그룹 수
            n_total = sum(len(group) for group in clean_groups)
            df_between = k - 1
            df_within = n_total - k
            
            # 효과 크기 (Eta-squared)
            ss_between = sum(len(group) * (group.mean() - np.concatenate(clean_groups).mean())**2 
                           for group in clean_groups)
            ss_total = sum((np.concatenate(clean_groups) - np.concatenate(clean_groups).mean())**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            # 해석
            interpretation = self._interpret_anova(f_stat, p_value, k)
            
            # 그룹 정보
            if group_names is None:
                group_names = [f"Group_{i+1}" for i in range(len(clean_groups))]
            
            group_stats = {}
            for i, (name, group) in enumerate(zip(group_names, clean_groups)):
                group_stats[name] = {
                    "mean": float(group.mean()),
                    "std": float(group.std()),
                    "size": len(group)
                }
            
            return TestResult(
                test_type=TestType.ONE_WAY_ANOVA,
                statistic=float(f_stat),
                p_value=float(p_value),
                degrees_of_freedom=(df_between, df_within),
                effect_size=float(eta_squared),
                confidence_interval=None,
                is_significant=p_value < self.alpha,
                interpretation=interpretation,
                assumptions_met={"equal_variance": True, "normality": True},
                metadata={
                    "num_groups": k,
                    "total_sample_size": n_total,
                    "group_stats": group_stats
                }
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'test_type': 'one_way_anova'})
            raise StatisticsException(f"일원분산분석 실패: {error_info['message']}")
    
    def two_way_anova(self, 
                     data: pd.DataFrame,
                     dependent_var: str,
                     factor1: str,
                     factor2: str) -> Dict[str, TestResult]:
        """
        이원분산분석 (Two-way ANOVA)
        
        Args:
            data: 데이터프레임
            dependent_var: 종속변수 컬럼명
            factor1: 첫 번째 요인 컬럼명
            factor2: 두 번째 요인 컬럼명
            
        Returns:
            Dict[str, TestResult]: 주효과 및 상호작용 효과 결과
        """
        try:
            # 결측값 제거
            clean_data = data[[dependent_var, factor1, factor2]].dropna()
            
            if len(clean_data) < 4:
                raise StatisticsException("이원분산분석을 위해서는 최소 4개의 관측값이 필요합니다")
            
            # 그룹별 데이터 준비
            groups = clean_data.groupby([factor1, factor2])[dependent_var].apply(list)
            
            # 간단한 이원분산분석 구현 (statsmodels 없이)
            results = self._simple_two_way_anova(clean_data, dependent_var, factor1, factor2)
            
            return results
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'test_type': 'two_way_anova'})
            raise StatisticsException(f"이원분산분석 실패: {error_info['message']}")
    
    def _simple_two_way_anova(self, 
                             data: pd.DataFrame,
                             dependent_var: str,
                             factor1: str,
                             factor2: str) -> Dict[str, TestResult]:
        """간단한 이원분산분석 구현"""
        from scipy import stats
        
        # 전체 평균
        grand_mean = data[dependent_var].mean()
        n_total = len(data)
        
        # 요인별 수준
        levels_f1 = data[factor1].unique()
        levels_f2 = data[factor2].unique()
        
        # 제곱합 계산
        ss_total = ((data[dependent_var] - grand_mean) ** 2).sum()
        
        # 주효과 A (factor1)
        ss_a = 0
        for level in levels_f1:
            subset = data[data[factor1] == level]
            n_level = len(subset)
            mean_level = subset[dependent_var].mean()
            ss_a += n_level * (mean_level - grand_mean) ** 2
        
        # 주효과 B (factor2)
        ss_b = 0
        for level in levels_f2:
            subset = data[data[factor2] == level]
            n_level = len(subset)
            mean_level = subset[dependent_var].mean()
            ss_b += n_level * (mean_level - grand_mean) ** 2
        
        # 상호작용 효과 AB
        ss_ab = 0
        for level_a in levels_f1:
            for level_b in levels_f2:
                subset = data[(data[factor1] == level_a) & (data[factor2] == level_b)]
                if len(subset) > 0:
                    n_cell = len(subset)
                    mean_cell = subset[dependent_var].mean()
                    mean_a = data[data[factor1] == level_a][dependent_var].mean()
                    mean_b = data[data[factor2] == level_b][dependent_var].mean()
                    ss_ab += n_cell * (mean_cell - mean_a - mean_b + grand_mean) ** 2
        
        # 오차 제곱합
        ss_error = ss_total - ss_a - ss_b - ss_ab
        
        # 자유도
        df_a = len(levels_f1) - 1
        df_b = len(levels_f2) - 1
        df_ab = df_a * df_b
        df_error = n_total - len(levels_f1) * len(levels_f2)
        
        # 평균제곱
        ms_a = ss_a / df_a if df_a > 0 else 0
        ms_b = ss_b / df_b if df_b > 0 else 0
        ms_ab = ss_ab / df_ab if df_ab > 0 else 0
        ms_error = ss_error / df_error if df_error > 0 else 1
        
        # F 통계량
        f_a = ms_a / ms_error if ms_error > 0 else 0
        f_b = ms_b / ms_error if ms_error > 0 else 0
        f_ab = ms_ab / ms_error if ms_error > 0 else 0
        
        # p-value
        p_a = 1 - stats.f.cdf(f_a, df_a, df_error) if f_a > 0 else 1
        p_b = 1 - stats.f.cdf(f_b, df_b, df_error) if f_b > 0 else 1
        p_ab = 1 - stats.f.cdf(f_ab, df_ab, df_error) if f_ab > 0 else 1
        
        # 효과 크기 (부분 Eta-squared)
        eta_sq_a = ss_a / (ss_a + ss_error) if (ss_a + ss_error) > 0 else 0
        eta_sq_b = ss_b / (ss_b + ss_error) if (ss_b + ss_error) > 0 else 0
        eta_sq_ab = ss_ab / (ss_ab + ss_error) if (ss_ab + ss_error) > 0 else 0
        
        results = {}
        
        # 주효과 A
        results[f"main_effect_{factor1}"] = TestResult(
            test_type=TestType.TWO_WAY_ANOVA,
            statistic=float(f_a),
            p_value=float(p_a),
            degrees_of_freedom=(df_a, df_error),
            effect_size=float(eta_sq_a),
            confidence_interval=None,
            is_significant=p_a < self.alpha,
            interpretation=f"{factor1}의 주효과가 {'유의함' if p_a < self.alpha else '유의하지 않음'}",
            assumptions_met={"equal_variance": True, "normality": True},
            metadata={"effect_type": "main_effect", "factor": factor1}
        )
        
        # 주효과 B
        results[f"main_effect_{factor2}"] = TestResult(
            test_type=TestType.TWO_WAY_ANOVA,
            statistic=float(f_b),
            p_value=float(p_b),
            degrees_of_freedom=(df_b, df_error),
            effect_size=float(eta_sq_b),
            confidence_interval=None,
            is_significant=p_b < self.alpha,
            interpretation=f"{factor2}의 주효과가 {'유의함' if p_b < self.alpha else '유의하지 않음'}",
            assumptions_met={"equal_variance": True, "normality": True},
            metadata={"effect_type": "main_effect", "factor": factor2}
        )
        
        # 상호작용 효과
        results["interaction"] = TestResult(
            test_type=TestType.TWO_WAY_ANOVA,
            statistic=float(f_ab),
            p_value=float(p_ab),
            degrees_of_freedom=(df_ab, df_error),
            effect_size=float(eta_sq_ab),
            confidence_interval=None,
            is_significant=p_ab < self.alpha,
            interpretation=f"{factor1}과 {factor2}의 상호작용 효과가 {'유의함' if p_ab < self.alpha else '유의하지 않음'}",
            assumptions_met={"equal_variance": True, "normality": True},
            metadata={"effect_type": "interaction", "factors": [factor1, factor2]}
        )
        
        return results
    
    def post_hoc_test(self, 
                     *groups: Union[pd.Series, np.ndarray, List],
                     method: PostHocMethod = PostHocMethod.TUKEY,
                     group_names: Optional[List[str]] = None) -> PostHocResult:
        """
        사후 검정
        
        Args:
            *groups: 비교할 그룹들
            method: 사후 검정 방법
            group_names: 그룹 이름들
            
        Returns:
            PostHocResult: 사후 검정 결과
        """
        try:
            if len(groups) < 2:
                raise StatisticsException("사후 검정을 위해서는 최소 2개 그룹이 필요합니다")
            
            # 데이터 정리
            clean_groups = []
            for group in groups:
                if isinstance(group, (list, np.ndarray)):
                    group = pd.Series(group)
                clean_groups.append(group.dropna())
            
            if group_names is None:
                group_names = [f"Group_{i+1}" for i in range(len(clean_groups))]
            
            # 방법별 사후 검정 수행
            if method == PostHocMethod.TUKEY:
                comparisons = self._tukey_hsd(clean_groups, group_names)
            elif method == PostHocMethod.BONFERRONI:
                comparisons = self._bonferroni_correction(clean_groups, group_names)
            else:
                raise StatisticsException(f"지원하지 않는 사후 검정 방법: {method}")
            
            # 전체 유의성 판단
            overall_significant = any(comp['is_significant'] for comp in comparisons)
            
            return PostHocResult(
                method=method,
                comparisons=comparisons,
                overall_significant=overall_significant,
                metadata={
                    "num_groups": len(clean_groups),
                    "num_comparisons": len(comparisons),
                    "alpha": self.alpha
                }
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'method': method.value})
            raise StatisticsException(f"사후 검정 실패: {error_info['message']}")
    
    def _tukey_hsd(self, groups: List[pd.Series], group_names: List[str]) -> List[Dict[str, Any]]:
        """Tukey HSD 검정"""
        from scipy import stats
        
        comparisons = []
        
        # 전체 MSE 계산
        all_data = pd.concat(groups, ignore_index=True)
        grand_mean = all_data.mean()
        
        ss_within = sum((group - group.mean()).pow(2).sum() for group in groups)
        df_within = sum(len(group) - 1 for group in groups)
        mse = ss_within / df_within if df_within > 0 else 1
        
        # 모든 쌍 비교
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group1, group2 = groups[i], groups[j]
                name1, name2 = group_names[i], group_names[j]
                
                # 평균 차이
                mean_diff = group1.mean() - group2.mean()
                
                # 표준오차
                se = np.sqrt(mse * (1/len(group1) + 1/len(group2)))
                
                # q 통계량 (Studentized range)
                q_stat = abs(mean_diff) / se
                
                # 임계값 (근사치)
                # 정확한 계산을 위해서는 scipy.stats.tukey_hsd 필요
                alpha_adj = self.alpha
                t_critical = stats.t.ppf(1 - alpha_adj/2, df_within)
                q_critical = t_critical * np.sqrt(2)  # 근사치
                
                is_significant = q_stat > q_critical
                
                comparisons.append({
                    "group1": name1,
                    "group2": name2,
                    "mean_diff": float(mean_diff),
                    "statistic": float(q_stat),
                    "critical_value": float(q_critical),
                    "is_significant": is_significant,
                    "interpretation": f"{name1}과 {name2} 간 차이가 {'유의함' if is_significant else '유의하지 않음'}"
                })
        
        return comparisons
    
    def _bonferroni_correction(self, groups: List[pd.Series], group_names: List[str]) -> List[Dict[str, Any]]:
        """Bonferroni 보정"""
        from scipy import stats
        
        comparisons = []
        num_comparisons = len(groups) * (len(groups) - 1) // 2
        alpha_adj = self.alpha / num_comparisons
        
        # 모든 쌍 비교
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group1, group2 = groups[i], groups[j]
                name1, name2 = group_names[i], group_names[j]
                
                # 독립표본 t-검정
                t_stat, p_value = stats.ttest_ind(group1, group2)
                
                # Bonferroni 보정된 p-value
                p_adj = min(p_value * num_comparisons, 1.0)
                
                is_significant = p_adj < self.alpha
                
                comparisons.append({
                    "group1": name1,
                    "group2": name2,
                    "mean_diff": float(group1.mean() - group2.mean()),
                    "statistic": float(t_stat),
                    "p_value": float(p_value),
                    "p_adjusted": float(p_adj),
                    "is_significant": is_significant,
                    "interpretation": f"{name1}과 {name2} 간 차이가 {'유의함' if is_significant else '유의하지 않음'}"
                })
        
        return comparisons
    
    def _interpret_one_sample_t(self, t_stat: float, p_value: float, 
                               sample_mean: float, population_mean: float, 
                               alternative: str) -> str:
        """일표본 t-검정 해석"""
        if p_value < self.alpha:
            if alternative == "two-sided":
                return f"표본 평균({sample_mean:.3f})이 모집단 평균({population_mean})과 유의하게 다릅니다."
            elif alternative == "greater":
                return f"표본 평균({sample_mean:.3f})이 모집단 평균({population_mean})보다 유의하게 큽니다."
            else:
                return f"표본 평균({sample_mean:.3f})이 모집단 평균({population_mean})보다 유의하게 작습니다."
        else:
            return f"표본 평균({sample_mean:.3f})과 모집단 평균({population_mean}) 간에 유의한 차이가 없습니다."
    
    def _interpret_independent_t(self, t_stat: float, p_value: float,
                                mean1: float, mean2: float, alternative: str) -> str:
        """독립표본 t-검정 해석"""
        if p_value < self.alpha:
            if alternative == "two-sided":
                return f"두 그룹 간 평균에 유의한 차이가 있습니다 (그룹1: {mean1:.3f}, 그룹2: {mean2:.3f})."
            elif alternative == "greater":
                return f"그룹1의 평균({mean1:.3f})이 그룹2의 평균({mean2:.3f})보다 유의하게 큽니다."
            else:
                return f"그룹1의 평균({mean1:.3f})이 그룹2의 평균({mean2:.3f})보다 유의하게 작습니다."
        else:
            return f"두 그룹 간 평균에 유의한 차이가 없습니다 (그룹1: {mean1:.3f}, 그룹2: {mean2:.3f})."
    
    def _interpret_paired_t(self, t_stat: float, p_value: float,
                           mean_diff: float, alternative: str) -> str:
        """대응표본 t-검정 해석"""
        if p_value < self.alpha:
            if alternative == "two-sided":
                return f"처치 전후 간에 유의한 차이가 있습니다 (평균 차이: {mean_diff:.3f})."
            elif alternative == "greater":
                return f"처치 후가 처치 전보다 유의하게 큽니다 (평균 차이: {mean_diff:.3f})."
            else:
                return f"처치 후가 처치 전보다 유의하게 작습니다 (평균 차이: {mean_diff:.3f})."
        else:
            return f"처치 전후 간에 유의한 차이가 없습니다 (평균 차이: {mean_diff:.3f})."
    
    def _interpret_anova(self, f_stat: float, p_value: float, num_groups: int) -> str:
        """ANOVA 해석"""
        if p_value < self.alpha:
            return f"{num_groups}개 그룹 간에 유의한 차이가 있습니다 (F = {f_stat:.3f}, p = {p_value:.3f})."
        else:
            return f"{num_groups}개 그룹 간에 유의한 차이가 없습니다 (F = {f_stat:.3f}, p = {p_value:.3f})."
    
    def calculate_effect_size(self, test_result: TestResult) -> Dict[str, Any]:
        """효과 크기 해석"""
        effect_size = test_result.effect_size
        
        if test_result.test_type in [TestType.ONE_SAMPLE_T, TestType.INDEPENDENT_T, TestType.PAIRED_T]:
            # Cohen's d 해석
            if abs(effect_size) < 0.2:
                magnitude = "작음"
            elif abs(effect_size) < 0.5:
                magnitude = "중간"
            elif abs(effect_size) < 0.8:
                magnitude = "큼"
            else:
                magnitude = "매우 큼"
        
        elif test_result.test_type in [TestType.ONE_WAY_ANOVA, TestType.TWO_WAY_ANOVA]:
            # Eta-squared 해석
            if effect_size < 0.01:
                magnitude = "작음"
            elif effect_size < 0.06:
                magnitude = "중간"
            elif effect_size < 0.14:
                magnitude = "큼"
            else:
                magnitude = "매우 큼"
        
        else:
            magnitude = "알 수 없음"
        
        return {
            "effect_size": effect_size,
            "magnitude": magnitude,
            "interpretation": f"효과 크기는 {magnitude}입니다 ({effect_size:.3f})"
        } 