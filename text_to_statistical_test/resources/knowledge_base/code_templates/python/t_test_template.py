"""
t-검정 Python 코드 템플릿

독립표본, 대응표본, 일표본 t-검정을 수행하는 통합 템플릿
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import normaltest, levene, shapiro
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, Union
import warnings

def check_assumptions(data1: np.ndarray, data2: Optional[np.ndarray] = None) -> Dict:
    """
    t-검정의 가정 확인
    
    Args:
        data1: 첫 번째 그룹 데이터
        data2: 두 번째 그룹 데이터 (독립표본 t-검정시)
    
    Returns:
        가정 검정 결과 딕셔너리
    """
    results = {}
    
    # 정규성 검정
    # Shapiro-Wilk 검정 (표본 크기 < 50일 때 권장)
    if len(data1) < 50:
        stat1, p1 = shapiro(data1)
        results['normality_test'] = 'shapiro'
    else:
        # D'Agostino와 Pearson의 정규성 검정 (큰 표본)
        stat1, p1 = normaltest(data1)
        results['normality_test'] = 'dagostino'
    
    results['normality_group1'] = {
        'statistic': stat1,
        'p_value': p1,
        'is_normal': p1 > 0.05
    }
    
    if data2 is not None:
        if len(data2) < 50:
            stat2, p2 = shapiro(data2)
        else:
            stat2, p2 = normaltest(data2)
        
        results['normality_group2'] = {
            'statistic': stat2,
            'p_value': p2,
            'is_normal': p2 > 0.05
        }
        
        # 등분산성 검정 (Levene's test)
        stat_var, p_var = levene(data1, data2)
        results['equal_variance'] = {
            'statistic': stat_var,
            'p_value': p_var,
            'is_equal': p_var > 0.05
        }
    
    return results

def one_sample_ttest(data: np.ndarray, 
                    mu: float,
                    alternative: str = 'two-sided',
                    alpha: float = 0.05) -> Dict:
    """
    일표본 t-검정
    
    Args:
        data: 표본 데이터
        mu: 검정하고자 하는 모집단 평균
        alternative: 대립가설 ('two-sided', 'less', 'greater')
        alpha: 유의수준
    
    Returns:
        검정 결과 딕셔너리
    """
    # 가정 확인
    assumptions = check_assumptions(data)
    
    # t-검정 수행
    t_stat, p_value = stats.ttest_1samp(data, mu, alternative=alternative)
    
    # 효과 크기 계산 (Cohen's d)
    cohens_d = (np.mean(data) - mu) / np.std(data, ddof=1)
    
    # 신뢰구간 계산
    n = len(data)
    df = n - 1
    mean_diff = np.mean(data) - mu
    se = stats.sem(data)
    t_critical = stats.t.ppf(1 - alpha/2, df)
    ci_lower = mean_diff - t_critical * se
    ci_upper = mean_diff + t_critical * se
    
    return {
        'test_type': 'one_sample_ttest',
        'sample_size': n,
        'sample_mean': np.mean(data),
        'sample_std': np.std(data, ddof=1),
        'hypothesized_mean': mu,
        'mean_difference': mean_diff,
        't_statistic': t_stat,
        'degrees_of_freedom': df,
        'p_value': p_value,
        'alpha': alpha,
        'is_significant': p_value < alpha,
        'effect_size_cohens_d': cohens_d,
        'confidence_interval': (ci_lower, ci_upper),
        'alternative_hypothesis': alternative,
        'assumptions': assumptions
    }

def independent_ttest(group1: np.ndarray, 
                     group2: np.ndarray,
                     equal_var: Optional[bool] = None,
                     alternative: str = 'two-sided',
                     alpha: float = 0.05) -> Dict:
    """
    독립표본 t-검정
    
    Args:
        group1: 첫 번째 그룹 데이터
        group2: 두 번째 그룹 데이터
        equal_var: 등분산 가정 (None이면 자동 결정)
        alternative: 대립가설 ('two-sided', 'less', 'greater')
        alpha: 유의수준
    
    Returns:
        검정 결과 딕셔너리
    """
    # 가정 확인
    assumptions = check_assumptions(group1, group2)
    
    # 등분산성 자동 결정
    if equal_var is None:
        equal_var = assumptions['equal_variance']['is_equal']
    
    # t-검정 수행
    t_stat, p_value = stats.ttest_ind(
        group1, group2, 
        equal_var=equal_var, 
        alternative=alternative
    )
    
    # 효과 크기 계산 (Cohen's d)
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    if equal_var:
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        cohens_d = (mean1 - mean2) / pooled_std
        df = n1 + n2 - 2
    else:
        # Welch's t-test degrees of freedom
        se1, se2 = std1/np.sqrt(n1), std2/np.sqrt(n2)
        se_diff = np.sqrt(se1**2 + se2**2)
        df = se_diff**4 / (se1**4/(n1-1) + se2**4/(n2-1))
        cohens_d = (mean1 - mean2) / np.sqrt((std1**2 + std2**2) / 2)
    
    # 신뢰구간 계산
    mean_diff = mean1 - mean2
    if equal_var:
        se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)
    else:
        se_diff = np.sqrt(std1**2/n1 + std2**2/n2)
    
    t_critical = stats.t.ppf(1 - alpha/2, df)
    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff
    
    return {
        'test_type': 'independent_ttest',
        'equal_variance_assumed': equal_var,
        'sample_sizes': (n1, n2),
        'group1_mean': mean1,
        'group1_std': std1,
        'group2_mean': mean2,
        'group2_std': std2,
        'mean_difference': mean_diff,
        't_statistic': t_stat,
        'degrees_of_freedom': df,
        'p_value': p_value,
        'alpha': alpha,
        'is_significant': p_value < alpha,
        'effect_size_cohens_d': cohens_d,
        'confidence_interval': (ci_lower, ci_upper),
        'alternative_hypothesis': alternative,
        'assumptions': assumptions
    }

def paired_ttest(before: np.ndarray, 
                after: np.ndarray,
                alternative: str = 'two-sided',
                alpha: float = 0.05) -> Dict:
    """
    대응표본 t-검정
    
    Args:
        before: 처치 전 데이터
        after: 처치 후 데이터
        alternative: 대립가설 ('two-sided', 'less', 'greater')
        alpha: 유의수준
    
    Returns:
        검정 결과 딕셔너리
    """
    # 차이값 계산
    differences = after - before
    
    # 가정 확인 (차이값의 정규성)
    assumptions = check_assumptions(differences)
    
    # 대응표본 t-검정 수행
    t_stat, p_value = stats.ttest_rel(before, after, alternative=alternative)
    
    # 효과 크기 계산 (Cohen's d)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff
    
    # 신뢰구간 계산
    n = len(differences)
    df = n - 1
    se_diff = stats.sem(differences)
    t_critical = stats.t.ppf(1 - alpha/2, df)
    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff
    
    return {
        'test_type': 'paired_ttest',
        'sample_size': n,
        'before_mean': np.mean(before),
        'before_std': np.std(before, ddof=1),
        'after_mean': np.mean(after),
        'after_std': np.std(after, ddof=1),
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        't_statistic': t_stat,
        'degrees_of_freedom': df,
        'p_value': p_value,
        'alpha': alpha,
        'is_significant': p_value < alpha,
        'effect_size_cohens_d': cohens_d,
        'confidence_interval': (ci_lower, ci_upper),
        'alternative_hypothesis': alternative,
        'assumptions': assumptions
    }

def create_ttest_visualization(data1: np.ndarray, 
                              data2: Optional[np.ndarray] = None,
                              test_type: str = 'independent',
                              title: str = 't-test Results') -> plt.Figure:
    """
    t-검정 결과 시각화
    
    Args:
        data1: 첫 번째 그룹 데이터
        data2: 두 번째 그룹 데이터
        test_type: 검정 유형 ('independent', 'paired', 'one_sample')
        title: 그래프 제목
    
    Returns:
        matplotlib Figure 객체
    """
    if test_type == 'independent' and data2 is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 히스토그램
        axes[0, 0].hist(data1, alpha=0.7, label='Group 1', bins=20)
        axes[0, 0].hist(data2, alpha=0.7, label='Group 2', bins=20)
        axes[0, 0].set_title('Distribution Comparison')
        axes[0, 0].legend()
        
        # 박스플롯
        axes[0, 1].boxplot([data1, data2], labels=['Group 1', 'Group 2'])
        axes[0, 1].set_title('Box Plot Comparison')
        
        # Q-Q 플롯 (정규성 확인)
        stats.probplot(data1, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot Group 1')
        
        stats.probplot(data2, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot Group 2')
        
    elif test_type == 'one_sample':
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 히스토그램
        axes[0].hist(data1, bins=20, alpha=0.7)
        axes[0].axvline(np.mean(data1), color='red', linestyle='--', label='Sample Mean')
        axes[0].set_title('Sample Distribution')
        axes[0].legend()
        
        # Q-Q 플롯
        stats.probplot(data1, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot')
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    return fig

def interpret_ttest_results(results: Dict) -> str:
    """
    t-검정 결과 해석
    
    Args:
        results: t-검정 결과 딕셔너리
    
    Returns:
        해석 텍스트
    """
    interpretation = []
    
    # 기본 정보
    test_type = results['test_type']
    p_value = results['p_value']
    alpha = results['alpha']
    is_significant = results['is_significant']
    cohens_d = results['effect_size_cohens_d']
    
    interpretation.append(f"=== {test_type.upper()} 결과 해석 ===")
    
    # 유의성 해석
    if is_significant:
        interpretation.append(f"p-값({p_value:.4f})이 유의수준({alpha})보다 작으므로 통계적으로 유의합니다.")
        interpretation.append("귀무가설을 기각하고 대립가설을 채택합니다.")
    else:
        interpretation.append(f"p-값({p_value:.4f})이 유의수준({alpha})보다 크므로 통계적으로 유의하지 않습니다.")
        interpretation.append("귀무가설을 채택합니다.")
    
    # 효과 크기 해석
    effect_magnitude = abs(cohens_d)
    if effect_magnitude < 0.2:
        effect_desc = "매우 작은"
    elif effect_magnitude < 0.5:
        effect_desc = "작은"
    elif effect_magnitude < 0.8:
        effect_desc = "중간"
    else:
        effect_desc = "큰"
    
    interpretation.append(f"효과 크기(Cohen's d = {cohens_d:.3f})는 {effect_desc} 효과를 나타냅니다.")
    
    # 신뢰구간 해석
    ci_lower, ci_upper = results['confidence_interval']
    interpretation.append(f"95% 신뢰구간: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # 가정 검토 결과
    assumptions = results['assumptions']
    if 'normality_group1' in assumptions:
        if not assumptions['normality_group1']['is_normal']:
            interpretation.append("⚠️ 주의: 정규성 가정이 만족되지 않습니다. 비모수 검정을 고려하세요.")
    
    if 'equal_variance' in assumptions and not assumptions['equal_variance']['is_equal']:
        interpretation.append("⚠️ 주의: 등분산성 가정이 만족되지 않습니다. Welch's t-test를 사용했습니다.")
    
    return "\n".join(interpretation)

# 사용 예시
if __name__ == "__main__":
    # 예시 데이터 생성
    np.random.seed(42)
    group1 = np.random.normal(100, 15, 30)  # 평균 100, 표준편차 15
    group2 = np.random.normal(105, 16, 25)  # 평균 105, 표준편차 16
    
    # 독립표본 t-검정
    results = independent_ttest(group1, group2)
    print(interpret_ttest_results(results))
    
    # 시각화
    fig = create_ttest_visualization(group1, group2, 'independent')
    plt.show() 