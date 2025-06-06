"""
플롯 생성 서비스 모듈
통계 분석 결과를 시각화하기 위한 다양한 차트와 플롯을 생성합니다.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from utils.error_handler import ErrorHandler

class PlotGenerator:
    """통계 분석을 위한 플롯 생성기"""
    
    def __init__(self, 
                 default_style: str = "whitegrid",
                 figure_size: Tuple[int, int] = (10, 6),
                 color_palette: str = "Set2"):
        """
        Args:
            default_style: 기본 플롯 스타일
            figure_size: 기본 figure 크기
            color_palette: 기본 컬러 팔레트
        """
        self.logger = logging.getLogger(__name__)
        self.error_handler = ErrorHandler()
        
        # 기본 설정
        self.default_style = default_style
        self.figure_size = figure_size
        self.color_palette = color_palette
        
        # matplotlib 및 seaborn 기본 설정
        plt.style.use('default')
        sns.set_style(default_style)
        sns.set_palette(color_palette)
        
        # 플롯 유형별 생성 함수 매핑
        self.plot_generators = {
            # 기본 탐색적 플롯
            'histogram': self._create_histogram,
            'boxplot': self._create_boxplot,
            'scatterplot': self._create_scatterplot,
            'correlation_matrix': self._create_correlation_matrix,
            'pairplot': self._create_pairplot,
            
            # 통계 분석 플롯
            'qq_plot': self._create_qq_plot,
            'residual_plot': self._create_residual_plot,
            'regression_plot': self._create_regression_plot,
            'anova_plot': self._create_anova_plot,
            
            # 분포 및 확률 플롯
            'distribution_comparison': self._create_distribution_comparison,
            'probability_plot': self._create_probability_plot,
            'density_plot': self._create_density_plot,
            
            # 특수 분석 플롯
            'bootstrap_plot': self._create_bootstrap_plot,
            'power_analysis_plot': self._create_power_analysis_plot,
            'effect_size_plot': self._create_effect_size_plot,
            
            # 대화형 플롯
            'interactive_scatter': self._create_interactive_scatter,
            'interactive_histogram': self._create_interactive_histogram,
            'interactive_boxplot': self._create_interactive_boxplot
        }
    
    def create_plot(self,
                   plot_type: str,
                   data: pd.DataFrame,
                   plot_config: Dict[str, Any],
                   style_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        지정된 유형의 플롯 생성
        
        Args:
            plot_type: 플롯 유형
            data: 데이터프레임
            plot_config: 플롯 설정
            style_config: 스타일 설정
            
        Returns:
            플롯 정보 딕셔너리
        """
        try:
            if plot_type not in self.plot_generators:
                raise ValueError(f"지원하지 않는 플롯 유형: {plot_type}")
            
            # 스타일 적용
            if style_config:
                self._apply_style_config(style_config)
            
            # 플롯 생성
            generator_func = self.plot_generators[plot_type]
            plot_result = generator_func(data, plot_config)
            
            # 결과 정보 추가
            plot_result.update({
                'plot_type': plot_type,
                'data_shape': data.shape,
                'config_used': plot_config,
                'style_used': style_config or {}
            })
            
            self.logger.info(f"플롯 생성 완료: {plot_type}")
            return plot_result
            
        except Exception as e:
            self.logger.error(f"플롯 생성 오류 ({plot_type}): {e}")
            return self.error_handler.handle_error(e, default_return={
                'plot_type': plot_type,
                'success': False,
                'error': str(e)
            })
    
    def create_analysis_dashboard(self,
                                 data: pd.DataFrame,
                                 analysis_results: Dict[str, Any],
                                 dashboard_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        분석 결과 종합 대시보드 생성
        
        Args:
            data: 원본 데이터
            analysis_results: 분석 결과
            dashboard_config: 대시보드 설정
            
        Returns:
            대시보드 정보
        """
        try:
            analysis_type = analysis_results.get('analysis_type', 'unknown')
            
            if analysis_type == 't_test':
                return self._create_ttest_dashboard(data, analysis_results, dashboard_config)
            elif analysis_type == 'anova':
                return self._create_anova_dashboard(data, analysis_results, dashboard_config)
            elif analysis_type == 'correlation':
                return self._create_correlation_dashboard(data, analysis_results, dashboard_config)
            elif analysis_type == 'regression':
                return self._create_regression_dashboard(data, analysis_results, dashboard_config)
            else:
                return self._create_general_dashboard(data, analysis_results, dashboard_config)
                
        except Exception as e:
            self.logger.error(f"대시보드 생성 오류: {e}")
            return self.error_handler.handle_error(e, default_return={})
    
    def _create_histogram(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """히스토그램 생성"""
        column = config.get('column')
        bins = config.get('bins', 'auto')
        
        if column not in data.columns:
            raise ValueError(f"컬럼을 찾을 수 없습니다: {column}")
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 히스토그램 생성
        n, bins_array, patches = ax.hist(data[column].dropna(), 
                                        bins=bins, 
                                        alpha=0.7,
                                        edgecolor='black')
        
        # 정규분포 곡선 추가 (옵션)
        if config.get('show_normal_curve', False):
            mu, sigma = data[column].mean(), data[column].std()
            x = np.linspace(data[column].min(), data[column].max(), 100)
            y = stats.norm.pdf(x, mu, sigma)
            y_scaled = y * len(data[column]) * (bins_array[1] - bins_array[0])
            ax.plot(x, y_scaled, 'r-', linewidth=2, label='Normal Distribution')
            ax.legend()
        
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Histogram of {column}')
        
        # 통계 정보 추가
        mean_val = data[column].mean()
        std_val = data[column].std()
        ax.axvline(mean_val, color='red', linestyle='--', 
                  label=f'Mean: {mean_val:.2f}')
        
        plt.tight_layout()
        
        return {
            'figure': fig,
            'statistics': {
                'mean': mean_val,
                'std': std_val,
                'skewness': stats.skew(data[column].dropna()),
                'kurtosis': stats.kurtosis(data[column].dropna())
            },
            'success': True
        }
    
    def _create_boxplot(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """박스플롯 생성"""
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        if x_col and y_col:
            # 그룹별 박스플롯
            sns.boxplot(data=data, x=x_col, y=y_col, ax=ax)
            ax.set_title(f'Box Plot: {y_col} by {x_col}')
        else:
            # 단일 변수 박스플롯
            column = y_col or x_col
            sns.boxplot(data=data, y=column, ax=ax)
            ax.set_title(f'Box Plot: {column}')
        
        # 이상치 정보 계산
        if y_col:
            Q1 = data[y_col].quantile(0.25)
            Q3 = data[y_col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data[y_col] < Q1 - 1.5*IQR) | (data[y_col] > Q3 + 1.5*IQR)]
        
        plt.tight_layout()
        
        return {
            'figure': fig,
            'outlier_count': len(outliers) if y_col else 0,
            'success': True
        }
    
    def _create_scatterplot(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """산점도 생성"""
        x_col = config['x_column']
        y_col = config['y_column']
        color_col = config.get('color_column')
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        if color_col and color_col in data.columns:
            scatter = ax.scatter(data[x_col], data[y_col], 
                               c=data[color_col], alpha=0.6, cmap='viridis')
            plt.colorbar(scatter, label=color_col)
        else:
            ax.scatter(data[x_col], data[y_col], alpha=0.6)
        
        # 회귀선 추가 (옵션)
        if config.get('show_regression_line', False):
            z = np.polyfit(data[x_col], data[y_col], 1)
            p = np.poly1d(z)
            ax.plot(data[x_col], p(data[x_col]), "r--", alpha=0.8)
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
        
        # 상관계수 계산
        correlation = data[x_col].corr(data[y_col])
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white'))
        
        plt.tight_layout()
        
        return {
            'figure': fig,
            'correlation': correlation,
            'success': True
        }
    
    def _create_correlation_matrix(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """상관관계 매트릭스 히트맵 생성"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            raise ValueError("상관관계 분석을 위해서는 최소 2개의 숫자형 변수가 필요합니다.")
        
        # 상관관계 계산
        corr_matrix = data[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 히트맵 생성
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, 
                   mask=mask if config.get('show_upper_triangle', False) else None,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   ax=ax)
        
        ax.set_title('Correlation Matrix')
        plt.tight_layout()
        
        # 강한 상관관계 찾기
        strong_correlations = self._find_strongest_correlations(corr_matrix)
        
        return {
            'figure': fig,
            'correlation_matrix': corr_matrix,
            'strong_correlations': strong_correlations,
            'success': True
        }

    def _create_pairplot(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """페어플롯 생성"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            raise ValueError("페어플롯을 위해서는 최소 2개의 숫자형 변수가 필요합니다.")
        
        # 변수 개수 제한 (성능상 이유)
        max_vars = config.get('max_variables', 5)
        if len(numeric_cols) > max_vars:
            numeric_cols = numeric_cols[:max_vars]
            
        # 색상 구분 변수
        hue_col = config.get('hue_column')
        
        try:
            # 페어플롯 생성
            if hue_col and hue_col in data.columns:
                g = sns.pairplot(data[list(numeric_cols) + [hue_col]], 
                               hue=hue_col, 
                               diag_kind='hist',
                               plot_kws={'alpha': 0.6})
            else:
                g = sns.pairplot(data[numeric_cols], 
                               diag_kind='hist',
                               plot_kws={'alpha': 0.6})
            
            g.fig.suptitle('Pairwise Relationships', y=1.02)
            
            return {
                'figure': g.fig,
                'variables_used': list(numeric_cols),
                'hue_variable': hue_col,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"페어플롯 생성 오류: {e}")
            # 대안으로 간단한 산점도 매트릭스 생성
            return self._create_simple_scatter_matrix(data[numeric_cols])

    def _create_simple_scatter_matrix(self, data: pd.DataFrame) -> Dict[str, Any]:
        """간단한 산점도 매트릭스 생성 (페어플롯 대안)"""
        n_vars = len(data.columns)
        fig, axes = plt.subplots(n_vars, n_vars, figsize=(12, 12))
        
        for i, col1 in enumerate(data.columns):
            for j, col2 in enumerate(data.columns):
                ax = axes[i, j] if n_vars > 1 else axes
                
                if i == j:
                    # 대각선: 히스토그램
                    ax.hist(data[col1].dropna(), bins=20, alpha=0.7)
                    ax.set_title(col1)
                else:
                    # 비대각선: 산점도
                    ax.scatter(data[col2], data[col1], alpha=0.6)
                    ax.set_xlabel(col2)
                    ax.set_ylabel(col1)
        
        plt.tight_layout()
        
        return {
            'figure': fig,
            'variables_used': list(data.columns),
            'success': True
        }
    
    def _create_qq_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Q-Q 플롯 생성 (정규성 검정)"""
        column = config['column']
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Q-Q 플롯 생성
        stats.probplot(data[column].dropna(), dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot: {column}')
        
        # 정규성 검정 수행
        shapiro_stat, shapiro_p = stats.shapiro(data[column].dropna()[:5000])  # 샘플 크기 제한
        
        # 결과 텍스트 추가
        ax.text(0.05, 0.95, f'Shapiro-Wilk p-value: {shapiro_p:.4f}', 
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white'))
        
        plt.tight_layout()
        
        return {
            'figure': fig,
            'normality_test': {
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'is_normal': shapiro_p > 0.05
            },
            'success': True
        }
    
    def _create_residual_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """잔차 플롯 생성"""
        predicted = config['predicted_values']
        residuals = config['residuals']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 잔차 vs 예측값
        ax1.scatter(predicted, residuals, alpha=0.6)
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted')
        
        # 2. 잔차 히스토그램
        ax2.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residuals Distribution')
        
        # 3. Q-Q 플롯 (잔차의 정규성)
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot of Residuals')
        
        # 4. Scale-Location 플롯
        standardized_residuals = np.sqrt(np.abs(residuals / np.std(residuals)))
        ax4.scatter(predicted, standardized_residuals, alpha=0.6)
        ax4.set_xlabel('Predicted Values')
        ax4.set_ylabel('√|Standardized Residuals|')
        ax4.set_title('Scale-Location Plot')
        
        plt.tight_layout()
        
        # 잔차 분석 통계
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'shapiro_p': stats.shapiro(residuals[:5000])[1]
        }
        
        return {
            'figure': fig,
            'residual_statistics': residual_stats,
            'success': True
        }
    
    def _create_regression_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """회귀 분석 플롯 생성"""
        x_col = config['x_column']
        y_col = config['y_column']
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 회귀 플롯 생성
        sns.regplot(data=data, x=x_col, y=y_col, ax=ax, 
                   scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
        
        # 신뢰구간 추가 (옵션)
        if config.get('show_confidence_interval', True):
            ax.fill_between(data[x_col].sort_values(), 
                           *self._calculate_confidence_interval(data, x_col, y_col),
                           alpha=0.2, color='blue', label='95% CI')
        
        ax.set_title(f'Regression Plot: {x_col} vs {y_col}')
        
        # 회귀 통계 계산
        slope, intercept, r_value, p_value, std_err = stats.linregress(data[x_col], data[y_col])
        
        # 통계 정보 텍스트 추가
        stats_text = f'R² = {r_value**2:.3f}\np-value = {p_value:.4f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                bbox=dict(boxstyle="round", facecolor='white'))
        
        plt.tight_layout()
        
        return {
            'figure': fig,
            'regression_stats': {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_error': std_err
            },
            'success': True
        }
    
    def _create_anova_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """ANOVA 분석 플롯 생성"""
        group_col = config['group_column']
        value_col = config['value_column']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 그룹별 박스플롯
        sns.boxplot(data=data, x=group_col, y=value_col, ax=ax1)
        ax1.set_title(f'Box Plot by {group_col}')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 그룹별 바이올린 플롯
        sns.violinplot(data=data, x=group_col, y=value_col, ax=ax2)
        ax2.set_title(f'Violin Plot by {group_col}')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 그룹별 평균과 오차막대
        group_stats = data.groupby(group_col)[value_col].agg(['mean', 'std', 'count'])
        se = group_stats['std'] / np.sqrt(group_stats['count'])
        
        ax3.errorbar(range(len(group_stats)), group_stats['mean'], 
                    yerr=se, fmt='o', capsize=5, capthick=2)
        ax3.set_xticks(range(len(group_stats)))
        ax3.set_xticklabels(group_stats.index, rotation=45)
        ax3.set_ylabel(f'Mean {value_col}')
        ax3.set_title('Group Means with Standard Error')
        
        # 4. 잔차 플롯 (ANOVA 가정 확인)
        overall_mean = data[value_col].mean()
        residuals = data[value_col] - overall_mean
        ax4.scatter(data.index, residuals, alpha=0.6)
        ax4.axhline(y=0, color='red', linestyle='--')
        ax4.set_xlabel('Observation Index')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residuals Plot')
        
        plt.tight_layout()
        
        return {
            'figure': fig,
            'group_statistics': group_stats.to_dict(),
            'success': True
        }
    
    def _create_interactive_scatter(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """대화형 산점도 생성 (Plotly)"""
        x_col = config['x_column']
        y_col = config['y_column']
        color_col = config.get('color_column')
        size_col = config.get('size_column')
        
        fig = px.scatter(data, 
                        x=x_col, 
                        y=y_col,
                        color=color_col,
                        size=size_col,
                        hover_data=config.get('hover_columns', []),
                        title=f'Interactive Scatter: {x_col} vs {y_col}')
        
        # 레이아웃 설정
        fig.update_layout(
            width=config.get('width', 800),
            height=config.get('height', 600),
            template='plotly_white'
        )
        
        return {
            'figure': fig,
            'plot_type': 'interactive',
            'success': True
        }
    
    def _create_bootstrap_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """부트스트랩 분석 플롯 생성"""
        column = config['column']
        n_bootstrap = config.get('n_bootstrap', 1000)
        statistic = config.get('statistic', 'mean')
        
        # 부트스트랩 샘플링
        original_data = data[column].dropna()
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(original_data, size=len(original_data), replace=True)
            if statistic == 'mean':
                bootstrap_stats.append(np.mean(bootstrap_sample))
            elif statistic == 'median':
                bootstrap_stats.append(np.median(bootstrap_sample))
            elif statistic == 'std':
                bootstrap_stats.append(np.std(bootstrap_sample))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 부트스트랩 분포 히스토그램
        ax1.hist(bootstrap_stats, bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(bootstrap_stats), color='red', linestyle='--', 
                   label=f'Bootstrap {statistic}: {np.mean(bootstrap_stats):.3f}')
        ax1.set_xlabel(f'Bootstrap {statistic}')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Bootstrap Distribution of {statistic}')
        ax1.legend()
        
        # 신뢰구간 시각화
        ci_lower = np.percentile(bootstrap_stats, 2.5)
        ci_upper = np.percentile(bootstrap_stats, 97.5)
        
        ax2.hist(bootstrap_stats, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(ci_lower, color='green', linestyle='--', label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
        ax2.axvline(ci_upper, color='green', linestyle='--')
        ax2.fill_between([ci_lower, ci_upper], [0, 0], [ax2.get_ylim()[1], ax2.get_ylim()[1]], 
                        alpha=0.3, color='green')
        ax2.set_xlabel(f'Bootstrap {statistic}')
        ax2.set_ylabel('Frequency')
        ax2.set_title('95% Confidence Interval')
        ax2.legend()
        
        plt.tight_layout()
        
        return {
            'figure': fig,
            'bootstrap_statistics': {
                'mean': np.mean(bootstrap_stats),
                'std': np.std(bootstrap_stats),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            },
            'success': True
        }
    
    def _create_ttest_dashboard(self, data: pd.DataFrame, results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """T-검정 결과 대시보드"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Group Distributions', 'Box Plot Comparison', 
                          'Effect Size Visualization', 'Statistical Summary'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                  [{"secondary_y": False}, {"type": "table"}]]
        )
        
        # 그룹별 분포 비교 (히스토그램)
        group1_data = results.get('group1_data', [])
        group2_data = results.get('group2_data', [])
        
        fig.add_trace(
            go.Histogram(x=group1_data, name='Group 1', opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=group2_data, name='Group 2', opacity=0.7),
            row=1, col=1
        )
        
        # 박스플롯 비교
        fig.add_trace(
            go.Box(y=group1_data, name='Group 1'),
            row=1, col=2
        )
        fig.add_trace(
            go.Box(y=group2_data, name='Group 2'),
            row=1, col=2
        )
        
        # 효과 크기 시각화
        effect_size = results.get('effect_size', 0)
        fig.add_trace(
            go.Bar(x=['Effect Size'], y=[effect_size], name='Cohen\'s d'),
            row=2, col=1
        )
        
        # 통계 요약 테이블
        summary_data = [
            ['Statistic', 'Value'],
            ['t-statistic', f"{results.get('t_statistic', 0):.4f}"],
            ['p-value', f"{results.get('p_value', 0):.4f}"],
            ['Effect Size (Cohen\'s d)', f"{effect_size:.4f}"],
            ['95% CI Lower', f"{results.get('ci_lower', 0):.4f}"],
            ['95% CI Upper', f"{results.get('ci_upper', 0):.4f}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=summary_data[0]),
                cells=dict(values=list(zip(*summary_data[1:])))
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="T-Test Analysis Dashboard")
        
        return {
            'dashboard': fig,
            'dashboard_type': 'ttest',
            'success': True
        }
    
    def _apply_style_config(self, style_config: Dict[str, Any]) -> None:
        """스타일 설정 적용"""
        if 'style' in style_config:
            sns.set_style(style_config['style'])
        
        if 'palette' in style_config:
            sns.set_palette(style_config['palette'])
        
        if 'figure_size' in style_config:
            plt.rcParams['figure.figsize'] = style_config['figure_size']
    
    def _find_strongest_correlations(self, corr_matrix: pd.DataFrame) -> List[Dict[str, Any]]:
        """가장 강한 상관관계 찾기"""
        # 대각선과 상삼각 마스크
        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        masked_corr = corr_matrix.where(mask)
        
        # 절댓값 기준으로 정렬
        correlations = []
        for col in masked_corr.columns:
            for idx in masked_corr.index:
                if not pd.isna(masked_corr.loc[idx, col]):
                    correlations.append({
                        'var1': idx,
                        'var2': col,
                        'correlation': masked_corr.loc[idx, col],
                        'abs_correlation': abs(masked_corr.loc[idx, col])
                    })
        
        # 절댓값 기준으로 정렬하여 상위 5개 반환
        correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        return correlations[:5]
    
    def _calculate_confidence_interval(self, data: pd.DataFrame, x_col: str, y_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """회귀선의 신뢰구간 계산"""
        from sklearn.linear_model import LinearRegression
        
        X = data[x_col].values.reshape(-1, 1)
        y = data[y_col].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # 예측값과 잔차 계산
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = np.mean(residuals**2)
        
        # 신뢰구간 계산 (간단한 버전)
        std_error = np.sqrt(mse)
        ci_lower = y_pred - 1.96 * std_error
        ci_upper = y_pred + 1.96 * std_error
        
        return ci_lower, ci_upper
    
    def get_available_plot_types(self) -> List[str]:
        """사용 가능한 플롯 유형 목록 반환"""
        return list(self.plot_generators.keys())
    
    def get_plot_requirements(self, plot_type: str) -> Dict[str, Any]:
        """특정 플롯 유형의 요구사항 반환"""
        requirements = {
            'histogram': {
                'required_config': ['column'],
                'optional_config': ['bins', 'show_normal_curve'],
                'data_requirements': 'numeric column'
            },
            'boxplot': {
                'required_config': ['y_column'],
                'optional_config': ['x_column'],
                'data_requirements': 'numeric column for y, categorical for x (optional)'
            },
            'scatterplot': {
                'required_config': ['x_column', 'y_column'],
                'optional_config': ['color_column', 'show_regression_line'],
                'data_requirements': 'two numeric columns'
            },
            'correlation_matrix': {
                'required_config': [],
                'optional_config': ['columns', 'mask_upper'],
                'data_requirements': 'multiple numeric columns'
            }
        }
        
        return requirements.get(plot_type, {
            'required_config': [],
            'optional_config': [],
            'data_requirements': 'varies by plot type'
        })

    def _create_distribution_comparison(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """분포 비교 플롯 생성"""
        columns = config.get('columns', [])
        if not columns:
            columns = data.select_dtypes(include=[np.number]).columns[:3]
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        for col in columns:
            if col in data.columns:
                ax.hist(data[col].dropna(), alpha=0.6, label=col, bins=30)
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution Comparison')
        ax.legend()
        plt.tight_layout()
        
        return {
            'figure': fig,
            'columns_compared': list(columns),
            'success': True
        }

    def _create_probability_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """확률 플롯 생성"""
        column = config.get('column')
        if not column or column not in data.columns:
            raise ValueError("유효한 컬럼을 지정해야 합니다.")
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Q-Q 플롯
        stats.probplot(data[column].dropna(), dist="norm", plot=ax)
        ax.set_title(f'Probability Plot: {column}')
        plt.tight_layout()
        
        return {
            'figure': fig,
            'column': column,
            'success': True
        }

    def _create_density_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """밀도 플롯 생성"""
        column = config.get('column')
        if not column or column not in data.columns:
            raise ValueError("유효한 컬럼을 지정해야 합니다.")
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 밀도 플롯
        data[column].dropna().plot.density(ax=ax)
        ax.set_title(f'Density Plot: {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Density')
        plt.tight_layout()
        
        return {
            'figure': fig,
            'column': column,
            'success': True
        }

    def _create_power_analysis_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """검정력 분석 플롯 생성"""
        effect_sizes = np.arange(0.1, 2.0, 0.1)
        sample_sizes = config.get('sample_sizes', [10, 20, 30, 50, 100])
        alpha = config.get('alpha', 0.05)
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        for n in sample_sizes:
            powers = []
            for effect_size in effect_sizes:
                # 간단한 검정력 계산 (t-test 기준)
                power = 1 - stats.t.cdf(stats.t.ppf(1-alpha/2, n-1), n-1, effect_size*np.sqrt(n))
                powers.append(power)
            
            ax.plot(effect_sizes, powers, label=f'n={n}')
        
        ax.set_xlabel('Effect Size')
        ax.set_ylabel('Power')
        ax.set_title('Power Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return {
            'figure': fig,
            'sample_sizes': sample_sizes,
            'alpha': alpha,
            'success': True
        }

    def _create_effect_size_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """효과 크기 플롯 생성"""
        group_col = config.get('group_column')
        value_col = config.get('value_column')
        
        if not group_col or not value_col:
            raise ValueError("그룹 컬럼과 값 컬럼을 지정해야 합니다.")
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        groups = data[group_col].unique()
        if len(groups) == 2:
            # Cohen's d 계산
            group1 = data[data[group_col] == groups[0]][value_col]
            group2 = data[data[group_col] == groups[1]][value_col]
            
            pooled_std = np.sqrt(((len(group1)-1)*group1.var() + (len(group2)-1)*group2.var()) / (len(group1)+len(group2)-2))
            cohens_d = (group1.mean() - group2.mean()) / pooled_std
            
            # 효과 크기 시각화
            ax.bar(['Cohen\'s d'], [abs(cohens_d)])
            ax.set_ylabel('Effect Size')
            ax.set_title(f'Effect Size: {cohens_d:.3f}')
            
            # 해석 기준선
            ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Small (0.2)')
            ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium (0.5)')
            ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Large (0.8)')
            ax.legend()
        
        plt.tight_layout()
        
        return {
            'figure': fig,
            'effect_size': cohens_d if len(groups) == 2 else None,
            'success': True
        }

    def _create_interactive_histogram(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """대화형 히스토그램 생성 (plotly 사용)"""
        column = config.get('column')
        if not column or column not in data.columns:
            raise ValueError("유효한 컬럼을 지정해야 합니다.")
        
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[go.Histogram(x=data[column].dropna(), name=column)])
            fig.update_layout(
                title=f'Interactive Histogram: {column}',
                xaxis_title=column,
                yaxis_title='Frequency'
            )
            
            return {
                'figure': fig,
                'column': column,
                'interactive': True,
                'success': True
            }
        except ImportError:
            # plotly가 없으면 일반 히스토그램으로 대체
            return self._create_histogram(data, config)

    def _create_interactive_boxplot(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """대화형 박스플롯 생성 (plotly 사용)"""
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        
        try:
            import plotly.graph_objects as go
            
            if x_col and y_col:
                fig = go.Figure()
                for group in data[x_col].unique():
                    group_data = data[data[x_col] == group][y_col]
                    fig.add_trace(go.Box(y=group_data, name=str(group)))
                
                fig.update_layout(
                    title=f'Interactive Box Plot: {y_col} by {x_col}',
                    xaxis_title=x_col,
                    yaxis_title=y_col
                )
            else:
                column = y_col or x_col
                fig = go.Figure(data=[go.Box(y=data[column].dropna(), name=column)])
                fig.update_layout(
                    title=f'Interactive Box Plot: {column}',
                    yaxis_title=column
                )
            
            return {
                'figure': fig,
                'interactive': True,
                'success': True
            }
        except ImportError:
            # plotly가 없으면 일반 박스플롯으로 대체
            return self._create_boxplot(data, config) 