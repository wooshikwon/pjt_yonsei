"""
Visualization Engine

데이터 시각화 및 차트 생성 담당
"""

import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class VisualizationEngine:
    """데이터 시각화 엔진"""
    
    def __init__(self):
        """VisualizationEngine 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 시각화 스타일 설정
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_visualization(self, 
                           data: pd.DataFrame,
                           chart_type: str,
                           **kwargs) -> Optional[str]:
        """
        시각화 생성
        
        Args:
            data: 시각화할 데이터
            chart_type: 차트 타입
            **kwargs: 추가 옵션
            
        Returns:
            생성된 차트 파일 경로
        """
        try:
            if chart_type == 'histogram':
                return self._create_histogram(data, **kwargs)
            elif chart_type == 'boxplot':
                return self._create_boxplot(data, **kwargs)
            elif chart_type == 'scatter':
                return self._create_scatter(data, **kwargs)
            elif chart_type == 'correlation':
                return self._create_correlation_matrix(data, **kwargs)
            else:
                self.logger.warning(f"지원하지 않는 차트 타입: {chart_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"시각화 생성 실패: {e}")
            return None
    
    def _create_histogram(self, data: pd.DataFrame, **kwargs) -> str:
        """히스토그램 생성"""
        column = kwargs.get('column')
        if not column or column not in data.columns:
            raise ValueError("유효한 컬럼명이 필요합니다")
        
        plt.figure(figsize=(10, 6))
        plt.hist(data[column].dropna(), bins=30, alpha=0.7)
        plt.title(f'{column} 분포')
        plt.xlabel(column)
        plt.ylabel('빈도')
        
        filepath = f"histogram_{column}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_boxplot(self, data: pd.DataFrame, **kwargs) -> str:
        """박스플롯 생성"""
        column = kwargs.get('column')
        if not column or column not in data.columns:
            raise ValueError("유효한 컬럼명이 필요합니다")
        
        plt.figure(figsize=(8, 6))
        plt.boxplot(data[column].dropna())
        plt.title(f'{column} 박스플롯')
        plt.ylabel(column)
        
        filepath = f"boxplot_{column}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_scatter(self, data: pd.DataFrame, **kwargs) -> str:
        """산점도 생성"""
        x_col = kwargs.get('x_column')
        y_col = kwargs.get('y_column')
        
        if not x_col or not y_col or x_col not in data.columns or y_col not in data.columns:
            raise ValueError("유효한 x, y 컬럼명이 필요합니다")
        
        plt.figure(figsize=(10, 8))
        plt.scatter(data[x_col], data[y_col], alpha=0.6)
        plt.title(f'{x_col} vs {y_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        
        filepath = f"scatter_{x_col}_{y_col}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_correlation_matrix(self, data: pd.DataFrame, **kwargs) -> str:
        """상관관계 매트릭스 생성"""
        numeric_data = data.select_dtypes(include=['number'])
        
        if numeric_data.empty:
            raise ValueError("수치형 데이터가 없습니다")
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = numeric_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('상관관계 매트릭스')
        
        filepath = "correlation_matrix.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath 