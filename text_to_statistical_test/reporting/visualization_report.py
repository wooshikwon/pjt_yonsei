"""
VisualizationReport: 시각화가 포함된 보고서 생성

통계 분석 결과와 함께 관련된 시각화를 자동으로 생성하고
이를 통합한 시각적 보고서를 만드는 기능을 제공합니다.
"""

import base64
import io
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import pandas as pd
import numpy as np

# 시각화 라이브러리
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class VisualizationReport:
    """
    시각화가 포함된 분석 보고서 생성
    
    통계 분석 결과에 따라 적절한 시각화를 자동으로 생성하고,
    이를 포함한 종합적인 보고서를 만듭니다.
    """
    
    def __init__(self, output_dir: str = "output_results/visualizations"):
        """
        VisualizationReport 초기화
        
        Args:
            output_dir: 시각화 파일 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # 사용 가능한 시각화 라이브러리 확인
        self.viz_engines = {
            'matplotlib': MATPLOTLIB_AVAILABLE,
            'plotly': PLOTLY_AVAILABLE
        }
        
        self.logger.info(f"시각화 엔진 상태: {self.viz_engines}")
        
        # 시각화 설정
        self._setup_visualization_settings()
    
    def _setup_visualization_settings(self):
        """시각화 기본 설정"""
        if MATPLOTLIB_AVAILABLE:
            # Matplotlib 설정
            plt.style.use('default')
            plt.rcParams['figure.figsize'] = (10, 6)
            plt.rcParams['font.size'] = 11
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
            
        if PLOTLY_AVAILABLE:
            # Plotly 기본 템플릿 설정
            self.plotly_template = 'plotly_white'
    
    def create_comprehensive_report(self, analysis_results: List[Dict],
                                  data: pd.DataFrame = None,
                                  session_metadata: Dict = None) -> str:
        """
        종합적인 시각화 보고서 생성
        
        Args:
            analysis_results: 분석 결과 리스트
            data: 원본 데이터프레임
            session_metadata: 세션 메타데이터
            
        Returns:
            str: 생성된 HTML 보고서 파일 경로
        """
        self.logger.info("종합 시각화 보고서 생성 시작")
        
        # 시각화 생성
        visualizations = self._generate_visualizations(analysis_results, data)
        
        # HTML 보고서 생성
        html_content = self._create_html_report(
            analysis_results, visualizations, session_metadata
        )
        
        # 파일 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"visualization_report_{timestamp}.html"
        report_path = self.output_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"시각화 보고서 생성 완료: {report_path}")
        return str(report_path)
    
    def _generate_visualizations(self, analysis_results: List[Dict],
                               data: pd.DataFrame = None) -> Dict[str, str]:
        """
        분석 결과에 따른 시각화 생성
        
        Args:
            analysis_results: 분석 결과 리스트
            data: 원본 데이터
            
        Returns:
            Dict[str, str]: 시각화 종류별 base64 인코딩된 이미지
        """
        visualizations = {}
        
        try:
            # 1. 분석 결과 요약 차트
            if analysis_results:
                visualizations['summary_chart'] = self._create_results_summary_chart(analysis_results)
            
            # 2. 데이터 기반 시각화 (데이터가 있는 경우)
            if data is not None:
                visualizations.update(self._create_data_visualizations(data))
            
            # 3. 통계 테스트별 시각화
            for i, result in enumerate(analysis_results):
                analysis_type = result.get('metadata', {}).get('analysis_type', 'unknown')
                viz_name = f"analysis_{i+1}_{analysis_type}"
                visualizations[viz_name] = self._create_analysis_specific_visualization(result)
        
        except Exception as e:
            self.logger.error(f"시각화 생성 중 오류: {e}")
        
        return visualizations
    
    def _create_results_summary_chart(self, analysis_results: List[Dict]) -> str:
        """분석 결과 요약 차트 생성"""
        try:
            # 분석 유형별 집계
            analysis_types = {}
            significant_counts = {'significant': 0, 'not_significant': 0}
            
            for result in analysis_results:
                # 분석 유형 집계
                analysis_type = result.get('metadata', {}).get('analysis_type', 'unknown')
                analysis_types[analysis_type] = analysis_types.get(analysis_type, 0) + 1
                
                # 유의성 집계
                is_significant = result.get('summary', {}).get('significant', False)
                if is_significant:
                    significant_counts['significant'] += 1
                else:
                    significant_counts['not_significant'] += 1
            
            if PLOTLY_AVAILABLE:
                return self._create_plotly_summary_chart(analysis_types, significant_counts)
            elif MATPLOTLIB_AVAILABLE:
                return self._create_matplotlib_summary_chart(analysis_types, significant_counts)
            else:
                return self._create_text_summary(analysis_types, significant_counts)
                
        except Exception as e:
            self.logger.error(f"요약 차트 생성 실패: {e}")
            return ""
    
    def _create_plotly_summary_chart(self, analysis_types: Dict, 
                                   significant_counts: Dict) -> str:
        """Plotly를 사용한 요약 차트 생성"""
        # 서브플롯 생성
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('분석 유형별 분포', '유의성 결과'),
            specs=[[{"type": "pie"}, {"type": "pie"}]]
        )
        
        # 분석 유형 파이 차트
        fig.add_trace(
            go.Pie(
                labels=list(analysis_types.keys()),
                values=list(analysis_types.values()),
                name="분석 유형"
            ),
            row=1, col=1
        )
        
        # 유의성 파이 차트
        fig.add_trace(
            go.Pie(
                labels=['유의함', '유의하지 않음'],
                values=[significant_counts['significant'], significant_counts['not_significant']],
                name="유의성",
                marker_colors=['#2ecc71', '#e74c3c']
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="분석 결과 요약",
            template=self.plotly_template,
            height=400
        )
        
        # HTML로 변환 후 base64 인코딩
        html_str = pyo.plot(fig, output_type='div', include_plotlyjs=True)
        return base64.b64encode(html_str.encode()).decode()
    
    def _create_matplotlib_summary_chart(self, analysis_types: Dict, 
                                       significant_counts: Dict) -> str:
        """Matplotlib를 사용한 요약 차트 생성"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 분석 유형 파이 차트
        if analysis_types:
            ax1.pie(analysis_types.values(), labels=analysis_types.keys(), autopct='%1.1f%%')
            ax1.set_title('분석 유형별 분포')
        
        # 유의성 파이 차트
        if sum(significant_counts.values()) > 0:
            ax2.pie(
                [significant_counts['significant'], significant_counts['not_significant']],
                labels=['유의함', '유의하지 않음'],
                colors=['#2ecc71', '#e74c3c'],
                autopct='%1.1f%%'
            )
            ax2.set_title('유의성 결과')
        
        plt.tight_layout()
        
        # 이미지를 base64로 인코딩
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def _create_text_summary(self, analysis_types: Dict, significant_counts: Dict) -> str:
        """텍스트 기반 요약 (시각화 라이브러리가 없는 경우)"""
        summary = f"""
        <div class="text-summary">
            <h3>분석 결과 요약</h3>
            <p><strong>분석 유형별 분포:</strong></p>
            <ul>
        """
        
        for analysis_type, count in analysis_types.items():
            summary += f"<li>{analysis_type}: {count}개</li>"
        
        summary += f"""
            </ul>
            <p><strong>유의성 결과:</strong></p>
            <ul>
                <li>유의함: {significant_counts['significant']}개</li>
                <li>유의하지 않음: {significant_counts['not_significant']}개</li>
            </ul>
        </div>
        """
        
        return base64.b64encode(summary.encode()).decode()
    
    def _create_data_visualizations(self, data: pd.DataFrame) -> Dict[str, str]:
        """데이터 기반 시각화 생성"""
        visualizations = {}
        
        try:
            # 수치형 변수 히스토그램
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                visualizations['data_distribution'] = self._create_distribution_plots(data, numeric_cols)
            
            # 상관관계 히트맵
            if len(numeric_cols) > 1:
                visualizations['correlation_heatmap'] = self._create_correlation_heatmap(data, numeric_cols)
            
            # 범주형 변수 분포
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                visualizations['categorical_distribution'] = self._create_categorical_plots(data, categorical_cols)
        
        except Exception as e:
            self.logger.error(f"데이터 시각화 생성 실패: {e}")
        
        return visualizations
    
    def _create_distribution_plots(self, data: pd.DataFrame, numeric_cols: List[str]) -> str:
        """수치형 변수 분포 시각화"""
        if MATPLOTLIB_AVAILABLE:
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols[:9]):  # 최대 9개
                if i < len(axes):
                    data[col].hist(bins=30, ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'{col} 분포')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('빈도')
            
            # 빈 subplot 제거
            for i in range(len(numeric_cols), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
        
        return ""
    
    def _create_correlation_heatmap(self, data: pd.DataFrame, numeric_cols: List[str]) -> str:
        """상관관계 히트맵 생성"""
        if MATPLOTLIB_AVAILABLE and len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            
            correlation_matrix = data[numeric_cols].corr()
            
            if MATPLOTLIB_AVAILABLE:
                sns.heatmap(
                    correlation_matrix, 
                    annot=True, 
                    cmap='coolwarm', 
                    center=0,
                    square=True,
                    fmt='.2f'
                )
                plt.title('변수 간 상관관계')
                plt.tight_layout()
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                return image_base64
        
        return ""
    
    def _create_categorical_plots(self, data: pd.DataFrame, categorical_cols: List[str]) -> str:
        """범주형 변수 분포 시각화"""
        if MATPLOTLIB_AVAILABLE and len(categorical_cols) > 0:
            n_cols = min(2, len(categorical_cols))
            n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(categorical_cols[:6]):  # 최대 6개
                if i < len(axes):
                    value_counts = data[col].value_counts().head(10)  # 상위 10개만
                    value_counts.plot(kind='bar', ax=axes[i])
                    axes[i].set_title(f'{col} 분포')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('빈도')
                    axes[i].tick_params(axis='x', rotation=45)
            
            # 빈 subplot 제거
            for i in range(len(categorical_cols), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
        
        return ""
    
    def _create_analysis_specific_visualization(self, result: Dict) -> str:
        """분석 유형별 특화 시각화"""
        analysis_type = result.get('metadata', {}).get('analysis_type', 'unknown')
        statistical_results = result.get('statistical_results', {})
        
        if 'p_value' in statistical_results:
            return self._create_p_value_visualization(statistical_results)
        
        return ""
    
    def _create_p_value_visualization(self, stats: Dict) -> str:
        """p-value 시각화"""
        if not MATPLOTLIB_AVAILABLE:
            return ""
        
        try:
            p_value = float(stats.get('p_value', 0))
            
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # p-value 막대 그래프
            colors = ['red' if p_value < 0.05 else 'blue']
            bars = ax.bar(['p-value'], [p_value], color=colors, alpha=0.7)
            
            # 유의수준 선
            ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='α = 0.05')
            
            ax.set_ylabel('p-value')
            ax.set_title(f'통계적 유의성 (p = {p_value:.4f})')
            ax.set_ylim(0, max(0.1, p_value * 1.2))
            ax.legend()
            
            # p-value 값 표시
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(0.005, height*0.05),
                       f'{height:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            self.logger.error(f"p-value 시각화 생성 실패: {e}")
            return ""
    
    def _create_html_report(self, analysis_results: List[Dict],
                          visualizations: Dict[str, str],
                          session_metadata: Dict = None) -> str:
        """HTML 보고서 생성"""
        
        session_id = session_metadata.get('session_id', 'Unknown') if session_metadata else 'Unknown'
        created_at = session_metadata.get('created_at', datetime.now().isoformat()) if session_metadata else datetime.now().isoformat()
        
        html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>시각화 분석 보고서 - {session_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .content {{
            padding: 40px;
        }}
        .viz-section {{
            margin-bottom: 40px;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            background-color: #fafafa;
        }}
        .viz-section h2 {{
            color: #333;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .viz-image {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .analysis-summary {{
            background-color: #e8f4fd;
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid #2196f3;
            margin-bottom: 30px;
        }}
        .footer {{
            background-color: #f5f5f5;
            padding: 20px 40px;
            text-align: center;
            color: #666;
            border-top: 1px solid #ddd;
        }}
        .no-viz {{
            text-align: center;
            color: #666;
            font-style: italic;
            padding: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 시각화 분석 보고서</h1>
            <p>세션 ID: {session_id}</p>
            <p>생성일시: {created_at}</p>
        </div>
        
        <div class="content">
            <div class="analysis-summary">
                <h2>📋 분석 요약</h2>
                <p><strong>총 분석 수:</strong> {len(analysis_results)}개</p>
                <p><strong>생성된 시각화:</strong> {len(visualizations)}개</p>
                <p><strong>시각화 엔진:</strong> {', '.join([k for k, v in self.viz_engines.items() if v])}</p>
            </div>
"""
        
        # 시각화 추가
        if visualizations:
            for viz_name, viz_data in visualizations.items():
                if viz_data:
                    # base64 데이터가 HTML인지 이미지인지 확인
                    try:
                        decoded_data = base64.b64decode(viz_data).decode()
                        if decoded_data.startswith('<'):
                            # HTML 데이터 (Plotly)
                            html_content += f"""
            <div class="viz-section">
                <h2>📈 {viz_name.replace('_', ' ').title()}</h2>
                {decoded_data}
            </div>
"""
                        else:
                            # 텍스트 데이터
                            html_content += f"""
            <div class="viz-section">
                <h2>📈 {viz_name.replace('_', ' ').title()}</h2>
                {decoded_data}
            </div>
"""
                    except:
                        # 이미지 데이터
                        html_content += f"""
            <div class="viz-section">
                <h2>📈 {viz_name.replace('_', ' ').title()}</h2>
                <img src="data:image/png;base64,{viz_data}" alt="{viz_name}" class="viz-image">
            </div>
"""
        else:
            html_content += """
            <div class="viz-section">
                <div class="no-viz">시각화를 생성할 수 없습니다. 시각화 라이브러리를 확인해주세요.</div>
            </div>
"""
        
        html_content += f"""
        </div>
        
        <div class="footer">
            <p>본 보고서는 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}에 자동 생성되었습니다.</p>
            <p>Enhanced RAG 기반 Multi-turn 통계 분석 시스템</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html_content
    
    def save_individual_visualization(self, viz_data: str, viz_name: str, 
                                    format: str = 'png') -> str:
        """개별 시각화를 파일로 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{viz_name}_{timestamp}.{format}"
        filepath = self.output_dir / filename
        
        try:
            if format.lower() in ['png', 'jpg', 'jpeg']:
                # 이미지 파일로 저장
                image_data = base64.b64decode(viz_data)
                with open(filepath, 'wb') as f:
                    f.write(image_data)
            else:
                # 텍스트 파일로 저장
                decoded_data = base64.b64decode(viz_data).decode()
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(decoded_data)
            
            self.logger.info(f"시각화 저장 완료: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"시각화 저장 실패: {e}")
            return "" 