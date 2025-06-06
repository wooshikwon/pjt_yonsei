"""
Report Generator

보고서 생성을 담당하는 클래스
Agent가 생성한 분석 결과를 다양한 형태의 보고서로 변환
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import json

from .report_builder import ReportBuilder
from .output_formatter import OutputFormatter
from .visualization_engine import VisualizationEngine


class ReportGenerator:
    """
    보고서 생성 클래스
    
    통계 분석 결과를 다양한 형태의 보고서로 생성하며,
    Agent가 요구하는 다양한 출력 형식을 지원합니다.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """ReportGenerator 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 하위 컴포넌트들 초기화
        self.report_builder = ReportBuilder()
        self.output_formatter = OutputFormatter()
        self.visualization_engine = VisualizationEngine()
        
        # 출력 디렉토리 설정
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 보고서 생성 이력
        self.generation_history: List[Dict[str, Any]] = []
        
        self.logger.info("ReportGenerator 초기화 완료")
    
    def generate_report(self,
                       analysis_results: Dict[str, Any],
                       report_config: Optional[Dict[str, Any]] = None,
                       output_format: str = "html") -> Dict[str, Any]:
        """
        보고서 생성
        
        Args:
            analysis_results: 분석 결과 데이터
            report_config: 보고서 설정
            output_format: 출력 형식 (html, pdf, json, markdown)
            
        Returns:
            Dict[str, Any]: 보고서 생성 결과
        """
        try:
            generation_start_time = datetime.now()
            
            # 기본 설정 적용
            config = self._apply_default_config(report_config or {})
            
            # 1. 보고서 구조 생성
            report_structure = self.report_builder.build_report_structure(
                analysis_results, config
            )
            
            # 2. 시각화 생성 (필요시)
            if config.get('include_visualizations', True):
                visualizations = self._generate_visualizations(
                    analysis_results, config
                )
                report_structure['visualizations'] = visualizations
            
            # 3. 포맷팅 및 최종 보고서 생성
            formatted_report = self.output_formatter.format_output(
                report_structure, output_format, config
            )
            
            # 4. 파일 저장
            output_path = self._save_report(formatted_report, output_format, config)
            
            # 5. 생성 이력 기록
            generation_record = {
                'timestamp': generation_start_time.isoformat(),
                'output_format': output_format,
                'output_path': str(output_path),
                'config': config,
                'success': True,
                'generation_time': (datetime.now() - generation_start_time).total_seconds()
            }
            self.generation_history.append(generation_record)
            
            return {
                'success': True,
                'output_path': str(output_path),
                'report_structure': report_structure,
                'generation_info': generation_record
            }
            
        except Exception as e:
            self.logger.error(f"보고서 생성 오류: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'report_generation_error'
            }
    
    def generate_summary_report(self,
                              analysis_results: Dict[str, Any],
                              target_audience: str = "general") -> Dict[str, Any]:
        """
        요약 보고서 생성
        
        Args:
            analysis_results: 분석 결과
            target_audience: 대상 독자 (general, technical, executive)
            
        Returns:
            Dict[str, Any]: 요약 보고서 결과
        """
        try:
            config = {
                'report_type': 'summary',
                'target_audience': target_audience,
                'include_methodology': target_audience == 'technical',
                'include_executive_summary': target_audience == 'executive',
                'include_detailed_statistics': target_audience == 'technical',
                'length': 'short' if target_audience == 'executive' else 'medium'
            }
            
            return self.generate_report(
                analysis_results=analysis_results,
                report_config=config,
                output_format="html"
            )
            
        except Exception as e:
            self.logger.error(f"요약 보고서 생성 오류: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_detailed_report(self,
                               analysis_results: Dict[str, Any],
                               include_code: bool = True,
                               include_data_exploration: bool = True) -> Dict[str, Any]:
        """
        상세 보고서 생성
        
        Args:
            analysis_results: 분석 결과
            include_code: 코드 포함 여부
            include_data_exploration: 데이터 탐색 포함 여부
            
        Returns:
            Dict[str, Any]: 상세 보고서 결과
        """
        try:
            config = {
                'report_type': 'detailed',
                'include_code': include_code,
                'include_data_exploration': include_data_exploration,
                'include_assumptions': True,
                'include_methodology': True,
                'include_interpretation': True,
                'include_recommendations': True,
                'length': 'long'
            }
            
            return self.generate_report(
                analysis_results=analysis_results,
                report_config=config,
                output_format="html"
            )
            
        except Exception as e:
            self.logger.error(f"상세 보고서 생성 오류: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_presentation_slides(self,
                                   analysis_results: Dict[str, Any],
                                   slide_count: int = 10) -> Dict[str, Any]:
        """
        프레젠테이션 슬라이드 생성
        
        Args:
            analysis_results: 분석 결과
            slide_count: 슬라이드 수
            
        Returns:
            Dict[str, Any]: 슬라이드 생성 결과
        """
        try:
            config = {
                'report_type': 'presentation',
                'slide_count': slide_count,
                'focus_on_insights': True,
                'include_visualizations': True,
                'minimize_text': True,
                'include_recommendations': True
            }
            
            return self.generate_report(
                analysis_results=analysis_results,
                report_config=config,
                output_format="html"  # 추후 PowerPoint 형식 지원 가능
            )
            
        except Exception as e:
            self.logger.error(f"프레젠테이션 생성 오류: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_available_formats(self) -> List[str]:
        """지원하는 출력 형식 목록 반환"""
        return ['html', 'json', 'markdown', 'pdf', 'txt']
    
    def get_generation_summary(self) -> Dict[str, Any]:
        """보고서 생성 요약 정보 반환"""
        return {
            'total_reports_generated': len(self.generation_history),
            'successful_generations': sum(1 for g in self.generation_history if g.get('success')),
            'recent_generations': self.generation_history[-5:] if self.generation_history else [],
            'available_formats': self.get_available_formats(),
            'output_directory': str(self.output_dir)
        }
    
    def _apply_default_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """기본 설정 적용"""
        default_config = {
            'report_type': 'standard',
            'include_visualizations': True,
            'include_methodology': True,
            'include_interpretation': True,
            'include_recommendations': False,
            'target_audience': 'general',
            'length': 'medium',
            'style': 'professional'
        }
        
        # 사용자 설정으로 기본값 업데이트
        merged_config = default_config.copy()
        merged_config.update(config)
        
        return merged_config
    
    def _generate_visualizations(self,
                               analysis_results: Dict[str, Any],
                               config: Dict[str, Any]) -> Dict[str, Any]:
        """시각화 생성"""
        try:
            visualization_config = {
                'style': config.get('style', 'professional'),
                'target_audience': config.get('target_audience', 'general'),
                'include_statistical_plots': config.get('include_methodology', True)
            }
            
            visualizations = self.visualization_engine.create_visualizations(
                analysis_results, visualization_config
            )
            
            return visualizations
            
        except Exception as e:
            self.logger.warning(f"시각화 생성 중 오류: {e}")
            return {}
    
    def _save_report(self,
                    formatted_report: Dict[str, Any],
                    output_format: str,
                    config: Dict[str, Any]) -> Path:
        """보고서 파일 저장"""
        try:
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_type = config.get('report_type', 'standard')
            filename = f"report_{report_type}_{timestamp}.{output_format}"
            
            output_path = self.output_dir / filename
            
            # 형식에 따른 저장
            if output_format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(formatted_report, f, ensure_ascii=False, indent=2)
            elif output_format in ['html', 'markdown', 'txt']:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_report.get('content', ''))
            else:
                # 기타 형식은 바이너리로 저장
                with open(output_path, 'wb') as f:
                    f.write(formatted_report.get('binary_content', b''))
            
            self.logger.info(f"보고서 저장 완료: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"보고서 저장 오류: {e}")
            # 임시 경로 반환
            return self.output_dir / f"report_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    def _create_report_metadata(self,
                              analysis_results: Dict[str, Any],
                              config: Dict[str, Any]) -> Dict[str, Any]:
        """보고서 메타데이터 생성"""
        return {
            'generated_at': datetime.now().isoformat(),
            'generator_version': '1.0.0',
            'report_type': config.get('report_type', 'standard'),
            'target_audience': config.get('target_audience', 'general'),
            'analysis_summary': {
                'analysis_type': analysis_results.get('analysis_type', 'unknown'),
                'data_points': analysis_results.get('sample_size', 0),
                'variables_analyzed': len(analysis_results.get('variables', [])),
                'success': analysis_results.get('success', False)
            },
            'configuration': config
        } 