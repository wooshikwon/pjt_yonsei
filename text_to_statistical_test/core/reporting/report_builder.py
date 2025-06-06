"""
Report Builder

보고서 생성 및 구조화 담당
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import json


class ReportBuilder:
    """보고서 생성기"""
    
    def __init__(self):
        """ReportBuilder 초기화"""
        self.logger = logging.getLogger(__name__)
    
    def build_report(self, 
                    analysis_results: Dict[str, Any],
                    interpretation: str,
                    business_insights: str,
                    recommendations: str) -> Dict[str, Any]:
        """
        종합 보고서 생성
        
        Args:
            analysis_results: 분석 결과
            interpretation: 통계적 해석
            business_insights: 비즈니스 인사이트
            recommendations: 권장사항
            
        Returns:
            구조화된 보고서
        """
        try:
            report = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'report_type': 'statistical_analysis',
                    'version': '1.0'
                },
                'executive_summary': {
                    'key_findings': self._extract_key_findings(analysis_results),
                    'business_impact': business_insights,
                    'recommendations': recommendations
                },
                'analysis_details': analysis_results,
                'interpretation': interpretation,
                'appendix': {
                    'methodology': self._get_methodology(analysis_results),
                    'assumptions': self._get_assumptions(analysis_results),
                    'limitations': self._get_limitations(analysis_results)
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"보고서 생성 실패: {e}")
            return {}
    
    def _extract_key_findings(self, analysis_results: Dict[str, Any]) -> List[str]:
        """주요 발견사항 추출"""
        findings = []
        
        if 'test_results' in analysis_results:
            for test_name, result in analysis_results['test_results'].items():
                if isinstance(result, dict) and 'p_value' in result:
                    p_val = result['p_value']
                    if p_val < 0.05:
                        findings.append(f"{test_name}: 통계적으로 유의한 결과 (p={p_val:.4f})")
                    else:
                        findings.append(f"{test_name}: 통계적으로 유의하지 않음 (p={p_val:.4f})")
        
        return findings
    
    def _get_methodology(self, analysis_results: Dict[str, Any]) -> str:
        """분석 방법론 설명"""
        if 'methodology' in analysis_results:
            return analysis_results['methodology']
        return "통계적 검정을 통한 데이터 분석"
    
    def _get_assumptions(self, analysis_results: Dict[str, Any]) -> List[str]:
        """분석 가정사항"""
        if 'assumptions' in analysis_results:
            return analysis_results['assumptions']
        return ["정규성 가정", "독립성 가정", "등분산성 가정"]
    
    def _get_limitations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """분석 한계점"""
        if 'limitations' in analysis_results:
            return analysis_results['limitations']
        return ["표본 크기의 한계", "관찰 연구의 한계", "일반화 가능성의 제한"] 