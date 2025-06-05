"""
ReportGenerator: Enhanced RAG 기반 비즈니스 컨텍스트 인식 분석 결과 보고서 생성

비즈니스 도메인 지식과 DB 스키마 구조를 활용한 지능형 보고서 생성
- 자연어 요청 기반 분석 과정 추적
- 비즈니스 인사이트 포함 결과 해석
- 다중턴 대화 세션 히스토리 통합
- TemplateManager와 VisualizationReport 완전 통합
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json
import uuid
from dataclasses import dataclass, asdict

from .template_manager import TemplateManager
from .visualization_report import VisualizationReport


@dataclass
class AnalysisSession:
    """분석 세션 정보"""
    session_id: str
    natural_language_request: str
    business_domain: str
    data_context: Dict[str, Any]
    schema_context: Dict[str, Any]
    recommended_methods: List[Dict[str, Any]]
    selected_method: Dict[str, Any]
    analysis_results: Dict[str, Any]
    business_insights: List[str]
    created_at: str
    status: str = "completed"


@dataclass
class BusinessContext:
    """비즈니스 컨텍스트 정보"""
    domain: str
    terminology: Dict[str, str]
    guidelines: List[str]
    constraints: List[str]
    success_metrics: List[str]


@dataclass
class SchemaContext:
    """DB 스키마 컨텍스트 정보"""
    primary_table: str
    key_columns: Dict[str, Dict[str, str]]
    relationships: List[Dict[str, str]]
    constraints: Dict[str, List[str]]
    analytical_patterns: Dict[str, Any]


class ReportGenerator:
    """
    Enhanced RAG 기반 비즈니스 컨텍스트 인식 보고서 생성기
    
    자연어 요청부터 최종 분석 결과까지의 전체 과정을 추적하고
    비즈니스 도메인 지식과 DB 스키마 구조를 활용한 인사이트를 포함한
    종합적인 분석 보고서를 생성합니다.
    """
    
    def __init__(self, output_directory: str = "output_results/reports"):
        """
        ReportGenerator 초기화
        
        Args:
            output_directory: 결과 보고서 저장 디렉토리
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # 통합 모듈 초기화
        self.template_manager = TemplateManager()
        self.viz_reporter = VisualizationReport()
        
        # 세션 관리
        self.current_session: Optional[AnalysisSession] = None
        self.session_history: Dict[str, AnalysisSession] = {}
    
    def start_analysis_session(self, 
                             natural_language_request: str,
                             business_context: BusinessContext,
                             schema_context: SchemaContext,
                             data_context: Dict[str, Any]) -> str:
        """
        새로운 분석 세션 시작
        
        Args:
            natural_language_request: 사용자의 자연어 분석 요청
            business_context: 비즈니스 컨텍스트 정보
            schema_context: DB 스키마 컨텍스트 정보
            data_context: 데이터 컨텍스트 정보
            
        Returns:
            str: 생성된 세션 ID
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        self.current_session = AnalysisSession(
            session_id=session_id,
            natural_language_request=natural_language_request,
            business_domain=business_context.domain,
            data_context=data_context,
            schema_context=asdict(schema_context),
            recommended_methods=[],
            selected_method={},
            analysis_results={},
            business_insights=[],
            created_at=datetime.now().isoformat()
        )
        
        self.session_history[session_id] = self.current_session
        
        self.logger.info(f"분석 세션 시작: {session_id}")
        self.logger.info(f"자연어 요청: {natural_language_request}")
        self.logger.info(f"비즈니스 도메인: {business_context.domain}")
        
        return session_id
    
    def update_recommended_methods(self, methods: List[Dict[str, Any]]):
        """AI 추천 방법 업데이트"""
        if self.current_session:
            self.current_session.recommended_methods = methods
            self.logger.info(f"추천 방법 업데이트: {len(methods)}개")
    
    def update_selected_method(self, method: Dict[str, Any]):
        """사용자 선택 방법 업데이트"""
        if self.current_session:
            self.current_session.selected_method = method
            self.logger.info(f"선택된 방법: {method.get('name', 'Unknown')}")
    
    def update_analysis_results(self, results: Dict[str, Any]):
        """분석 결과 업데이트"""
        if self.current_session:
            self.current_session.analysis_results = results
            
            # 비즈니스 인사이트 자동 생성
            insights = self._generate_business_insights(results)
            self.current_session.business_insights = insights
            
            self.logger.info("분석 결과 및 비즈니스 인사이트 업데이트 완료")
    
    def generate_comprehensive_report(self, 
                                    session_id: str = None,
                                    include_visualizations: bool = True,
                                    include_process_details: bool = True,
                                    output_formats: List[str] = None) -> Dict[str, str]:
        """
        종합 분석 보고서 생성
        
        Args:
            session_id: 보고서 생성할 세션 ID (None이면 현재 세션)
            include_visualizations: 시각화 포함 여부
            include_process_details: 분석 과정 상세 포함 여부
            output_formats: 출력 형식 리스트 ['html', 'pdf', 'markdown']
            
        Returns:
            Dict[str, str]: 생성된 보고서 파일 경로들 {형식: 경로}
        """
        if output_formats is None:
            output_formats = ['html', 'markdown']
        
        # 세션 선택
        session = self._get_session(session_id)
        if not session:
            raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
        
        self.logger.info(f"종합 보고서 생성 시작: {session.session_id}")
        
        # 보고서 데이터 준비
        report_data = self._prepare_comprehensive_report_data(
            session, include_process_details
        )
        
        generated_reports = {}
        
        try:
            # 1. 시각화 보고서 생성
            if include_visualizations and session.analysis_results:
                viz_path = self._generate_visualization_section(session)
                if viz_path:
                    report_data['visualization_path'] = viz_path
            
            # 2. 각 형식별 보고서 생성
            for format_type in output_formats:
                if format_type == 'html':
                    html_path = self._generate_html_report(session, report_data)
                    if html_path:
                        generated_reports['html'] = html_path
                
                elif format_type == 'markdown':
                    md_path = self._generate_markdown_report(session, report_data)
                    if md_path:
                        generated_reports['markdown'] = md_path
                
                elif format_type == 'pdf':
                    pdf_path = self._generate_pdf_report(session, report_data)
                    if pdf_path:
                        generated_reports['pdf'] = pdf_path
            
            # 3. 요약 대시보드 생성
            dashboard_path = self._generate_dashboard(session, report_data)
            if dashboard_path:
                generated_reports['dashboard'] = dashboard_path
            
            self.logger.info(f"종합 보고서 생성 완료: {len(generated_reports)}개 파일")
            
        except Exception as e:
            self.logger.error(f"보고서 생성 중 오류: {e}")
            raise
        
        return generated_reports
    
    def _prepare_comprehensive_report_data(self, 
                                         session: AnalysisSession,
                                         include_process_details: bool) -> Dict[str, Any]:
        """종합 보고서 데이터 준비"""
        
        # 기본 세션 정보
        data = {
            'session_id': session.session_id,
            'analysis_date': datetime.fromisoformat(session.created_at).strftime('%Y년 %m월 %d일'),
            'natural_language_request': session.natural_language_request,
            'business_domain': session.business_domain,
            'status': session.status,
            
            # 데이터 컨텍스트
            'data_shape': session.data_context.get('shape', '정보 없음'),
            'data_columns': session.data_context.get('columns', []),
            'data_summary': session.data_context.get('summary', {}),
            
            # 분석 방법 정보
            'total_recommended_methods': len(session.recommended_methods),
            'selected_method_name': session.selected_method.get('name', '선택되지 않음'),
            'selected_method_reason': session.selected_method.get('reason', ''),
            'selected_method_assumptions': session.selected_method.get('assumptions', []),
            
            # 분석 결과
            'analysis_success': bool(session.analysis_results.get('success', False)),
            'statistical_significance': session.analysis_results.get('significance', {}),
            'effect_size': session.analysis_results.get('effect_size', {}),
            'confidence_intervals': session.analysis_results.get('confidence_intervals', {}),
            
            # 비즈니스 인사이트
            'business_insights': session.business_insights,
            'key_findings': self._extract_key_findings(session),
            'practical_implications': self._extract_practical_implications(session),
            'recommendations': self._generate_recommendations(session),
            
            # 스키마 컨텍스트
            'schema_context': session.schema_context,
            'data_relationships': session.schema_context.get('relationships', []),
            'data_constraints': session.schema_context.get('constraints', {}),
        }
        
        # 상세 프로세스 정보 (옵션)
        if include_process_details:
            data.update({
                'recommended_methods_details': session.recommended_methods,
                'assumption_tests': session.analysis_results.get('assumption_tests', {}),
                'preprocessing_steps': session.analysis_results.get('preprocessing', {}),
                'alternative_methods': self._get_alternative_methods(session),
            })
        
        return data
    
    def _generate_html_report(self, session: AnalysisSession, 
                            report_data: Dict[str, Any]) -> str:
        """HTML 형식 보고서 생성"""
        try:
            # Enhanced RAG 전용 HTML 템플릿 사용
            template_name = "enhanced_rag_analysis_report.html"
            
            html_content = self.template_manager.render_template(
                template_name, report_data
            )
            
            # 파일 저장
            filename = f"analysis_report_{session.session_id}.html"
            file_path = self.output_directory / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML 보고서 생성: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"HTML 보고서 생성 실패: {e}")
            return ""
    
    def _generate_markdown_report(self, session: AnalysisSession,
                                report_data: Dict[str, Any]) -> str:
        """Markdown 형식 보고서 생성"""
        try:
            template_name = "enhanced_rag_analysis_report.md"
            
            md_content = self.template_manager.render_template(
                template_name, report_data
            )
            
            filename = f"analysis_report_{session.session_id}.md"
            file_path = self.output_directory / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            self.logger.info(f"Markdown 보고서 생성: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Markdown 보고서 생성 실패: {e}")
            return ""
    
    def _generate_pdf_report(self, session: AnalysisSession,
                           report_data: Dict[str, Any]) -> str:
        """PDF 형식 보고서 생성"""
        try:
            # HTML을 먼저 생성한 후 PDF로 변환
            html_path = self._generate_html_report(session, report_data)
            if not html_path:
                return ""
            
            # PDF 변환 로직 (weasyprint 또는 다른 라이브러리 사용)
            # 현재는 HTML 경로만 반환 (추후 PDF 변환 구현)
            pdf_filename = f"analysis_report_{session.session_id}.pdf"
            pdf_path = self.output_directory / pdf_filename
            
            # TODO: HTML to PDF 변환 구현
            self.logger.info(f"PDF 보고서 생성 예정: {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            self.logger.error(f"PDF 보고서 생성 실패: {e}")
            return ""
    
    def _generate_dashboard(self, session: AnalysisSession,
                          report_data: Dict[str, Any]) -> str:
        """대화형 대시보드 생성"""
        try:
            # 대시보드 전용 템플릿 사용
            template_name = "interactive_dashboard.html"
            
            # 대시보드용 데이터 준비
            dashboard_data = {
                **report_data,
                'charts_data': self._prepare_charts_data(session),
                'interactive_elements': self._prepare_interactive_elements(session)
            }
            
            dashboard_content = self.template_manager.render_template(
                template_name, dashboard_data
            )
            
            filename = f"dashboard_{session.session_id}.html"
            file_path = self.output_directory / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(dashboard_content)
            
            self.logger.info(f"대시보드 생성: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"대시보드 생성 실패: {e}")
            return ""
    
    def _generate_visualization_section(self, session: AnalysisSession) -> str:
        """시각화 섹션 생성"""
        try:
            viz_data = {
                'analysis_results': session.analysis_results,
                'data_context': session.data_context,
                'selected_method': session.selected_method
            }
            
            return self.viz_reporter.create_comprehensive_report(
                [viz_data], None, {'session_id': session.session_id}
            )
            
        except Exception as e:
            self.logger.error(f"시각화 섹션 생성 실패: {e}")
            return ""
    
    def _generate_business_insights(self, results: Dict[str, Any]) -> List[str]:
        """분석 결과에서 비즈니스 인사이트 추출"""
        insights = []
        
        try:
            # 통계적 유의성 해석
            if results.get('significance', {}).get('p_value'):
                p_value = results['significance']['p_value']
                if p_value < 0.001:
                    insights.append("매우 강한 통계적 유의성으로 비즈니스 의사결정에 높은 신뢰도 제공")
                elif p_value < 0.01:
                    insights.append("강한 통계적 유의성으로 전략적 의사결정 지원 가능")
                elif p_value < 0.05:
                    insights.append("통계적 유의성 확인, 추가 검증과 함께 의사결정 참고 가능")
                else:
                    insights.append("통계적 유의성 부족, 현재 데이터로는 명확한 결론 도출 어려움")
            
            # 효과 크기 해석
            if results.get('effect_size', {}).get('value'):
                effect_size = results['effect_size']['value']
                if effect_size > 0.8:
                    insights.append("큰 효과 크기로 실무적 의미가 매우 높음")
                elif effect_size > 0.5:
                    insights.append("중간 정도 효과 크기로 실무적 고려 가치 있음")
                elif effect_size > 0.2:
                    insights.append("작은 효과 크기지만 대규모 환경에서는 의미 있을 수 있음")
                else:
                    insights.append("매우 작은 효과 크기로 실무적 영향 제한적")
            
            # 비즈니스 컨텍스트 기반 추가 인사이트
            if self.current_session:
                domain = self.current_session.business_domain
                if domain == "영업/매출 분석":
                    insights.append("매출 데이터 특성상 계절성과 외부 경제 요인 고려 필요")
                elif domain == "고객 분석":
                    insights.append("고객 행동 분석 시 장기적 관점에서의 생애 가치 고려 권장")
                elif domain == "제품 분석":
                    insights.append("제품 성능 분석 시 시장 경쟁 상황과 기술 트렌드 함께 고려")
        
        except Exception as e:
            self.logger.error(f"비즈니스 인사이트 생성 중 오류: {e}")
            insights.append("추가 분석을 통한 인사이트 도출 필요")
        
        return insights
    
    def _extract_key_findings(self, session: AnalysisSession) -> List[str]:
        """핵심 발견사항 추출"""
        findings = []
        
        try:
            results = session.analysis_results
            method = session.selected_method
            
            # 분석 방법별 핵심 발견사항
            if method.get('name') == 'ANOVA':
                if results.get('significance', {}).get('significant'):
                    findings.append("그룹 간 유의한 차이 존재")
                    findings.append("사후검정을 통한 구체적 차이 그룹 식별 필요")
                else:
                    findings.append("그룹 간 통계적으로 유의한 차이 없음")
            
            elif method.get('name') in ['t-test', 'T-검정']:
                if results.get('significance', {}).get('significant'):
                    findings.append("두 그룹 간 유의한 차이 존재")
                else:
                    findings.append("두 그룹 간 통계적 차이 없음")
            
            elif method.get('name') in ['correlation', '상관분석']:
                corr_value = results.get('correlation', {}).get('value', 0)
                if abs(corr_value) > 0.7:
                    findings.append("강한 상관관계 존재")
                elif abs(corr_value) > 0.3:
                    findings.append("중간 정도 상관관계 존재")
                else:
                    findings.append("약한 상관관계 또는 무상관")
            
            # 가정 검정 결과
            assumptions = results.get('assumption_tests', {})
            if assumptions.get('normality', {}).get('violated'):
                findings.append("정규성 가정 위반, 비모수 방법 고려 필요")
            if assumptions.get('homoscedasticity', {}).get('violated'):
                findings.append("등분산성 가정 위반, 로버스트 방법 적용 고려")
        
        except Exception as e:
            self.logger.error(f"핵심 발견사항 추출 중 오류: {e}")
        
        return findings
    
    def _extract_practical_implications(self, session: AnalysisSession) -> List[str]:
        """실무적 함의 추출"""
        implications = []
        
        try:
            domain = session.business_domain
            results = session.analysis_results
            
            if domain == "영업/매출 분석":
                if results.get('significance', {}).get('significant'):
                    implications.append("지역별/제품별 차별화된 영업 전략 수립 필요")
                    implications.append("성과가 높은 지역/제품의 성공 요인 분석 및 확산")
                    implications.append("저성과 지역/제품에 대한 개선 방안 마련")
                
            elif domain == "고객 분석":
                implications.append("고객 세그먼트별 맞춤형 마케팅 전략 개발")
                implications.append("고객 생애 가치 극대화를 위한 로열티 프로그램 강화")
                
            elif domain == "제품 분석":
                implications.append("제품 포트폴리오 최적화 검토")
                implications.append("신제품 개발 시 시장 수요 반영")
        
        except Exception as e:
            self.logger.error(f"실무적 함의 추출 중 오류: {e}")
        
        return implications
    
    def _generate_recommendations(self, session: AnalysisSession) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        try:
            results = session.analysis_results
            
            # 통계적 결과 기반 권장사항
            if results.get('significance', {}).get('significant'):
                recommendations.append("현재 분석 결과를 바탕으로 실행 계획 수립")
                recommendations.append("정기적인 모니터링을 통한 지속적 검증")
            else:
                recommendations.append("표본 크기 확대 또는 다른 분석 방법 검토")
                recommendations.append("추가 변수 수집을 통한 심화 분석")
            
            # 데이터 품질 기반 권장사항
            data_quality = session.data_context.get('quality_score', 0)
            if data_quality < 0.8:
                recommendations.append("데이터 품질 개선을 위한 수집 프로세스 점검")
                recommendations.append("결측치 처리 및 이상치 관리 체계 구축")
            
            # 비즈니스 컨텍스트 기반 권장사항
            recommendations.append("도메인 전문가와의 협의를 통한 결과 해석 보완")
            recommendations.append("정기적인 분석 업데이트로 트렌드 변화 추적")
        
        except Exception as e:
            self.logger.error(f"권장사항 생성 중 오류: {e}")
        
        return recommendations
    
    def _get_alternative_methods(self, session: AnalysisSession) -> List[Dict[str, Any]]:
        """대안 분석 방법 제안"""
        alternatives = []
        
        try:
            selected_method = session.selected_method.get('name', '')
            
            if 'ANOVA' in selected_method:
                alternatives.append({
                    'name': 'Kruskal-Wallis 검정',
                    'reason': '정규성 가정 위반 시 비모수적 대안',
                    'when_to_use': '데이터 분포가 정규분포를 따르지 않을 때'
                })
            
            elif 't-test' in selected_method or 'T-검정' in selected_method:
                alternatives.append({
                    'name': 'Mann-Whitney U 검정',
                    'reason': '비모수적 두 그룹 비교',
                    'when_to_use': '정규성 가정 위반 또는 순서형 데이터'
                })
            
            elif '상관분석' in selected_method:
                alternatives.append({
                    'name': 'Spearman 순위 상관',
                    'reason': '비선형 관계 또는 순서형 데이터',
                    'when_to_use': '단조 관계이지만 선형이 아닐 때'
                })
        
        except Exception as e:
            self.logger.error(f"대안 방법 제안 중 오류: {e}")
        
        return alternatives
    
    def _prepare_charts_data(self, session: AnalysisSession) -> Dict[str, Any]:
        """차트 데이터 준비"""
        try:
            # 시각화 데이터 구조화
            return {
                'statistical_summary': session.analysis_results.get('summary', {}),
                'assumption_tests': session.analysis_results.get('assumption_tests', {}),
                'effect_sizes': session.analysis_results.get('effect_size', {}),
                'confidence_intervals': session.analysis_results.get('confidence_intervals', {})
            }
        except Exception as e:
            self.logger.error(f"차트 데이터 준비 중 오류: {e}")
            return {}
    
    def _prepare_interactive_elements(self, session: AnalysisSession) -> Dict[str, Any]:
        """인터랙티브 요소 준비"""
        try:
            return {
                'method_comparison': session.recommended_methods,
                'parameter_sensitivity': {},  # 매개변수 민감도 분석
                'scenario_analysis': {},  # 시나리오 분석
                'drill_down_options': []  # 드릴다운 옵션
            }
        except Exception as e:
            self.logger.error(f"인터랙티브 요소 준비 중 오류: {e}")
            return {}
    
    def _get_session(self, session_id: str = None) -> Optional[AnalysisSession]:
        """세션 가져오기"""
        if session_id is None:
            return self.current_session
        return self.session_history.get(session_id)
    
    def export_session_data(self, session_id: str = None) -> str:
        """세션 데이터를 JSON으로 내보내기"""
        session = self._get_session(session_id)
        if not session:
            raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
        
        export_data = asdict(session)
        
        filename = f"session_data_{session.session_id}.json"
        file_path = self.output_directory / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"세션 데이터 내보내기: {file_path}")
        return str(file_path)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """세션 목록 조회"""
        return [
            {
                'session_id': session.session_id,
                'natural_language_request': session.natural_language_request,
                'business_domain': session.business_domain,
                'status': session.status,
                'created_at': session.created_at
            }
            for session in self.session_history.values()
        ]
    
    # 레거시 호환성을 위한 메서드
    def generate_report(self, agent_final_state: Dict, full_interaction_history: List[Dict], 
                       data_profile: Dict = None, workflow_graph_info: Dict = None) -> str:
        """
        레거시 인터페이스 호환성을 위한 메서드
        기존 Agent 시스템과의 연동을 위해 유지
        """
        try:
            # 레거시 데이터를 새 형식으로 변환
            if not self.current_session:
                # 기본 세션 생성
                business_context = BusinessContext(
                    domain="일반 분석",
                    terminology={},
                    guidelines=[],
                    constraints=[],
                    success_metrics=[]
                )
                schema_context = SchemaContext(
                    primary_table="data",
                    key_columns={},
                    relationships=[],
                    constraints={},
                    analytical_patterns={}
                )
                
                self.start_analysis_session(
                    natural_language_request=agent_final_state.get('user_request', '분석 요청'),
                    business_context=business_context,
                    schema_context=schema_context,
                    data_context=data_profile or {}
                )
            
            # 분석 결과 업데이트
            if agent_final_state.get('analysis_results'):
                self.update_analysis_results(agent_final_state['analysis_results'])
            
            # 보고서 생성
            reports = self.generate_comprehensive_report()
            
            # 주요 보고서 반환
            return reports.get('html', reports.get('markdown', ''))
            
        except Exception as e:
            self.logger.error(f"레거시 보고서 생성 중 오류: {e}")
            return "" 