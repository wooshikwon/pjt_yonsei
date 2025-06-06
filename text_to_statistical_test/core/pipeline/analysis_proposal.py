"""
Analysis Proposal Pipeline

4단계: Agentic LLM의 분석 전략 제안
사용자의 요청, 데이터 특성, RAG를 통해 확보한 도메인 지식 및 통계적 지식을 종합하여 
가능한 분석 방법들과 각 방법의 장단점을 제시합니다.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base_pipeline_step import BasePipelineStep, PipelineStepRegistry
from core.rag.rag_manager import RAGManager
from services.llm.llm_client import LLMClient
from services.llm.prompt_engine import PromptEngine


class AnalysisProposalStep(BasePipelineStep):
    """4단계: Agentic LLM의 분석 전략 제안"""
    
    def __init__(self):
        """AnalysisProposalStep 초기화"""
        super().__init__("Agentic LLM의 분석 전략 제안", 4)
        self.rag_manager = RAGManager()
        self.llm_client = LLMClient()
        self.prompt_engine = PromptEngine()
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        입력 데이터 유효성 검증
        
        Args:
            input_data: 3단계에서 전달받은 데이터
            
        Returns:
            bool: 유효성 검증 결과
        """
        required_fields = [
            'data_overview', 'descriptive_statistics', 'data_quality_assessment',
            'variable_analysis', 'analysis_recommendations', 'summary_insights',
            'data_object'
        ]
        return all(field in input_data for field in required_fields)
    
    def get_expected_output_schema(self) -> Dict[str, Any]:
        """
        예상 출력 스키마 반환
        
        Returns:
            Dict[str, Any]: 출력 데이터 스키마
        """
        return {
            'analysis_proposals': {
                'recommended_methods': list,
                'alternative_methods': list,
                'method_details': dict,
                'rationale': dict
            },
            'statistical_context': {
                'assumptions': list,
                'limitations': list,
                'considerations': list
            },
            'domain_insights': {
                'business_context': dict,
                'similar_cases': list,
                'domain_specific_considerations': list
            },
            'execution_plan': {
                'steps': list,
                'required_validations': list,
                'potential_adjustments': list
            },
            'visualization_suggestions': {
                'pre_analysis': list,
                'during_analysis': list,
                'post_analysis': list
            }
        }
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agentic LLM의 분석 전략 제안 파이프라인 실행
        
        Args:
            input_data: 파이프라인 실행 컨텍스트
                - data_overview: 데이터 개요
                - descriptive_statistics: 기술 통계
                - data_quality_assessment: 데이터 품질 평가
                - variable_analysis: 변수 분석
                - analysis_recommendations: 분석 추천사항
                - summary_insights: 요약 인사이트
                - data_object: 데이터 객체
            
        Returns:
            Dict: 실행 결과
        """
        self.logger.info("4단계: Agentic LLM의 분석 전략 제안 시작")
        
        try:
            # 1. RAG를 통한 관련 지식 검색
            rag_context = self._retrieve_relevant_knowledge(input_data)
            
            # 2. 통계적 분석 방법 제안
            analysis_proposals = self._generate_analysis_proposals(input_data, rag_context)
            
            # 3. 통계적 컨텍스트 구성
            statistical_context = self._build_statistical_context(input_data, rag_context)
            
            # 4. 도메인 인사이트 생성
            domain_insights = self._generate_domain_insights(input_data, rag_context)
            
            # 5. 실행 계획 수립
            execution_plan = self._create_execution_plan(
                analysis_proposals, statistical_context, domain_insights
            )
            
            # 6. 시각화 제안
            visualization_suggestions = self._suggest_visualizations(
                input_data, analysis_proposals
            )
            
            self.logger.info("분석 전략 제안 완료")
            
            return {
                'analysis_proposals': analysis_proposals,
                'statistical_context': statistical_context,
                'domain_insights': domain_insights,
                'execution_plan': execution_plan,
                'visualization_suggestions': visualization_suggestions,
                'success_message': "📊 분석 전략 제안이 완료되었습니다."
            }
                
        except Exception as e:
            self.logger.error(f"분석 전략 제안 파이프라인 오류: {e}")
            return {
                'error': True,
                'error_message': str(e),
                'error_type': 'proposal_error'
            }
    
    def _retrieve_relevant_knowledge(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """RAG를 통한 관련 지식 검색"""
        try:
            # 1. 통계 방법론 관련 지식 검색
            statistical_knowledge = self.rag_manager.search(
                collection="statistical_concepts",
                query=self._build_statistical_query(input_data),
                top_k=5
            )
            
            # 2. 비즈니스 도메인 지식 검색
            domain_knowledge = self.rag_manager.search(
                collection="business_domains",
                query=self._build_domain_query(input_data),
                top_k=3
            )
            
            # 3. 코드 템플릿 검색
            code_templates = self.rag_manager.search(
                collection="code_templates",
                query=self._build_code_query(input_data),
                top_k=3
            )
            
            # 4. 컨텍스트 통합
            integrated_context = self.rag_manager.build_context(
                statistical_knowledge=statistical_knowledge,
                domain_knowledge=domain_knowledge,
                code_templates=code_templates,
                analysis_context=input_data
            )
            
            return integrated_context
            
        except Exception as e:
            self.logger.error(f"RAG 검색 오류: {e}")
            return {
                'error': True,
                'error_message': str(e)
            }
    
    def _build_statistical_query(self, input_data: Dict[str, Any]) -> str:
        """통계 방법론 검색을 위한 쿼리 생성"""
        analysis_type = input_data.get('analysis_recommendations', {}).get('suitable_analyses', [])
        data_characteristics = input_data.get('summary_insights', {}).get('data_characteristics', [])
        
        query = f"""
        통계 분석 방법: {', '.join(analysis_type)}
        데이터 특성: {', '.join(data_characteristics)}
        """
        return query
    
    def _build_domain_query(self, input_data: Dict[str, Any]) -> str:
        """도메인 지식 검색을 위한 쿼리 생성"""
        # 사용자 요청에서 도메인 관련 키워드 추출
        domain_context = input_data.get('user_request', '')
        variables = input_data.get('variable_analysis', {})
        
        query = f"""
        분석 컨텍스트: {domain_context}
        관련 변수: {variables}
        """
        return query
    
    def _build_code_query(self, input_data: Dict[str, Any]) -> str:
        """코드 템플릿 검색을 위한 쿼리 생성"""
        analysis_type = input_data.get('analysis_recommendations', {}).get('suitable_analyses', [])
        return f"통계 분석 코드 템플릿: {', '.join(analysis_type)}"
    
    def _generate_analysis_proposals(self, input_data: Dict[str, Any], 
                                   rag_context: Dict[str, Any]) -> Dict[str, Any]:
        """통계적 분석 방법 제안"""
        # LLM에 분석 제안 요청
        prompt = self.prompt_engine.create_analysis_proposal_prompt(
            input_data=input_data,
            rag_context=rag_context
        )
        
        llm_response = self.llm_client.generate(prompt)
        
        # LLM 응답 파싱 및 구조화
        proposals = self._parse_analysis_proposals(llm_response)
        
        return {
            'recommended_methods': proposals.get('recommended_methods', []),
            'alternative_methods': proposals.get('alternative_methods', []),
            'method_details': proposals.get('method_details', {}),
            'rationale': proposals.get('rationale', {})
        }
    
    def _build_statistical_context(self, input_data: Dict[str, Any], 
                                 rag_context: Dict[str, Any]) -> Dict[str, Any]:
        """통계적 컨텍스트 구성"""
        # 데이터 특성 및 제약사항 분석
        data_constraints = self._analyze_data_constraints(input_data)
        
        # 통계적 가정 및 요구사항 식별
        statistical_requirements = self._identify_statistical_requirements(
            input_data, rag_context
        )
        
        return {
            'assumptions': statistical_requirements.get('assumptions', []),
            'limitations': data_constraints.get('limitations', []),
            'considerations': statistical_requirements.get('considerations', [])
        }
    
    def _generate_domain_insights(self, input_data: Dict[str, Any], 
                                rag_context: Dict[str, Any]) -> Dict[str, Any]:
        """도메인 인사이트 생성"""
        # LLM에 도메인 인사이트 요청
        prompt = self.prompt_engine.create_domain_insight_prompt(
            input_data=input_data,
            rag_context=rag_context
        )
        
        llm_response = self.llm_client.generate(prompt)
        
        # LLM 응답 파싱 및 구조화
        insights = self._parse_domain_insights(llm_response)
        
        return {
            'business_context': insights.get('business_context', {}),
            'similar_cases': insights.get('similar_cases', []),
            'domain_specific_considerations': insights.get('considerations', [])
        }
    
    def _create_execution_plan(self, analysis_proposals: Dict[str, Any],
                             statistical_context: Dict[str, Any],
                             domain_insights: Dict[str, Any]) -> Dict[str, Any]:
        """실행 계획 수립"""
        # 분석 단계 정의
        analysis_steps = self._define_analysis_steps(
            analysis_proposals, statistical_context
        )
        
        # 필요한 검증 단계 식별
        required_validations = self._identify_required_validations(
            analysis_proposals, statistical_context
        )
        
        # 잠재적 조정사항 식별
        potential_adjustments = self._identify_potential_adjustments(
            analysis_proposals, domain_insights
        )
        
        return {
            'steps': analysis_steps,
            'required_validations': required_validations,
            'potential_adjustments': potential_adjustments
        }
    
    def _suggest_visualizations(self, input_data: Dict[str, Any],
                              analysis_proposals: Dict[str, Any]) -> Dict[str, Any]:
        """시각화 제안"""
        # 데이터 특성 기반 시각화 제안
        pre_analysis_viz = self._suggest_pre_analysis_visualizations(input_data)
        
        # 분석 과정 시각화 제안
        analysis_viz = self._suggest_analysis_visualizations(
            input_data, analysis_proposals
        )
        
        # 결과 시각화 제안
        post_analysis_viz = self._suggest_post_analysis_visualizations(
            analysis_proposals
        )
        
        return {
            'pre_analysis': pre_analysis_viz,
            'during_analysis': analysis_viz,
            'post_analysis': post_analysis_viz
        }
    
    def _parse_analysis_proposals(self, llm_response: str) -> Dict[str, Any]:
        """LLM 응답에서 분석 제안 파싱"""
        try:
            # JSON 형태의 응답이 포함된 경우 추출
            import json
            import re
            
            # JSON 블록 찾기
            json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    return parsed
                except json.JSONDecodeError:
                    # JSON 파싱 실패시 기본 구조로 텍스트 파싱 시도
                    self.logger.warning("JSON 파싱 실패, 텍스트 파싱으로 전환")
                    return self._fallback_text_parsing(llm_response)
            
            # 구조화된 텍스트 파싱
            proposals = {
                'recommended_methods': [],
                'alternative_methods': [],
                'method_details': {},
                'rationale': {}
            }
            
            # 추천 방법 추출
            recommended_pattern = r'추천\s*방법[:\s]*(.+?)(?=대안|방법|$)'
            recommended_match = re.search(recommended_pattern, llm_response, re.IGNORECASE | re.DOTALL)
            if recommended_match:
                methods_text = recommended_match.group(1)
                methods = re.findall(r'[-•]\s*([^-•\n]+)', methods_text)
                proposals['recommended_methods'] = [m.strip() for m in methods if m.strip()]
            
            # 대안 방법 추출
            alternative_pattern = r'대안\s*방법[:\s]*(.+?)(?=근거|이유|$)'
            alternative_match = re.search(alternative_pattern, llm_response, re.IGNORECASE | re.DOTALL)
            if alternative_match:
                alt_text = alternative_match.group(1)
                alt_methods = re.findall(r'[-•]\s*([^-•\n]+)', alt_text)
                proposals['alternative_methods'] = [m.strip() for m in alt_methods if m.strip()]
            
            # 근거 추출
            rationale_pattern = r'근거[:\s]*(.+?)$'
            rationale_match = re.search(rationale_pattern, llm_response, re.IGNORECASE | re.DOTALL)
            if rationale_match:
                proposals['rationale']['general'] = rationale_match.group(1).strip()
            
            return proposals
            
        except Exception as e:
            self.logger.error(f"분석 제안 파싱 오류: {e}")
            return {
                'recommended_methods': ['기술통계분석'],
                'alternative_methods': [],
                'method_details': {},
                'rationale': {'general': '기본 분석으로 시작'}
            }
    
    def _analyze_data_constraints(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 제약사항 분석"""
        try:
            constraints = {
                'limitations': [],
                'sample_size_issues': [],
                'data_quality_issues': [],
                'variable_constraints': []
            }
            
            # 데이터 개요에서 정보 추출
            data_overview = input_data.get('data_overview', {})
            quality_assessment = input_data.get('data_quality_assessment', {})
            
            # 샘플 크기 확인
            sample_size = data_overview.get('shape', {}).get('rows', 0)
            if sample_size < 30:
                constraints['sample_size_issues'].append('소표본으로 인한 통계적 검정력 부족')
                constraints['limitations'].append('비모수 검정 고려 필요')
            elif sample_size < 100:
                constraints['sample_size_issues'].append('중간 규모 표본으로 정규성 검정 주의 필요')
            
            # 결측값 확인
            missing_data = quality_assessment.get('missing_data', {})
            if missing_data:
                for var, missing_info in missing_data.items():
                    missing_rate = missing_info.get('percentage', 0)
                    if missing_rate > 20:
                        constraints['data_quality_issues'].append(f'{var}: 높은 결측률 ({missing_rate:.1f}%)')
                        constraints['limitations'].append('결측값 처리 전략 필요')
                    elif missing_rate > 5:
                        constraints['data_quality_issues'].append(f'{var}: 중간 결측률 ({missing_rate:.1f}%)')
            
            # 이상값 확인
            outliers = quality_assessment.get('outliers', {})
            if outliers:
                for var, outlier_info in outliers.items():
                    outlier_count = outlier_info.get('count', 0)
                    if outlier_count > 0:
                        constraints['data_quality_issues'].append(f'{var}: {outlier_count}개 이상값 발견')
                        constraints['limitations'].append('이상값 처리 방법 검토 필요')
            
            # 변수 유형별 제약사항
            variable_analysis = input_data.get('variable_analysis', {})
            for var_type, variables in variable_analysis.items():
                if var_type == 'categorical' and len(variables) > 0:
                    for var in variables:
                        if var.get('unique_values', 0) > 10:
                            constraints['variable_constraints'].append(f'{var["name"]}: 범주가 많음 (재코딩 고려)')
                elif var_type == 'numerical' and len(variables) > 0:
                    for var in variables:
                        skewness = var.get('skewness', 0)
                        if abs(skewness) > 2:
                            constraints['variable_constraints'].append(f'{var["name"]}: 심한 비대칭성 (변환 고려)')
            
            return constraints
            
        except Exception as e:
            self.logger.error(f"데이터 제약사항 분석 오류: {e}")
            return {'limitations': [], 'sample_size_issues': [], 'data_quality_issues': [], 'variable_constraints': []}
    
    def _identify_statistical_requirements(self, input_data: Dict[str, Any],
                                         rag_context: Dict[str, Any]) -> Dict[str, Any]:
        """통계적 요구사항 식별"""
        try:
            requirements = {
                'assumptions': [],
                'considerations': [],
                'required_tests': []
            }
            
            # 분석 추천사항에서 요구사항 추출
            analysis_recs = input_data.get('analysis_recommendations', {})
            suitable_analyses = analysis_recs.get('suitable_analyses', [])
            
            for analysis in suitable_analyses:
                # t-검정 관련 요구사항
                if 't-test' in analysis.lower() or 't검정' in analysis:
                    requirements['assumptions'].extend([
                        '정규성 가정 확인 필요',
                        '독립성 가정 확인 필요'
                    ])
                    if '독립' in analysis:
                        requirements['assumptions'].append('등분산성 가정 확인 필요')
                    requirements['required_tests'].extend(['정규성 검정', '등분산성 검정'])
                
                # ANOVA 관련 요구사항
                elif 'anova' in analysis.lower() or '분산분석' in analysis:
                    requirements['assumptions'].extend([
                        '정규성 가정 확인 필요',
                        '등분산성 가정 확인 필요',
                        '독립성 가정 확인 필요'
                    ])
                    requirements['required_tests'].extend(['정규성 검정', '등분산성 검정'])
                    requirements['considerations'].append('사후검정 계획 필요')
                
                # 회귀분석 관련 요구사항
                elif 'regression' in analysis.lower() or '회귀' in analysis:
                    requirements['assumptions'].extend([
                        '선형성 가정 확인 필요',
                        '정규성 가정 확인 필요',
                        '등분산성 가정 확인 필요',
                        '독립성 가정 확인 필요'
                    ])
                    requirements['required_tests'].extend(['선형성 검정', '정규성 검정', '등분산성 검정'])
                    requirements['considerations'].extend(['다중공선성 검토', '잔차분석 필요'])
                
                # 비모수 검정 관련 요구사항
                elif any(nonparam in analysis.lower() for nonparam in ['mann-whitney', 'kruskal', 'wilcoxon']):
                    requirements['assumptions'].append('분포의 모양 유사성 확인 필요')
                    requirements['considerations'].append('모수 검정 대비 검정력 고려')
                
                # 범주형 분석 관련 요구사항
                elif 'chi' in analysis.lower() or '카이제곱' in analysis:
                    requirements['assumptions'].extend([
                        '기대빈도 5 이상 확인 필요',
                        '독립성 가정 확인 필요'
                    ])
                    requirements['considerations'].append('효과크기 계산 고려')
            
            # RAG 컨텍스트에서 추가 고려사항 추출
            statistical_context = rag_context.get('statistical_concepts', [])
            for concept in statistical_context:
                content = concept.get('content', '')
                if '가정' in content or 'assumption' in content.lower():
                    # 통계적 가정 관련 내용 추출
                    assumptions = re.findall(r'([^.!?]*가정[^.!?]*)', content)
                    requirements['considerations'].extend([a.strip() for a in assumptions if a.strip()])
            
            # 중복 제거
            requirements['assumptions'] = list(set(requirements['assumptions']))
            requirements['considerations'] = list(set(requirements['considerations']))
            requirements['required_tests'] = list(set(requirements['required_tests']))
            
            return requirements
            
        except Exception as e:
            self.logger.error(f"통계적 요구사항 식별 오류: {e}")
            return {'assumptions': [], 'considerations': [], 'required_tests': []}
    
    def _parse_domain_insights(self, llm_response: str) -> Dict[str, Any]:
        """LLM 응답에서 도메인 인사이트 파싱"""
        try:
            import json
            import re
            
            insights = {
                'business_context': {},
                'similar_cases': [],
                'considerations': []
            }
            
            # JSON 형태 응답 시도
            json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    return parsed
                except json.JSONDecodeError:
                    # JSON 파싱 실패시 기본 구조로 텍스트 파싱 시도
                    self.logger.warning("JSON 파싱 실패, 텍스트 파싱으로 전환")
                    return self._fallback_text_parsing(llm_response)
            
            # 비즈니스 컨텍스트 추출
            business_pattern = r'비즈니스\s*(?:컨텍스트|맥락)[:\s]*(.+?)(?=유사|고려|$)'
            business_match = re.search(business_pattern, llm_response, re.IGNORECASE | re.DOTALL)
            if business_match:
                business_text = business_match.group(1).strip()
                insights['business_context']['description'] = business_text
                
                # 핵심 지표 추출
                kpi_pattern = r'(?:KPI|지표|성과)[:\s]*([^.\n]+)'
                kpi_matches = re.findall(kpi_pattern, business_text, re.IGNORECASE)
                if kpi_matches:
                    insights['business_context']['key_metrics'] = [kpi.strip() for kpi in kpi_matches]
            
            # 유사 사례 추출
            similar_pattern = r'유사\s*(?:사례|경우)[:\s]*(.+?)(?=고려|권고|$)'
            similar_match = re.search(similar_pattern, llm_response, re.IGNORECASE | re.DOTALL)
            if similar_match:
                similar_text = similar_match.group(1)
                cases = re.findall(r'[-•]\s*([^-•\n]+)', similar_text)
                insights['similar_cases'] = [case.strip() for case in cases if case.strip()]
            
            # 고려사항 추출
            consideration_pattern = r'고려\s*(?:사항|할점)[:\s]*(.+?)$'
            consideration_match = re.search(consideration_pattern, llm_response, re.IGNORECASE | re.DOTALL)
            if consideration_match:
                consideration_text = consideration_match.group(1)
                considerations = re.findall(r'[-•]\s*([^-•\n]+)', consideration_text)
                insights['considerations'] = [cons.strip() for cons in considerations if cons.strip()]
            
            return insights
            
        except Exception as e:
            self.logger.error(f"도메인 인사이트 파싱 오류: {e}")
            return {'business_context': {}, 'similar_cases': [], 'considerations': []}
    
    def _define_analysis_steps(self, analysis_proposals: Dict[str, Any],
                             statistical_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """분석 단계 정의"""
        try:
            steps = []
            
            # 기본 데이터 탐색 단계
            steps.append({
                'step_number': 1,
                'name': '데이터 탐색',
                'description': '기술통계 및 시각화를 통한 데이터 이해',
                'tasks': ['기술통계 계산', '분포 확인', '이상값 탐지'],
                'estimated_time': '5-10분'
            })
            
            # 가정 검정 단계
            required_tests = statistical_context.get('required_tests', [])
            if required_tests:
                steps.append({
                    'step_number': 2,
                    'name': '통계적 가정 검정',
                    'description': '분석 전 필요한 가정들을 검증',
                    'tasks': required_tests,
                    'estimated_time': '3-5분'
                })
            
            # 주 분석 단계
            recommended_methods = analysis_proposals.get('recommended_methods', [])
            for i, method in enumerate(recommended_methods):
                steps.append({
                    'step_number': len(steps) + 1,
                    'name': f'주 분석 {i+1}: {method}',
                    'description': f'{method} 실행 및 결과 해석',
                    'tasks': [f'{method} 실행', '결과 해석', '효과크기 계산'],
                    'estimated_time': '10-15분'
                })
            
            # 대안 분석 단계 (조건부)
            alternative_methods = analysis_proposals.get('alternative_methods', [])
            if alternative_methods:
                steps.append({
                    'step_number': len(steps) + 1,
                    'name': '대안 분석',
                    'description': '가정 위배시 실행할 대안 분석',
                    'tasks': [f'{method} 실행' for method in alternative_methods],
                    'estimated_time': '5-10분'
                })
            
            # 결과 종합 단계
            steps.append({
                'step_number': len(steps) + 1,
                'name': '결과 종합',
                'description': '분석 결과 종합 및 해석',
                'tasks': ['결과 비교', '최종 해석', '보고서 작성'],
                'estimated_time': '10-15분'
            })
            
            return steps
            
        except Exception as e:
            self.logger.error(f"분석 단계 정의 오류: {e}")
            return [{'step_number': 1, 'name': '기본 분석', 'description': '기술통계 분석', 'tasks': ['기술통계'], 'estimated_time': '5분'}]
    
    def _identify_required_validations(self, analysis_proposals: Dict[str, Any],
                                     statistical_context: Dict[str, Any]) -> List[str]:
        """필요한 검증 단계 식별"""
        try:
            validations = []
            
            # 통계적 가정 검증
            assumptions = statistical_context.get('assumptions', [])
            for assumption in assumptions:
                if '정규성' in assumption:
                    validations.append('정규성 검정 (Shapiro-Wilk 또는 Kolmogorov-Smirnov)')
                elif '등분산성' in assumption:
                    validations.append('등분산성 검정 (Levene 또는 Bartlett)')
                elif '선형성' in assumption:
                    validations.append('선형성 검정 (산점도 및 잔차분석)')
                elif '독립성' in assumption:
                    validations.append('독립성 검정 (Durbin-Watson 또는 시각적 확인)')
            
            # 분석별 특화 검증
            recommended_methods = analysis_proposals.get('recommended_methods', [])
            for method in recommended_methods:
                if '회귀' in method:
                    validations.extend([
                        '다중공선성 검정 (VIF)',
                        '잔차의 정규성 검정',
                        '영향력 있는 관측값 탐지'
                    ])
                elif 'anova' in method.lower() or '분산분석' in method:
                    validations.append('집단 크기의 균형성 확인')
                elif '카이제곱' in method:
                    validations.append('기대빈도 조건 확인 (모든 셀 ≥ 5)')
            
            # 중복 제거
            validations = list(set(validations))
            
            return validations
            
        except Exception as e:
            self.logger.error(f"검증 단계 식별 오류: {e}")
            return ['기본 데이터 검증']
    
    def _identify_potential_adjustments(self, analysis_proposals: Dict[str, Any],
                                      domain_insights: Dict[str, Any]) -> List[str]:
        """잠재적 조정사항 식별"""
        try:
            adjustments = []
            
            # 비즈니스 컨텍스트 기반 조정
            business_context = domain_insights.get('business_context', {})
            if business_context:
                key_metrics = business_context.get('key_metrics', [])
                if key_metrics:
                    adjustments.append('비즈니스 핵심 지표에 맞춘 해석 방향 조정')
            
            # 도메인별 고려사항
            considerations = domain_insights.get('considerations', [])
            if considerations:
                adjustments.extend([
                    f'도메인 특화 고려사항 반영: {cons}' 
                    for cons in considerations[:3]  # 상위 3개만
                ])
            
            # 유사 사례 기반 조정
            similar_cases = domain_insights.get('similar_cases', [])
            if similar_cases:
                adjustments.append('유사 사례 분석 결과를 참고한 해석 방향 설정')
            
            # 분석 방법별 일반적 조정사항
            recommended_methods = analysis_proposals.get('recommended_methods', [])
            for method in recommended_methods:
                if '회귀' in method:
                    adjustments.extend([
                        '변수 선택 방법 조정 (stepwise, forward, backward)',
                        '상호작용 항 추가 고려'
                    ])
                elif 't-test' in method.lower() or 't검정' in method:
                    adjustments.append('효과크기 기준 실무적 유의성 판단')
                elif 'anova' in method.lower():
                    adjustments.append('사후검정 방법 선택 (Bonferroni, Tukey, 등)')
            
            # 중복 제거 및 개수 제한
            adjustments = list(set(adjustments))[:8]  # 최대 8개로 제한
            
            return adjustments
            
        except Exception as e:
            self.logger.error(f"조정사항 식별 오류: {e}")
            return ['결과 해석시 도메인 전문성 반영']
    
    def _suggest_pre_analysis_visualizations(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """분석 전 시각화 제안"""
        try:
            visualizations = []
            
            # 변수 분석에서 시각화 제안
            variable_analysis = input_data.get('variable_analysis', {})
            
            # 수치형 변수 시각화
            numerical_vars = variable_analysis.get('numerical', [])
            if numerical_vars:
                visualizations.extend([
                    {
                        'type': 'histogram',
                        'title': '수치형 변수 분포 확인',
                        'description': '각 수치형 변수의 분포와 정규성 확인',
                        'variables': [var['name'] for var in numerical_vars[:4]],  # 최대 4개
                        'purpose': '정규성 가정 검토'
                    },
                    {
                        'type': 'boxplot',
                        'title': '이상값 탐지',
                        'description': '수치형 변수의 이상값 시각적 확인',
                        'variables': [var['name'] for var in numerical_vars[:4]],
                        'purpose': '이상값 식별'
                    }
                ])
                
                # 상관관계 매트릭스 (변수가 2개 이상일 때)
                if len(numerical_vars) >= 2:
                    visualizations.append({
                        'type': 'correlation_matrix',
                        'title': '변수 간 상관관계',
                        'description': '수치형 변수들 간의 선형 관계 확인',
                        'variables': [var['name'] for var in numerical_vars],
                        'purpose': '다중공선성 예비 확인'
                    })
            
            # 범주형 변수 시각화
            categorical_vars = variable_analysis.get('categorical', [])
            if categorical_vars:
                visualizations.extend([
                    {
                        'type': 'bar_chart',
                        'title': '범주형 변수 빈도',
                        'description': '각 범주의 빈도 및 분포 확인',
                        'variables': [var['name'] for var in categorical_vars[:3]],
                        'purpose': '범주 균형성 확인'
                    }
                ])
            
            # 변수 간 관계 시각화
            if numerical_vars and categorical_vars:
                visualizations.append({
                    'type': 'grouped_boxplot',
                    'title': '그룹별 수치형 변수 분포',
                    'description': '범주형 변수별 수치형 변수의 분포 비교',
                    'variables': {
                        'numerical': numerical_vars[0]['name'],
                        'categorical': categorical_vars[0]['name']
                    },
                    'purpose': '그룹 간 차이 예비 탐색'
                })
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"분석 전 시각화 제안 오류: {e}")
            return [{'type': 'basic_plot', 'title': '기본 데이터 탐색', 'description': '데이터 기본 구조 확인', 'variables': [], 'purpose': '데이터 이해'}]
    
    def _suggest_analysis_visualizations(self, input_data: Dict[str, Any],
                                       analysis_proposals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """분석 과정 시각화 제안"""
        try:
            visualizations = []
            
            recommended_methods = analysis_proposals.get('recommended_methods', [])
            
            for method in recommended_methods:
                if 't-test' in method.lower() or 't검정' in method:
                    visualizations.extend([
                        {
                            'type': 'qq_plot',
                            'title': 'Q-Q 플롯',
                            'description': '정규성 가정 시각적 검증',
                            'purpose': '정규성 가정 확인'
                        },
                        {
                            'type': 'group_comparison',
                            'title': '그룹 비교 시각화',
                            'description': '그룹 간 평균 및 분산 비교',
                            'purpose': '차이 시각화'
                        }
                    ])
                
                elif 'anova' in method.lower() or '분산분석' in method:
                    visualizations.extend([
                        {
                            'type': 'residual_plot',
                            'title': '잔차 분석',
                            'description': 'ANOVA 가정 검증을 위한 잔차 분석',
                            'purpose': '가정 검증'
                        },
                        {
                            'type': 'means_plot',
                            'title': '그룹별 평균 비교',
                            'description': '각 그룹의 평균과 신뢰구간',
                            'purpose': '그룹 차이 시각화'
                        }
                    ])
                
                elif '회귀' in method or 'regression' in method.lower():
                    visualizations.extend([
                        {
                            'type': 'scatter_regression',
                            'title': '회귀선 포함 산점도',
                            'description': '독립변수와 종속변수의 관계 및 회귀선',
                            'purpose': '선형관계 확인'
                        },
                        {
                            'type': 'residual_vs_fitted',
                            'title': '잔차 vs 적합값',
                            'description': '회귀 가정 검증을 위한 잔차 분석',
                            'purpose': '등분산성 및 선형성 확인'
                        }
                    ])
                
                elif '상관' in method or 'correlation' in method.lower():
                    visualizations.append({
                        'type': 'correlation_heatmap',
                        'title': '상관계수 히트맵',
                        'description': '변수들 간의 상관관계 강도 시각화',
                        'purpose': '상관관계 패턴 이해'
                    })
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"분석 과정 시각화 제안 오류: {e}")
            return [{'type': 'basic_analysis_plot', 'title': '기본 분석 시각화', 'description': '분석 결과 시각화', 'purpose': '결과 이해'}]
    
    def _suggest_post_analysis_visualizations(self, analysis_proposals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """분석 후 시각화 제안"""
        try:
            visualizations = []
            
            recommended_methods = analysis_proposals.get('recommended_methods', [])
            
            for method in recommended_methods:
                if 't-test' in method.lower() or 't검정' in method:
                    visualizations.extend([
                        {
                            'type': 'effect_size_plot',
                            'title': '효과크기 시각화',
                            'description': 'Cohen\'s d와 신뢰구간 표시',
                            'purpose': '실무적 유의성 판단'
                        },
                        {
                            'type': 'mean_difference_plot',
                            'title': '평균 차이 시각화',
                            'description': '그룹 간 평균 차이와 신뢰구간',
                            'purpose': '결과 해석 지원'
                        }
                    ])
                
                elif 'anova' in method.lower() or '분산분석' in method:
                    visualizations.extend([
                        {
                            'type': 'posthoc_comparison',
                            'title': '사후검정 결과',
                            'description': '그룹 간 다중비교 결과 시각화',
                            'purpose': '구체적 차이 파악'
                        },
                        {
                            'type': 'eta_squared_plot',
                            'title': '효과크기 (Eta-squared)',
                            'description': '설명 가능한 분산의 비율',
                            'purpose': '실무적 중요성 평가'
                        }
                    ])
                
                elif '회귀' in method or 'regression' in method.lower():
                    visualizations.extend([
                        {
                            'type': 'coefficient_plot',
                            'title': '회귀계수 시각화',
                            'description': '회귀계수와 신뢰구간',
                            'purpose': '변수 영향력 비교'
                        },
                        {
                            'type': 'prediction_plot',
                            'title': '예측값 vs 실제값',
                            'description': '모델의 예측 성능 시각화',
                            'purpose': '모델 성능 평가'
                        }
                    ])
                
                elif '카이제곱' in method or 'chi' in method.lower():
                    visualizations.extend([
                        {
                            'type': 'contingency_heatmap',
                            'title': '분할표 히트맵',
                            'description': '관찰빈도와 기대빈도 비교',
                            'purpose': '연관성 패턴 시각화'
                        },
                        {
                            'type': 'cramers_v_plot',
                            'title': 'Cramer\'s V 효과크기',
                            'description': '범주형 변수 간 연관성 강도',
                            'purpose': '연관성 크기 평가'
                        }
                    ])
            
            # 공통 결과 시각화
            visualizations.append({
                'type': 'summary_dashboard',
                'title': '분석 결과 대시보드',
                'description': '주요 결과를 종합한 대시보드',
                'purpose': '전체 결과 요약'
            })
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"분석 후 시각화 제안 오류: {e}")
            return [{'type': 'results_summary', 'title': '결과 요약', 'description': '분석 결과 요약', 'purpose': '결과 정리'}]
    
    def get_step_info(self) -> Dict[str, Any]:
        """단계 정보 반환"""
        return {
            'step_number': 4,
            'step_name': 'analysis_proposal',
            'description': 'Agentic LLM의 분석 전략 제안',
            'input_requirements': [
                'user_request',
                'data_overview', 
                'data_quality_assessment',
                'variable_analysis',
                'analysis_recommendations'
            ],
            'output_format': {
                'analysis_proposals': 'Dict',
                'statistical_context': 'Dict', 
                'domain_insights': 'Dict',
                'execution_plan': 'Dict',
                'visualization_suggestions': 'Dict'
            },
            'estimated_duration': '3-5 minutes'
        }

    def _fallback_text_parsing(self, text: str) -> Dict[str, Any]:
        """JSON 파싱 실패시 텍스트 기반 기본 파싱"""
        try:
            # 기본 구조 생성
            fallback_result = {
                'recommended_methods': [],
                'alternative_methods': [],
                'method_details': {},
                'rationale': {}
            }
            
            # 간단한 키워드 기반 추출
            text_lower = text.lower()
            
            # 일반적인 통계 방법들 검색
            common_methods = [
                't-test', 't검정', 'anova', '분산분석', '회귀분석', 'regression',
                '상관분석', 'correlation', '카이제곱', 'chi-square', 'mann-whitney',
                'kruskal-wallis', 'wilcoxon'
            ]
            
            found_methods = []
            for method in common_methods:
                if method in text_lower:
                    found_methods.append(method)
            
            # 발견된 방법이 있으면 추천 방법으로 설정
            if found_methods:
                fallback_result['recommended_methods'] = found_methods[:3]  # 최대 3개
                fallback_result['rationale']['general'] = '텍스트에서 추출된 분석 방법'
            else:
                # 기본 분석 방법 제공
                fallback_result['recommended_methods'] = ['기술통계분석', '탐색적 데이터 분석']
                fallback_result['rationale']['general'] = '기본 분석으로 시작'
            
            return fallback_result
            
        except Exception as e:
            self.logger.error(f"Fallback 파싱 오류: {e}")
            return {
                'recommended_methods': ['기술통계분석'],
                'alternative_methods': [],
                'method_details': {},
                'rationale': {'general': '기본 분석'}
            }


# 단계 등록
PipelineStepRegistry.register_step(4, AnalysisProposalStep) 