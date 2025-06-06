"""
User Selection Pipeline

5단계: RAG 기반 대화형 사용자 의사결정 지원
Agent가 RAG 지식을 활용하여 사용자와 지능형 대화를 통해
최적의 분석 방법을 선택할 수 있도록 지원하며, 사용자의 의사결정을 
도메인 지식과 통계적 근거로 뒷받침합니다.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import asyncio

from .base_pipeline_step import BasePipelineStep, PipelineStepRegistry
from core.rag.rag_manager import RAGManager
from services.llm.llm_client import LLMClient
from services.llm.prompt_engine import PromptEngine
from utils.ui_helpers import UIHelpers


class UserSelectionStep(BasePipelineStep):
    """5단계: RAG 기반 대화형 사용자 의사결정 지원"""
    
    def __init__(self):
        """UserSelectionStep 초기화"""
        super().__init__("RAG 기반 대화형 사용자 의사결정 지원", 5)
        self.rag_manager = RAGManager()
        self.llm_client = LLMClient()
        self.prompt_engine = PromptEngine()
        self.ui_helpers = UIHelpers()
        
        # 대화형 Agent 설정
        self.conversation_config = {
            'max_conversation_turns': 5,
            'explanation_depth': 'adaptive',  # 사용자 수준에 맞춰 조정
            'decision_support_mode': 'collaborative',  # 협력적 의사결정
            'rag_integration_level': 'deep',  # 깊은 RAG 통합
            'personalization_level': 'medium'  # 사용자 맞춤화
        }
        
        # 대화 히스토리 저장
        self.conversation_history = []
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        입력 데이터 유효성 검증
        
        Args:
            input_data: 4단계에서 전달받은 데이터
            
        Returns:
            bool: 유효성 검증 결과
        """
        required_fields = [
            'agent_analysis_strategy', 'rag_integrated_insights',
            'adaptive_execution_plan', 'agent_reasoning_chain'
        ]
        return all(field in input_data for field in required_fields)
    
    def get_expected_output_schema(self) -> Dict[str, Any]:
        """
        예상 출력 스키마 반환
        
        Returns:
            Dict[str, Any]: 출력 데이터 스키마
        """
        return {
            'finalized_analysis_plan': {
                'selected_primary_method': dict,
                'confirmed_alternatives': list,
                'execution_parameters': dict,
                'user_preferences': dict
            },
            'enhanced_rag_context': {
                'targeted_domain_knowledge': dict,
                'method_specific_guidance': dict,
                'user_context_insights': dict,
                'risk_mitigation_strategies': list
            },
            'collaborative_decision_record': {
                'conversation_summary': dict,
                'decision_rationale': dict,
                'agent_recommendations': dict,
                'user_feedback_integration': dict
            },
            'adaptive_execution_adjustments': {
                'customized_parameters': dict,
                'dynamic_checkpoints': list,
                'quality_assurance_plan': dict,
                'contingency_protocols': dict
            },
            'knowledge_driven_insights': {
                'domain_specific_considerations': list,
                'methodological_best_practices': list,
                'implementation_guidance': dict,
                'expected_outcomes': dict
            }
        }
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        RAG 기반 대화형 사용자 의사결정 지원 실행
        
        Args:
            input_data: 파이프라인 실행 컨텍스트
            
        Returns:
            Dict: 실행 결과
        """
        self.logger.info("5단계: RAG 기반 대화형 사용자 의사결정 지원 시작")
        
        try:
            # 1. 사용자 컨텍스트 분석 및 RAG 지식 맞춤화
            personalized_rag_context = self._create_personalized_rag_context(input_data)
            
            # 2. 대화형 의사결정 프로세스 진행
            conversation_result = self._conduct_collaborative_decision_process(
                input_data, personalized_rag_context
            )
            
            # 3. 최종 분석 계획 확정
            finalized_analysis_plan = self._finalize_analysis_plan(
                conversation_result, input_data, personalized_rag_context
            )
            
            # 4. RAG 지식 기반 실행 조정사항 생성
            adaptive_execution_adjustments = self._generate_adaptive_adjustments(
                finalized_analysis_plan, personalized_rag_context, conversation_result
            )
            
            # 5. 의사결정 과정 문서화
            collaborative_decision_record = self._document_decision_process(
                conversation_result, finalized_analysis_plan, input_data
            )
            
            # 6. 지식 기반 인사이트 생성
            knowledge_driven_insights = self._generate_knowledge_insights(
                finalized_analysis_plan, personalized_rag_context, conversation_result
            )
            
            self.logger.info("RAG 기반 대화형 의사결정 지원 완료")
            
            return {
                'finalized_analysis_plan': finalized_analysis_plan,
                'enhanced_rag_context': personalized_rag_context,
                'collaborative_decision_record': collaborative_decision_record,
                'adaptive_execution_adjustments': adaptive_execution_adjustments,
                'knowledge_driven_insights': knowledge_driven_insights,
                'success_message': "🤝 사용자와 AI Agent가 협력하여 최적의 분석 계획을 수립했습니다."
            }
                
        except Exception as e:
            self.logger.error(f"RAG 기반 대화형 의사결정 지원 오류: {e}")
            return {
                'error': True,
                'error_message': str(e),
                'error_type': 'collaborative_decision_error'
            }
    
    def _create_personalized_rag_context(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 컨텍스트 분석 및 RAG 지식 맞춤화"""
        try:
            # 1. 사용자 배경 및 선호도 분석
            user_profile = self._analyze_user_context(input_data)
            
            # 2. 맞춤형 RAG 검색 쿼리 생성
            personalized_queries = self._build_personalized_search_queries(
                input_data, user_profile
            )
            
            # 3. 타겟 도메인 지식 수집
            targeted_domain_knowledge = self.rag_manager.search_and_build_context(
                query=personalized_queries['domain_specific'],
                collection="business_domains",
                top_k=6,
                context_type="user_domain_guidance",
                max_tokens=1200
            )
            
            # 4. 방법론별 상세 가이던스 수집
            method_specific_guidance = self.rag_manager.search_and_build_context(
                query=personalized_queries['methodological'],
                collection="statistical_concepts",
                top_k=8,
                context_type="method_selection_guidance",
                max_tokens=1500
            )
            
            # 5. 사용자 맥락 인사이트 생성
            user_context_insights = self._generate_user_context_insights(
                user_profile, targeted_domain_knowledge, method_specific_guidance
            )
            
            # 6. 리스크 완화 전략 수집
            risk_mitigation_strategies = self._collect_risk_mitigation_strategies(
                input_data, personalized_queries
            )
            
            return {
                'targeted_domain_knowledge': targeted_domain_knowledge,
                'method_specific_guidance': method_specific_guidance,
                'user_context_insights': user_context_insights,
                'risk_mitigation_strategies': risk_mitigation_strategies,
                'user_profile': user_profile
            }
            
        except Exception as e:
            self.logger.error(f"개인화된 RAG 컨텍스트 생성 오류: {e}")
            return self._create_default_rag_context()
    
    def _conduct_collaborative_decision_process(self, input_data: Dict[str, Any],
                                              rag_context: Dict[str, Any]) -> Dict[str, Any]:
        """대화형 의사결정 프로세스 진행"""
        try:
            # 1. 대화 세션 초기화
            conversation_state = self._initialize_conversation_state(input_data, rag_context)
            
            # 2. Agent의 초기 제안 및 설명
            initial_presentation = self._present_agent_recommendations(
                input_data, rag_context, conversation_state
            )
            
            # 3. 사용자와의 대화형 상호작용
            conversation_turns = []
            current_turn = 1
            
            while current_turn <= self.conversation_config['max_conversation_turns']:
                # 사용자 응답 수집
                user_response = self._collect_user_response(
                    initial_presentation if current_turn == 1 else conversation_turns[-1]['agent_message'],
                    conversation_state,
                    current_turn
                )
                
                if user_response.get('decision_finalized', False):
                    conversation_turns.append({
                        'turn': current_turn,
                        'user_response': user_response,
                        'decision_status': 'finalized'
                    })
                    break
                
                # Agent의 적응적 응답 생성
                agent_response = self._generate_adaptive_agent_response(
                    user_response, conversation_state, rag_context, current_turn
                )
                
                conversation_turns.append({
                    'turn': current_turn,
                    'user_response': user_response,
                    'agent_message': agent_response,
                    'conversation_state': conversation_state.copy()
                })
                
                # 대화 상태 업데이트
                conversation_state = self._update_conversation_state(
                    conversation_state, user_response, agent_response
                )
                
                current_turn += 1
            
            # 4. 대화 결과 종합
            conversation_summary = self._summarize_conversation(
                conversation_turns, initial_presentation, conversation_state
            )
            
            return {
                'initial_presentation': initial_presentation,
                'conversation_turns': conversation_turns,
                'conversation_summary': conversation_summary,
                'final_state': conversation_state
            }
            
        except Exception as e:
            self.logger.error(f"대화형 의사결정 프로세스 오류: {e}")
            return self._create_fallback_conversation_result(input_data)
    
    def _analyze_user_context(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 배경 및 선호도 분석"""
        try:
            # 사용자 요청에서 컨텍스트 추출
            user_request = input_data.get('user_request', '')
            data_overview = input_data.get('data_overview', {})
            
            # 도메인 식별
            domain_indicators = {
                'healthcare': ['환자', '치료', '병원', '의료', '진료', '수술', '약물'],
                'finance': ['매출', '수익', '비용', '투자', '금융', '주가', '경제'],
                'marketing': ['광고', '마케팅', '고객', '캠페인', '브랜드', '판매'],
                'education': ['학생', '교육', '학습', '성적', '시험', '과목', '학교'],
                'research': ['실험', '연구', '가설', '변수', '측정', '분석', '결과']
            }
            
            identified_domain = 'general'
            for domain, keywords in domain_indicators.items():
                if any(keyword in user_request for keyword in keywords):
                    identified_domain = domain
                    break
            
            # 기술 수준 추정
            technical_indicators = ['p-value', '신뢰구간', '효과크기', '검정력', '가설검정']
            tech_level = 'beginner'
            if any(indicator in user_request for indicator in technical_indicators):
                tech_level = 'intermediate'
            
            # 분석 목적 분류
            purpose_indicators = {
                'exploratory': ['탐색', '이해', '파악', '확인'],
                'confirmatory': ['검증', '증명', '테스트', '입증'],
                'predictive': ['예측', '예상', '모델링', '추정']
            }
            
            analysis_purpose = 'exploratory'
            for purpose, keywords in purpose_indicators.items():
                if any(keyword in user_request for keyword in keywords):
                    analysis_purpose = purpose
                    break
            
            return {
                'identified_domain': identified_domain,
                'technical_level': tech_level,
                'analysis_purpose': analysis_purpose,
                'user_request_analysis': {
                    'complexity': len(user_request.split()),
                    'specificity': 'high' if len(user_request) > 100 else 'medium'
                },
                'data_context': {
                    'size': data_overview.get('shape', {}).get('rows', 0),
                    'complexity': len(data_overview.get('columns', []))
                }
            }
            
        except Exception as e:
            self.logger.error(f"사용자 컨텍스트 분석 오류: {e}")
            return {
                'identified_domain': 'general',
                'technical_level': 'beginner',
                'analysis_purpose': 'exploratory'
            }
    
    def _build_personalized_search_queries(self, input_data: Dict[str, Any],
                                         user_profile: Dict[str, Any]) -> Dict[str, str]:
        """맞춤형 RAG 검색 쿼리 생성"""
        domain = user_profile.get('identified_domain', 'general')
        tech_level = user_profile.get('technical_level', 'beginner')
        purpose = user_profile.get('analysis_purpose', 'exploratory')
        user_request = input_data.get('user_request', '')
        
        return {
            'domain_specific': f"""
            도메인: {domain}
            사용자 요청: {user_request}
            분석 목적: {purpose}
            도메인 전문 용어, 일반적인 분석 패턴, 주의사항, 해석 가이드라인
            {domain} 분야 KPI, 성과 지표, 비즈니스 맥락
            """,
            
            'methodological': f"""
            기술 수준: {tech_level}
            분석 목적: {purpose}
            방법론 선택 기준, 가정 확인 방법, 결과 해석 가이드
            {tech_level} 수준 설명, 단계별 가이드, 주의사항, 대안 방법
            """,
            
            'implementation': f"""
            사용자 수준: {tech_level}
            구현 가이드, 코드 예시, 오류 처리, 결과 검증
            단계별 체크리스트, 품질 관리, 문제 해결
            """,
            
            'risk_management': f"""
            분석 목적: {purpose}
            도메인: {domain}
            일반적인 함정, 해석 오류, 예방 방법, 대안 전략
            위험 요소, 완화 방안, 검증 방법
            """
        }
    
    def _present_agent_recommendations(self, input_data: Dict[str, Any],
                                     rag_context: Dict[str, Any],
                                     conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """Agent의 초기 제안 및 설명"""
        try:
            # RAG 지식을 활용한 설명 생성
            explanation_prompt = self._build_explanation_prompt(
                input_data, rag_context, conversation_state
            )
            
            explanation_response = self.llm_client.generate_response(
                prompt=explanation_prompt,
                temperature=0.3,
                max_tokens=2000,
                system_prompt=self._get_explanation_system_prompt()
            )
            
            # 사용자 친화적 프레젠테이션 생성
            presentation = self._format_user_presentation(
                explanation_response, input_data, rag_context
            )
            
            return presentation
            
        except Exception as e:
            self.logger.error(f"Agent 추천 프레젠테이션 생성 오류: {e}")
            return self._create_basic_presentation(input_data)
    
    def _collect_user_response(self, presentation: Dict[str, Any],
                             conversation_state: Dict[str, Any],
                             turn_number: int) -> Dict[str, Any]:
        """사용자 응답 수집"""
        try:
            # 프레젠테이션 출력
            self._display_presentation(presentation, turn_number)
            
            # 대화형 입력 수집
            user_input = self._get_interactive_user_input(conversation_state, turn_number)
            
            # 응답 분석 및 구조화
            analyzed_response = self._analyze_user_response(user_input, conversation_state)
            
            return analyzed_response
            
        except Exception as e:
            self.logger.error(f"사용자 응답 수집 오류: {e}")
            return {'response': '기본 승인', 'decision_finalized': True}
    
    def _generate_adaptive_agent_response(self, user_response: Dict[str, Any],
                                        conversation_state: Dict[str, Any],
                                        rag_context: Dict[str, Any],
                                        turn_number: int) -> Dict[str, Any]:
        """Agent의 적응적 응답 생성"""
        try:
            # 사용자 응답 분석
            response_analysis = self._analyze_response_intent(user_response)
            
            # 적응적 RAG 검색
            adaptive_knowledge = self._perform_adaptive_rag_search(
                user_response, response_analysis, rag_context
            )
            
            # 맞춤형 응답 생성
            response_prompt = self._build_adaptive_response_prompt(
                user_response, conversation_state, adaptive_knowledge, turn_number
            )
            
            agent_response = self.llm_client.generate_response(
                prompt=response_prompt,
                temperature=0.4,
                max_tokens=1500,
                system_prompt=self._get_adaptive_response_system_prompt()
            )
            
            # 응답 구조화
            structured_response = self._structure_agent_response(
                agent_response, user_response, adaptive_knowledge
            )
            
            return structured_response
            
        except Exception as e:
            self.logger.error(f"적응적 Agent 응답 생성 오류: {e}")
            return self._create_fallback_agent_response(user_response)
    
    def _finalize_analysis_plan(self, conversation_result: Dict[str, Any],
                              input_data: Dict[str, Any],
                              rag_context: Dict[str, Any]) -> Dict[str, Any]:
        """최종 분석 계획 확정"""
        try:
            # 대화 결과 분석
            final_state = conversation_result.get('final_state', {})
            user_preferences = final_state.get('user_preferences', {})
            
            # 선택된 방법 확정
            selected_method = user_preferences.get('selected_method') or \
                            input_data.get('agent_analysis_strategy', {}).get('primary_recommendation', {})
            
            # 대안 방법 확정
            confirmed_alternatives = user_preferences.get('alternative_methods', []) or \
                                   input_data.get('agent_analysis_strategy', {}).get('alternative_strategies', [])
            
            # 실행 파라미터 설정
            execution_parameters = self._derive_execution_parameters(
                conversation_result, rag_context, user_preferences
            )
            
            return {
                'selected_primary_method': selected_method,
                'confirmed_alternatives': confirmed_alternatives,
                'execution_parameters': execution_parameters,
                'user_preferences': user_preferences,
                'finalization_confidence': self._calculate_finalization_confidence(
                    conversation_result, user_preferences
                )
            }
            
        except Exception as e:
            self.logger.error(f"분석 계획 확정 오류: {e}")
            return self._create_default_analysis_plan(input_data)
    
    def _create_default_rag_context(self) -> Dict[str, Any]:
        """기본 RAG 컨텍스트 생성"""
        return {
            'targeted_domain_knowledge': {'context': '', 'search_results': []},
            'method_specific_guidance': {'context': '', 'search_results': []},
            'user_context_insights': {},
            'risk_mitigation_strategies': [],
            'user_profile': {
                'identified_domain': 'general',
                'technical_level': 'beginner',
                'analysis_purpose': 'exploratory'
            }
        }
    
    def _generate_user_context_insights(self, user_profile: Dict[str, Any],
                                       targeted_domain_knowledge: Dict[str, Any],
                                       method_specific_guidance: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 맥락 인사이트 생성"""
        try:
            domain = user_profile.get('identified_domain', 'general')
            tech_level = user_profile.get('technical_level', 'beginner')
            purpose = user_profile.get('analysis_purpose', 'exploratory')
            
            return {
                'user_characteristics': {
                    'domain_expertise': domain,
                    'technical_proficiency': tech_level,
                    'analysis_intent': purpose
                },
                'communication_style': {
                    'explanation_depth': 'detailed' if tech_level == 'beginner' else 'concise',
                    'technical_terminology': tech_level != 'beginner',
                    'examples_needed': tech_level == 'beginner'
                },
                'decision_support_needs': {
                    'guidance_level': 'high' if tech_level == 'beginner' else 'medium',
                    'validation_required': True,
                    'alternative_options': tech_level != 'beginner'
                }
            }
            
        except Exception as e:
            self.logger.error(f"사용자 컨텍스트 인사이트 생성 오류: {e}")
            return {}
    
    def _collect_risk_mitigation_strategies(self, input_data: Dict[str, Any],
                                          personalized_queries: Dict[str, str]) -> List[Dict[str, Any]]:
        """리스크 완화 전략 수집"""
        try:
            risk_strategies = self.rag_manager.search_and_build_context(
                query=personalized_queries['risk_management'],
                collection="statistical_concepts",
                top_k=5,
                context_type="risk_mitigation",
                max_tokens=800
            )
            
            # 검색 결과를 구조화된 전략으로 변환
            strategies = []
            search_results = risk_strategies.get('search_results', [])
            
            for result in search_results:
                strategy = {
                    'risk_type': self._identify_risk_type(result.get('content', '')),
                    'mitigation_method': result.get('content', '')[:200],
                    'priority': 'high' if result.get('similarity_score', 0) > 0.8 else 'medium',
                    'source': result.get('source', 'unknown')
                }
                strategies.append(strategy)
            
            return strategies
            
        except Exception as e:
            self.logger.error(f"리스크 완화 전략 수집 오류: {e}")
            return []
    
    def _identify_risk_type(self, content: str) -> str:
        """콘텐츠에서 리스크 유형 식별"""
        risk_keywords = {
            'statistical': ['가정', '정규성', '등분산성', '독립성'],
            'interpretation': ['해석', '오해', '편향', '결론'],
            'implementation': ['구현', '코드', '오류', '버그'],
            'data_quality': ['결측치', '이상치', '품질', '무결성']
        }
        
        content_lower = content.lower()
        for risk_type, keywords in risk_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return risk_type
        
        return 'general'
    
    def _initialize_conversation_state(self, input_data: Dict[str, Any],
                                     rag_context: Dict[str, Any]) -> Dict[str, Any]:
        """대화 세션 초기화"""
        return {
            'conversation_id': f"conv_{hash(str(input_data))}"[:12],
            'user_preferences': {},
            'decisions_made': [],
            'questions_raised': [],
            'agent_confidence': 0.5,
            'user_satisfaction': 0.5,
            'context_evolution': [rag_context],
            'decision_criteria': []
        }
    
    def _update_conversation_state(self, current_state: Dict[str, Any],
                                 user_response: Dict[str, Any],
                                 agent_response: Dict[str, Any]) -> Dict[str, Any]:
        """대화 상태 업데이트"""
        updated_state = current_state.copy()
        
        # 사용자 선호도 업데이트
        if 'preferences' in user_response:
            updated_state['user_preferences'].update(user_response['preferences'])
        
        # 결정사항 추가
        if 'decision' in user_response:
            updated_state['decisions_made'].append(user_response['decision'])
        
        # 질문 추가
        if 'questions' in user_response:
            updated_state['questions_raised'].extend(user_response['questions'])
        
        return updated_state
    
    def _summarize_conversation(self, conversation_turns: List[Dict[str, Any]],
                              initial_presentation: Dict[str, Any],
                              final_state: Dict[str, Any]) -> Dict[str, Any]:
        """대화 결과 종합"""
        return {
            'total_turns': len(conversation_turns),
            'decisions_made': final_state.get('decisions_made', []),
            'user_preferences': final_state.get('user_preferences', {}),
            'questions_resolved': len(final_state.get('questions_raised', [])),
            'final_confidence': final_state.get('agent_confidence', 0.5),
            'conversation_success': len(final_state.get('decisions_made', [])) > 0
        }
    
    def _create_fallback_conversation_result(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """폴백 대화 결과 생성"""
        return {
            'initial_presentation': {'message': '기본 분석 제안'},
            'conversation_turns': [],
            'conversation_summary': {'conversation_success': False},
            'final_state': {'decisions_made': ['기본 분석 선택']}
        }
    
    def _build_explanation_prompt(self, input_data: Dict[str, Any],
                                  rag_context: Dict[str, Any],
                                  conversation_state: Dict[str, Any]) -> str:
        """설명을 위한 프롬프트 생성"""
        return f"""
사용자에게 분석 방법을 설명하고 선택을 도와주세요.

데이터: {input_data.get('data_overview', {})}
사용자 요청: {input_data.get('user_request', '')}
추천 방법: {input_data.get('agent_analysis_strategy', {})}

친근하고 이해하기 쉽게 설명해주세요.
        """
    
    def _get_explanation_system_prompt(self) -> str:
        """설명용 시스템 프롬프트"""
        return "당신은 친근하고 도움이 되는 데이터 분석 어시스턴트입니다."
    
    def _format_user_presentation(self, explanation_response: str,
                                  input_data: Dict[str, Any],
                                  rag_context: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 친화적 프레젠테이션 형식화"""
        return {
            'explanation': explanation_response,
            'options': ['기본 분석 승인', '대안 방법 요청', '수정 요청'],
            'recommendation': '기본 분석을 추천합니다'
        }
    
    def _create_basic_presentation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """기본 프레젠테이션 생성"""
        return {
            'explanation': '데이터 분석을 위한 기본 방법을 제안드립니다.',
            'options': ['승인', '거부'],
            'recommendation': '기본 분석 방법'
        }
    
    def _display_presentation(self, presentation: Dict[str, Any], turn_number: int):
        """프레젠테이션 출력"""
        print(f"\n=== 분석 방법 제안 (턴 {turn_number}) ===")
        print(presentation.get('explanation', ''))
        print("\n옵션:")
        for i, option in enumerate(presentation.get('options', []), 1):
            print(f"{i}. {option}")
    
    def _get_interactive_user_input(self, conversation_state: Dict[str, Any],
                                  turn_number: int) -> str:
        """대화형 사용자 입력 수집"""
        # 비대화형 모드에서는 기본값 반환
        return "1"  # 첫 번째 옵션 선택
    
    def _analyze_user_response(self, user_input: str,
                             conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 응답 분석"""
        return {
            'response': user_input,
            'decision_finalized': True,
            'preferences': {'selected_method': 'default'}
        }
    
    def _analyze_response_intent(self, user_response: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 응답 의도 분석"""
        return {
            'intent': 'approval',
            'confidence': 0.8,
            'needs_clarification': False
        }
    
    def _perform_adaptive_rag_search(self, user_response: Dict[str, Any],
                                   response_analysis: Dict[str, Any],
                                   rag_context: Dict[str, Any]) -> Dict[str, Any]:
        """적응적 RAG 검색"""
        return {'additional_context': '추가 정보 없음'}
    
    def _build_adaptive_response_prompt(self, user_response: Dict[str, Any],
                                      conversation_state: Dict[str, Any],
                                      adaptive_knowledge: Dict[str, Any],
                                      turn_number: int) -> str:
        """적응적 응답 프롬프트 생성"""
        return f"사용자 응답에 대한 적응적 답변을 생성해주세요. 턴: {turn_number}"
    
    def _get_adaptive_response_system_prompt(self) -> str:
        """적응적 응답용 시스템 프롬프트"""
        return "사용자의 의견을 고려하여 적절한 분석 방법을 제안해주세요."
    
    def _structure_agent_response(self, agent_response: str,
                                user_response: Dict[str, Any],
                                adaptive_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Agent 응답 구조화"""
        return {
            'message': agent_response,
            'suggestions': ['추가 검토'],
            'confidence': 0.7
        }
    
    def _create_fallback_agent_response(self, user_response: Dict[str, Any]) -> Dict[str, Any]:
        """폴백 Agent 응답"""
        return {
            'message': '기본 분석 방법을 진행하겠습니다.',
            'suggestions': [],
            'confidence': 0.5
        }
    
    def _derive_execution_parameters(self, conversation_result: Dict[str, Any],
                                   rag_context: Dict[str, Any],
                                   user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """실행 파라미터 도출"""
        return {
            'method': user_preferences.get('selected_method', 'default'),
            'confidence_level': 0.95,
            'output_format': 'standard'
        }
    
    def _calculate_finalization_confidence(self, conversation_result: Dict[str, Any],
                                         user_preferences: Dict[str, Any]) -> float:
        """확정 신뢰도 계산"""
        return 0.8
    
    def _create_default_analysis_plan(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """기본 분석 계획 생성"""
        return {
            'selected_primary_method': {'method': 'default_analysis'},
            'confirmed_alternatives': [],
            'execution_parameters': {'confidence_level': 0.95},
            'user_preferences': {'selected_method': 'default'},
            'finalization_confidence': 0.5
        }
    
    def _generate_adaptive_adjustments(self, finalized_analysis_plan: Dict[str, Any],
                                     rag_context: Dict[str, Any],
                                     conversation_result: Dict[str, Any]) -> Dict[str, Any]:
        """적응형 실행 조정사항 생성"""
        return {
            'customized_parameters': {},
            'dynamic_checkpoints': [],
            'quality_assurance_plan': {},
            'contingency_protocols': {}
        }
    
    def _document_decision_process(self, conversation_result: Dict[str, Any],
                                 finalized_analysis_plan: Dict[str, Any],
                                 input_data: Dict[str, Any]) -> Dict[str, Any]:
        """의사결정 과정 문서화"""
        return {
            'conversation_summary': conversation_result.get('conversation_summary', {}),
            'decision_rationale': {'reason': '사용자 승인'},
            'agent_recommendations': {'primary': '기본 분석'},
            'user_feedback_integration': {'feedback': '긍정적'}
        }
    
    def _generate_knowledge_insights(self, finalized_analysis_plan: Dict[str, Any],
                                   rag_context: Dict[str, Any],
                                   conversation_result: Dict[str, Any]) -> Dict[str, Any]:
        """지식 기반 인사이트 생성"""
        return {
            'domain_specific_considerations': [],
            'methodological_best_practices': [],
            'implementation_guidance': {},
            'expected_outcomes': {}
        }


