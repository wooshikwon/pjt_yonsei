"""
Analysis Proposal Pipeline

4단계: RAG 기반 Agentic LLM의 지능형 분석 전략 제안
RAG를 통해 확보한 통계 지식, 도메인 지식, 코드 템플릿을 LLM Agent가 완전히 통합하여
데이터 특성과 사용자 요구에 최적화된 분석 전략을 자율적으로 생성합니다.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from .base_pipeline_step import BasePipelineStep, PipelineStepRegistry
from core.rag.rag_manager import RAGManager
from services.llm.llm_client import LLMClient
from services.llm.prompt_engine import PromptEngine


class AnalysisProposalStep(BasePipelineStep):
    """4단계: RAG 기반 Agentic LLM의 지능형 분석 전략 제안"""
    
    def __init__(self):
        """AnalysisProposalStep 초기화"""
        super().__init__("RAG 기반 Agentic LLM의 지능형 분석 전략 제안", 4)
        self.rag_manager = RAGManager()
        self.llm_client = LLMClient()
        self.prompt_engine = PromptEngine()
        
        # Agent 설정
        self.agent_config = {
            'analysis_creativity': 0.7,  # 분석 방법 창의성
            'risk_tolerance': 0.3,       # 위험 허용도
            'explanation_depth': 'detailed',  # 설명 깊이
            'domain_focus': True         # 도메인 특화 분석
        }
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        입력 데이터 유효성 검증
        
        Args:
            input_data: 3단계에서 전달받은 데이터
            
        Returns:
            bool: 유효성 검증 결과
        """
        required_fields = [
            'agent_data_analysis', 'data_insights', 'quality_assessment',
            'analysis_recommendations', 'data_object'
        ]
        return all(field in input_data for field in required_fields)
    
    def get_expected_output_schema(self) -> Dict[str, Any]:
        """
        예상 출력 스키마 반환
        
        Returns:
            Dict[str, Any]: 출력 데이터 스키마
        """
        return {
            'agent_analysis_strategy': {
                'primary_recommendation': dict,
                'alternative_strategies': list,
                'strategy_rationale': dict,
                'confidence_scores': dict
            },
            'rag_integrated_insights': {
                'statistical_foundations': dict,
                'domain_best_practices': dict,
                'similar_cases': list,
                'methodological_guidance': dict
            },
            'adaptive_execution_plan': {
                'primary_path': dict,
                'fallback_scenarios': list,
                'dynamic_checkpoints': list,
                'adjustment_triggers': dict
            },
            'agent_reasoning_chain': {
                'decision_factors': list,
                'trade_off_analysis': dict,
                'assumption_validation': dict,
                'risk_assessment': dict
            },
            'contextual_recommendations': {
                'data_driven_insights': list,
                'domain_specific_advice': list,
                'implementation_guidelines': dict,
                'quality_assurance_plan': dict
            }
        }
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        RAG 기반 Agentic LLM의 지능형 분석 전략 제안 실행
        
        Args:
            input_data: 파이프라인 실행 컨텍스트
            
        Returns:
            Dict: 실행 결과
        """
        self.logger.info("4단계: RAG 기반 Agentic LLM의 지능형 분석 전략 제안 시작")
        
        try:
            # 1. RAG 기반 종합 지식 수집 및 통합
            rag_knowledge_context = self._collect_comprehensive_rag_knowledge(input_data)
            
            # 2. Agent의 자율적 분석 전략 수립
            agent_analysis_strategy = self._generate_autonomous_analysis_strategy(
                input_data, rag_knowledge_context
            )
            
            # 3. RAG 지식과 Agent 추론의 통합된 인사이트
            rag_integrated_insights = self._integrate_rag_agent_insights(
                rag_knowledge_context, agent_analysis_strategy
            )
            
            # 4. 적응형 실행 계획 수립
            adaptive_execution_plan = self._create_adaptive_execution_plan(
                agent_analysis_strategy, rag_integrated_insights, input_data
            )
            
            # 5. Agent 추론 과정 투명화
            agent_reasoning_chain = self._document_agent_reasoning(
                input_data, rag_knowledge_context, agent_analysis_strategy
            )
            
            # 6. 맥락적 추천사항 생성
            contextual_recommendations = self._generate_contextual_recommendations(
                agent_analysis_strategy, rag_integrated_insights, input_data
            )
            
            self.logger.info("RAG 기반 지능형 분석 전략 제안 완료")
            
            return {
                'agent_analysis_strategy': agent_analysis_strategy,
                'rag_integrated_insights': rag_integrated_insights,
                'adaptive_execution_plan': adaptive_execution_plan,
                'agent_reasoning_chain': agent_reasoning_chain,
                'contextual_recommendations': contextual_recommendations,
                'success_message': "🤖 AI Agent가 RAG 지식을 바탕으로 최적의 분석 전략을 수립했습니다."
            }
                
        except Exception as e:
            self.logger.error(f"RAG 기반 분석 전략 제안 오류: {e}")
            return {
                'error': True,
                'error_message': str(e),
                'error_type': 'agent_strategy_error'
            }
    
    def _collect_comprehensive_rag_knowledge(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """RAG 기반 종합 지식 수집 및 통합"""
        try:
            # 1. 다층적 RAG 검색 전략
            search_queries = self._build_multi_layer_search_queries(input_data)
            
            # 2. 통계 방법론 지식 수집
            statistical_knowledge = self.rag_manager.search_and_build_context(
                query=search_queries['statistical_methods'],
                collection="statistical_concepts",
                top_k=8,
                context_type="statistical_analysis",
                max_tokens=1500
            )
            
            # 3. 도메인 전문 지식 수집
            domain_knowledge = self.rag_manager.search_and_build_context(
                query=search_queries['domain_context'],
                collection="business_domains",
                top_k=5,
                context_type="domain_expertise",
                max_tokens=1000
            )
            
            # 4. 코드 구현 패턴 수집
            code_patterns = self.rag_manager.search_and_build_context(
                query=search_queries['implementation_patterns'],
                collection="code_templates",
                top_k=6,
                context_type="implementation_guidance",
                max_tokens=1200
            )
            
            # 5. 유사 케이스 스터디 검색
            similar_cases = self.rag_manager.search_and_build_context(
                query=search_queries['case_studies'],
                collection="case_studies",  # 새로운 컬렉션
                top_k=4,
                context_type="case_analysis",
                max_tokens=800
            )
            
            # 6. 지식 통합 및 중요도 가중치 부여
            integrated_knowledge = self._integrate_knowledge_with_weights({
                'statistical_knowledge': statistical_knowledge,
                'domain_knowledge': domain_knowledge,
                'code_patterns': code_patterns,
                'similar_cases': similar_cases
            })
            
            return integrated_knowledge
            
        except Exception as e:
            self.logger.error(f"RAG 지식 수집 오류: {e}")
            return self._create_fallback_knowledge_context()
    
    def _build_multi_layer_search_queries(self, input_data: Dict[str, Any]) -> Dict[str, str]:
        """다층적 RAG 검색 쿼리 생성"""
        # 데이터 특성 추출
        data_characteristics = input_data.get('summary_insights', {}).get('data_characteristics', [])
        variable_types = input_data.get('variable_analysis', {})
        user_intent = input_data.get('user_request', '')
        recommended_analyses = input_data.get('analysis_recommendations', {}).get('suitable_analyses', [])
        
        return {
            'statistical_methods': f"""
            데이터 특성: {', '.join(data_characteristics)}
            변수 유형: {json.dumps(variable_types, ensure_ascii=False)}
            추천 분석: {', '.join(recommended_analyses)}
            통계적 가정 검증, 효과 크기, 검정력 분석, 사후 검정
            """,
            
            'domain_context': f"""
            분석 목적: {user_intent}
            데이터 도메인 특성: {', '.join(data_characteristics)}
            비즈니스 인사이트, 도메인별 분석 패턴, KPI 해석
            """,
            
            'implementation_patterns': f"""
            구현 방법: {', '.join(recommended_analyses)}
            데이터 전처리, 코드 구조, 오류 처리, 결과 검증
            Python 통계 분석, pandas, scipy, statsmodels
            """,
            
            'case_studies': f"""
            유사 분석 사례: {user_intent}
            데이터 크기 및 특성: {', '.join(data_characteristics)}
            성공 사례, 실패 요인, 해결 방안
            """
        }
    
    def _integrate_knowledge_with_weights(self, knowledge_sources: Dict[str, Any]) -> Dict[str, Any]:
        """지식 소스별 가중치를 적용한 통합"""
        # 지식 유형별 가중치
        weights = {
            'statistical_knowledge': 0.35,
            'domain_knowledge': 0.25,
            'code_patterns': 0.25,
            'similar_cases': 0.15
        }
        
        integrated = {
            'weighted_contexts': {},
            'combined_insights': [],
            'cross_references': {},
            'confidence_metrics': {}
        }
        
        for source_name, source_data in knowledge_sources.items():
            weight = weights.get(source_name, 0.2)
            
            # 가중치 적용된 컨텍스트 저장
            integrated['weighted_contexts'][source_name] = {
                'context': source_data.get('context', ''),
                'search_results': source_data.get('search_results', []),
                'weight': weight,
                'relevance_score': self._calculate_relevance_score(source_data)
            }
            
            # 주요 인사이트 추출
            insights = self._extract_key_insights(source_data, weight)
            integrated['combined_insights'].extend(insights)
        
        # 교차 참조 구축
        integrated['cross_references'] = self._build_cross_references(knowledge_sources)
        
        return integrated
    
    def _calculate_relevance_score(self, source_data: Dict[str, Any]) -> float:
        """소스 데이터의 관련성 점수 계산"""
        try:
            search_results = source_data.get('search_results', [])
            if not search_results:
                return 0.0
            
            # 결과 개수와 품질을 기반으로 점수 계산
            num_results = len(search_results)
            avg_score = sum(result.get('similarity_score', 0.0) for result in search_results) / num_results
            
            # 0.0-1.0 범위로 정규화
            return min(avg_score, 1.0)
            
        except Exception:
            return 0.5  # 기본값
    
    def _extract_key_insights(self, source_data: Dict[str, Any], weight: float) -> List[str]:
        """가중치가 적용된 주요 인사이트 추출"""
        try:
            insights = []
            search_results = source_data.get('search_results', [])
            
            for result in search_results[:3]:  # 상위 3개 결과만 사용
                content = result.get('content', '')
                if content and len(content) > 50:  # 의미있는 내용만
                    insight = f"[가중치: {weight:.2f}] {content[:200]}..."
                    insights.append(insight)
            
            return insights
            
        except Exception:
            return []
    
    def _build_cross_references(self, knowledge_sources: Dict[str, Any]) -> Dict[str, Any]:
        """지식 소스 간 교차 참조 구축"""
        try:
            cross_refs = {
                'statistical_domain_overlap': [],
                'implementation_statistical_overlap': [],
                'case_domain_overlap': [],
                'common_themes': []
            }
            
            # 간단한 키워드 기반 교차 참조
            all_contents = {}
            for source_name, source_data in knowledge_sources.items():
                contents = []
                for result in source_data.get('search_results', []):
                    contents.append(result.get('content', ''))
                all_contents[source_name] = ' '.join(contents).lower()
            
            # 공통 테마 추출 (예시)
            common_keywords = ['분석', '통계', '검정', '데이터', '변수']
            for keyword in common_keywords:
                sources_with_keyword = [name for name, content in all_contents.items() 
                                      if keyword in content]
                if len(sources_with_keyword) > 1:
                    cross_refs['common_themes'].append({
                        'theme': keyword,
                        'sources': sources_with_keyword
                    })
            
            return cross_refs
            
        except Exception:
            return {}
    
    def _generate_autonomous_analysis_strategy(self, input_data: Dict[str, Any], 
                                             rag_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Agent의 자율적 분석 전략 수립"""
        try:
            # 1. RAG 지식을 통합한 Agent 프롬프트 구성
            agent_prompt = self._build_autonomous_agent_prompt(input_data, rag_knowledge)
            
            # 2. Agent의 자율적 추론 실행
            agent_response = self.llm_client.generate_response(
                prompt=agent_prompt,
                temperature=self.agent_config['analysis_creativity'],
                max_tokens=3000,
                system_prompt=self._get_agent_system_prompt()
            )
            
            # 3. Agent 응답 구조화
            strategy = self._parse_agent_strategy_response(agent_response)
            
            # 4. 신뢰도 점수 계산
            confidence_scores = self._calculate_strategy_confidence(
                strategy, rag_knowledge, input_data
            )
            
            # 5. 전략 검증 및 보완
            validated_strategy = self._validate_and_enhance_strategy(
                strategy, confidence_scores, rag_knowledge
            )
            
            return validated_strategy
            
        except Exception as e:
            self.logger.error(f"Agent 전략 수립 오류: {e}")
            return self._create_fallback_strategy(input_data)
    
    def _build_autonomous_agent_prompt(self, input_data: Dict[str, Any], 
                                     rag_knowledge: Dict[str, Any]) -> str:
        """RAG 지식을 통합한 Agent 프롬프트 구성"""
        
        # RAG 컨텍스트 추출
        statistical_context = rag_knowledge.get('weighted_contexts', {}).get('statistical_knowledge', {}).get('context', '')
        domain_context = rag_knowledge.get('weighted_contexts', {}).get('domain_knowledge', {}).get('context', '')
        code_context = rag_knowledge.get('weighted_contexts', {}).get('code_patterns', {}).get('context', '')
        case_context = rag_knowledge.get('weighted_contexts', {}).get('similar_cases', {}).get('context', '')
        
        prompt = f"""
당신은 RAG 지식을 활용하는 전문 통계 분석 AI Agent입니다. 
제공된 지식을 바탕으로 데이터에 최적화된 분석 전략을 자율적으로 수립하세요.

## 데이터 컨텍스트
{json.dumps(input_data.get('data_overview', {}), ensure_ascii=False, indent=2)}

## 변수 분석 결과
{json.dumps(input_data.get('variable_analysis', {}), ensure_ascii=False, indent=2)}

## 사용자 요구사항
{input_data.get('user_request', '명시되지 않음')}

## RAG 지식 베이스

### 통계 방법론 지식
{statistical_context}

### 도메인 전문 지식
{domain_context}

### 구현 패턴 가이드
{code_context}

### 유사 사례 분석
{case_context}

## Agent 임무
위의 모든 정보를 종합하여 다음을 자율적으로 결정하세요:

1. **주 분석 전략**: RAG 지식을 바탕으로 한 최적 분석 방법
2. **대안 전략들**: 리스크 관리를 위한 대체 방안들
3. **전략별 근거**: 각 선택의 통계적/도메인적 근거
4. **실행 우선순위**: 효율성과 정확성을 고려한 순서
5. **적응 계획**: 중간 결과에 따른 동적 조정 방안

응답은 JSON 형식으로 구조화하여 제공하세요.
        """
        
        return prompt
    
    def _get_agent_system_prompt(self) -> str:
        """Agent 시스템 프롬프트"""
        return """
당신은 고도로 훈련된 통계 분석 전문 AI Agent입니다.

핵심 원칙:
1. RAG 지식을 분석 판단의 핵심 근거로 활용
2. 데이터 특성에 맞는 최적화된 접근법 선택
3. 불확실성과 리스크를 명확히 인식하고 관리
4. 단계적이고 논리적인 추론 과정 유지
5. 도메인 맥락을 고려한 실용적 솔루션 제시

분석 결정 시 고려사항:
- 통계적 가정의 만족 여부
- 샘플 크기의 적절성
- 효과 크기의 실용적 의미
- 도메인별 해석 기준
- 구현 복잡도와 신뢰성의 균형
        """
    
    def _parse_agent_strategy_response(self, response: str) -> Dict[str, Any]:
        """Agent 응답을 구조화된 전략으로 파싱"""
        try:
            # JSON 응답 파싱 시도
            if '{' in response and '}' in response:
                json_part = response[response.find('{'):response.rfind('}')+1]
                parsed = json.loads(json_part)
                return self._validate_strategy_structure(parsed)
            else:
                # 텍스트 응답 파싱
                return self._parse_text_strategy_response(response)
                
        except Exception as e:
            self.logger.warning(f"Agent 응답 파싱 실패, 폴백 처리: {e}")
            return self._extract_strategy_from_text(response)
    
    def _validate_strategy_structure(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """전략 구조 검증 및 보완"""
        required_fields = {
            'primary_recommendation': {},
            'alternative_strategies': [],
            'strategy_rationale': {},
            'confidence_scores': {}
        }
        
        for field, default in required_fields.items():
            if field not in strategy:
                strategy[field] = default
                
        return strategy
    
    def _calculate_strategy_confidence(self, strategy: Dict[str, Any],
                                     rag_knowledge: Dict[str, Any],
                                     input_data: Dict[str, Any]) -> Dict[str, Any]:
        """전략별 신뢰도 점수 계산"""
        confidence_metrics = {
            'rag_knowledge_alignment': 0.0,
            'data_suitability': 0.0,
            'methodological_soundness': 0.0,
            'implementation_feasibility': 0.0,
            'overall_confidence': 0.0
        }
        
        try:
            # RAG 지식 정렬도 평가
            confidence_metrics['rag_knowledge_alignment'] = self._assess_rag_alignment(
                strategy, rag_knowledge
            )
            
            # 데이터 적합성 평가
            confidence_metrics['data_suitability'] = self._assess_data_suitability(
                strategy, input_data
            )
            
            # 방법론적 건전성 평가
            confidence_metrics['methodological_soundness'] = self._assess_methodological_soundness(
                strategy, rag_knowledge
            )
            
            # 구현 가능성 평가
            confidence_metrics['implementation_feasibility'] = self._assess_implementation_feasibility(
                strategy, rag_knowledge
            )
            
            # 전체 신뢰도 계산
            weights = [0.3, 0.3, 0.25, 0.15]
            scores = [confidence_metrics[key] for key in list(confidence_metrics.keys())[:-1]]
            confidence_metrics['overall_confidence'] = sum(w * s for w, s in zip(weights, scores))
            
        except Exception as e:
            self.logger.error(f"신뢰도 계산 오류: {e}")
            
        return confidence_metrics

    def _create_fallback_knowledge_context(self) -> Dict[str, Any]:
        """빈 컨텍스트 반환"""
        return {
            'weighted_contexts': {},
            'combined_insights': [],
            'cross_references': {},
            'confidence_metrics': {}
        }

    def _create_fallback_strategy(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """기본 전략 반환"""
        return {
            'primary_recommendation': '기본 분석',
            'alternative_strategies': [],
            'strategy_rationale': {},
            'confidence_scores': {}
        }

    def _assess_rag_alignment(self, strategy: Dict[str, Any],
                             rag_knowledge: Dict[str, Any]) -> float:
        """RAG 지식 정렬도 평가"""
        # 구현 필요
        return 0.5

    def _assess_data_suitability(self, strategy: Dict[str, Any],
                                 input_data: Dict[str, Any]) -> float:
        """데이터 적합성 평가"""
        # 구현 필요
        return 0.5

    def _assess_methodological_soundness(self, strategy: Dict[str, Any],
                                         rag_knowledge: Dict[str, Any]) -> float:
        """방법론적 건전성 평가"""
        # 구현 필요
        return 0.5

    def _assess_implementation_feasibility(self, strategy: Dict[str, Any],
                                           rag_knowledge: Dict[str, Any]) -> float:
        """구현 가능성 평가"""
        # 구현 필요
        return 0.5

    def _extract_strategy_from_text(self, text: str) -> Dict[str, Any]:
        """텍스트 기반 기본 전략 추출"""
        # 구현 필요
        return {
            'primary_recommendation': '기본 분석',
            'alternative_strategies': [],
            'strategy_rationale': {},
            'confidence_scores': {}
        }

    def _validate_and_enhance_strategy(self, strategy: Dict[str, Any],
                                     confidence_scores: Dict[str, Any],
                                     rag_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """전략 검증 및 보완"""
        # 구현 필요
        return strategy

    def _integrate_rag_agent_insights(self, rag_knowledge: Dict[str, Any],
                                     agent_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """RAG 지식과 Agent 추론의 통합된 인사이트"""
        # 구현 필요
        return {
            'statistical_foundations': {},
            'domain_best_practices': {},
            'similar_cases': [],
            'methodological_guidance': {}
        }

    def _create_adaptive_execution_plan(self, agent_strategy: Dict[str, Any],
                                     rag_integrated_insights: Dict[str, Any],
                                     input_data: Dict[str, Any]) -> Dict[str, Any]:
        """적응형 실행 계획 수립"""
        # 구현 필요
        return {
            'primary_path': {},
            'fallback_scenarios': [],
            'dynamic_checkpoints': [],
            'adjustment_triggers': {}
        }

    def _document_agent_reasoning(self, input_data: Dict[str, Any],
                                 rag_knowledge: Dict[str, Any],
                                 agent_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Agent 추론 과정 투명화"""
        # 구현 필요
        return {
            'decision_factors': [],
            'trade_off_analysis': {},
            'assumption_validation': {},
            'risk_assessment': {}
        }

    def _generate_contextual_recommendations(self, agent_strategy: Dict[str, Any],
                                           rag_integrated_insights: Dict[str, Any],
                                           input_data: Dict[str, Any]) -> Dict[str, Any]:
        """맥락적 추천사항 생성"""
        # 구현 필요
        return {
            'data_driven_insights': [],
            'domain_specific_advice': [],
            'implementation_guidelines': {},
            'quality_assurance_plan': {}
        }

    def get_step_info(self) -> Dict[str, Any]:
        """단계 정보 반환"""
        return {
            'step_number': 4,
            'step_name': 'analysis_proposal',
            'description': 'RAG 기반 Agentic LLM의 지능형 분석 전략 제안',
            'input_requirements': [
                'user_request',
                'data_overview', 
                'data_quality_assessment',
                'variable_analysis',
                'analysis_recommendations'
            ],
            'output_format': {
                'agent_analysis_strategy': 'Dict',
                'rag_integrated_insights': 'Dict', 
                'adaptive_execution_plan': 'Dict',
                'agent_reasoning_chain': 'Dict',
                'contextual_recommendations': 'Dict'
            },
            'estimated_duration': '3-5 minutes'
        }


