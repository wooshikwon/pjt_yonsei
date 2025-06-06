"""
RAG 시스템의 컨텍스트 빌더 모듈
검색된 정보를 LLM 프롬프트에 효과적으로 통합하여 Agent의 추론 능력을 향상시킵니다.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import json
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
import re

from utils.error_handler import ErrorHandler
from .query_engine import SearchResult

@dataclass
class ContextTemplate:
    """컨텍스트 템플릿 정의"""
    name: str
    sections: List[str]
    max_length: int
    priority_weights: Dict[str, float]
    format_style: str

class ContextBuilder:
    """RAG 시스템의 컨텍스트 빌더"""
    
    def __init__(self, max_context_length: int = 4000):
        """
        Args:
            max_context_length: 최대 컨텍스트 길이 (토큰 수 근사)
        """
        self.logger = logging.getLogger(__name__)
        self.error_handler = ErrorHandler()
        self.max_context_length = max_context_length
        
        # 컨텍스트 템플릿 정의
        self.templates = self._initialize_templates()
        
        # 섹션별 우선순위
        self.section_priorities = {
            'statistical_concepts': 1.0,
            'domain_knowledge': 0.9,
            'code_examples': 0.8,
            'methodology': 0.7,
            'best_practices': 0.6,
            'historical_context': 0.5
        }
        
        # 지식 유형별 포맷터
        self.formatters = {
            'statistical_concepts': self._format_statistical_concept,
            'business_domains': self._format_domain_knowledge,
            'code_templates': self._format_code_template,
            'workflow_guidelines': self._format_workflow_guideline
        }
    
    def build_context(self,
                     search_results: List[SearchResult],
                     context_type: str = "general",
                     user_query: str = "",
                     additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        검색 결과를 바탕으로 LLM 컨텍스트 구성
        
        Args:
            search_results: 검색 결과 리스트
            context_type: 컨텍스트 유형 ("analysis", "interpretation", "planning" 등)
            user_query: 사용자 쿼리
            additional_context: 추가 컨텍스트 정보
            
        Returns:
            구성된 컨텍스트 딕셔너리
        """
        try:
            # 템플릿 선택
            template = self._select_template(context_type)
            
            # 검색 결과 분류 및 우선순위 설정
            categorized_results = self._categorize_search_results(search_results)
            
            # 컨텍스트 섹션별 구성
            context_sections = self._build_context_sections(
                categorized_results, template, user_query
            )
            
            # 길이 최적화
            optimized_context = self._optimize_context_length(
                context_sections, template
            )
            
            # 추가 컨텍스트 통합
            if additional_context:
                optimized_context = self._integrate_additional_context(
                    optimized_context, additional_context
                )
            
            # 메타데이터 추가
            final_context = self._add_context_metadata(
                optimized_context, search_results, template
            )
            
            self.logger.info(f"컨텍스트 구성 완료: {context_type} 타입, "
                           f"{len(search_results)}개 검색 결과 활용")
            
            return final_context
            
        except Exception as e:
            self.logger.error(f"컨텍스트 구성 오류: {e}")
            return self.error_handler.handle_error(e, default_return={
                'formatted_context': "",
                'sections': {},
                'metadata': {'error': str(e)}
            })
    
    def build_multi_turn_context(self,
                               current_results: List[SearchResult],
                               conversation_history: List[Dict[str, Any]],
                               context_type: str = "conversation") -> Dict[str, Any]:
        """
        다중 턴 대화를 위한 컨텍스트 구성
        
        Args:
            current_results: 현재 검색 결과
            conversation_history: 대화 이력
            context_type: 컨텍스트 유형
            
        Returns:
            대화 컨텍스트
        """
        try:
            # 대화 이력에서 관련 정보 추출
            historical_context = self._extract_historical_context(conversation_history)
            
            # 현재 검색 결과와 과거 컨텍스트 결합
            combined_results = self._combine_with_history(
                current_results, historical_context
            )
            
            # 일반 컨텍스트 구성
            base_context = self.build_context(
                combined_results, context_type, 
                additional_context={'conversation_history': historical_context}
            )
            
            # 대화 특화 섹션 추가
            conversation_sections = self._build_conversation_sections(
                conversation_history, current_results
            )
            
            base_context['sections'].update(conversation_sections)
            
            return base_context
            
        except Exception as e:
            self.logger.error(f"다중 턴 컨텍스트 구성 오류: {e}")
            return self.build_context(current_results, context_type)
    
    def build_analysis_context(self,
                              search_results: List[SearchResult],
                              data_info: Dict[str, Any],
                              analysis_goal: str) -> Dict[str, Any]:
        """
        데이터 분석을 위한 특화 컨텍스트 구성
        
        Args:
            search_results: 검색 결과
            data_info: 데이터 정보
            analysis_goal: 분석 목표
            
        Returns:
            분석 컨텍스트
        """
        try:
            # 분석 관련 검색 결과 필터링
            analysis_results = self._filter_analysis_relevant(
                search_results, data_info, analysis_goal
            )
            
            # 데이터 특성 기반 컨텍스트 추가
            data_context = self._build_data_context(data_info)
            
            # 분석 목표 기반 방법론 컨텍스트
            methodology_context = self._build_methodology_context(
                analysis_goal, analysis_results
            )
            
            # 기본 컨텍스트 구성
            base_context = self.build_context(
                analysis_results, "analysis",
                additional_context={
                    'data_info': data_context,
                    'methodology': methodology_context,
                    'analysis_goal': analysis_goal
                }
            )
            
            return base_context
            
        except Exception as e:
            self.logger.error(f"분석 컨텍스트 구성 오류: {e}")
            return self.build_context(search_results, "analysis")
    
    def _initialize_templates(self) -> Dict[str, ContextTemplate]:
        """컨텍스트 템플릿 초기화"""
        templates = {}
        
        # 일반 분석 템플릿
        templates['analysis'] = ContextTemplate(
            name="analysis",
            sections=[
                "statistical_concepts",
                "methodology", 
                "code_examples",
                "domain_knowledge",
                "best_practices"
            ],
            max_length=3500,
            priority_weights={
                "statistical_concepts": 1.0,
                "methodology": 0.9,
                "code_examples": 0.8,
                "domain_knowledge": 0.7,
                "best_practices": 0.6
            },
            format_style="structured"
        )
        
        # 해석 템플릿
        templates['interpretation'] = ContextTemplate(
            name="interpretation",
            sections=[
                "statistical_concepts",
                "domain_knowledge",
                "interpretation_guidelines",
                "examples",
                "best_practices"
            ],
            max_length=3000,
            priority_weights={
                "statistical_concepts": 1.0,
                "domain_knowledge": 0.9,
                "interpretation_guidelines": 0.8,
                "examples": 0.7,
                "best_practices": 0.6
            },
            format_style="narrative"
        )
        
        # 계획 수립 템플릿
        templates['planning'] = ContextTemplate(
            name="planning",
            sections=[
                "methodology",
                "workflow_guidelines",
                "best_practices",
                "statistical_concepts",
                "domain_knowledge"
            ],
            max_length=3200,
            priority_weights={
                "methodology": 1.0,
                "workflow_guidelines": 0.9,
                "best_practices": 0.8,
                "statistical_concepts": 0.7,
                "domain_knowledge": 0.6
            },
            format_style="procedural"
        )
        
        # 대화 템플릿
        templates['conversation'] = ContextTemplate(
            name="conversation",
            sections=[
                "conversation_context",
                "relevant_knowledge",
                "domain_knowledge",
                "examples"
            ],
            max_length=2800,
            priority_weights={
                "conversation_context": 1.0,
                "relevant_knowledge": 0.8,
                "domain_knowledge": 0.6,
                "examples": 0.5
            },
            format_style="conversational"
        )
        
        # 기본 템플릿
        templates['general'] = ContextTemplate(
            name="general",
            sections=[
                "relevant_knowledge",
                "domain_knowledge",
                "examples",
                "best_practices"
            ],
            max_length=3000,
            priority_weights={
                "relevant_knowledge": 1.0,
                "domain_knowledge": 0.8,
                "examples": 0.6,
                "best_practices": 0.5
            },
            format_style="informative"
        )
        
        return templates
    
    def _select_template(self, context_type: str) -> ContextTemplate:
        """컨텍스트 유형에 따른 템플릿 선택"""
        return self.templates.get(context_type, self.templates['general'])
    
    def _categorize_search_results(self, 
                                  search_results: List[SearchResult]) -> Dict[str, List[SearchResult]]:
        """검색 결과를 카테고리별로 분류"""
        categorized = defaultdict(list)
        
        for result in search_results:
            knowledge_type = result.knowledge_type
            
            # 지식 유형을 섹션으로 매핑
            if knowledge_type == 'statistical_concepts':
                categorized['statistical_concepts'].append(result)
            elif knowledge_type == 'business_domains':
                categorized['domain_knowledge'].append(result)
            elif knowledge_type == 'code_templates':
                categorized['code_examples'].append(result)
            elif knowledge_type == 'workflow_guidelines':
                categorized['methodology'].append(result)
            else:
                categorized['relevant_knowledge'].append(result)
        
        # 점수순으로 정렬
        for category in categorized:
            categorized[category].sort(key=lambda x: x.score, reverse=True)
        
        return dict(categorized)
    
    def _build_context_sections(self,
                               categorized_results: Dict[str, List[SearchResult]],
                               template: ContextTemplate,
                               user_query: str) -> Dict[str, str]:
        """템플릿에 따라 컨텍스트 섹션 구성"""
        sections = {}
        
        for section_name in template.sections:
            if section_name in categorized_results:
                results = categorized_results[section_name]
                formatted_section = self._format_section(
                    section_name, results, template.format_style, user_query
                )
                sections[section_name] = formatted_section
        
        return sections
    
    def _format_section(self,
                       section_name: str,
                       results: List[SearchResult],
                       format_style: str,
                       user_query: str) -> str:
        """섹션 포맷팅"""
        if not results:
            return ""
        
        section_content = []
        
        # 섹션 헤더
        if format_style == "structured":
            section_content.append(f"## {section_name.replace('_', ' ').title()}")
        elif format_style == "narrative":
            section_content.append(f"### {section_name.replace('_', ' ').title()} 관련 정보:")
        elif format_style == "procedural":
            section_content.append(f"### {section_name.replace('_', ' ').title()} 가이드라인:")
        
        # 결과 포맷팅
        for i, result in enumerate(results[:5]):  # 최대 5개 결과만 포함
            formatter = self.formatters.get(
                result.knowledge_type, 
                self._format_general_knowledge
            )
            formatted_content = formatter(result, user_query)
            
            if formatted_content:
                if format_style == "structured":
                    section_content.append(f"{i+1}. {formatted_content}")
                else:
                    section_content.append(formatted_content)
        
        return "\n".join(section_content)
    
    def _format_statistical_concept(self, result: SearchResult, user_query: str) -> str:
        """통계 개념 포맷팅"""
        content = result.content
        
        # 수식이나 기호가 포함된 경우 특별 처리
        if any(symbol in content for symbol in ['=', '∑', 'μ', 'σ', 'α', 'β']):
            return f"**통계 개념**: {result.snippet}\n   상세: {content[:200]}..."
        else:
            return f"**{result.metadata.get('concept_name', '통계 개념')}**: {result.snippet}"
    
    def _format_domain_knowledge(self, result: SearchResult, user_query: str) -> str:
        """도메인 지식 포맷팅"""
        domain = result.metadata.get('domain', '일반')
        return f"**{domain} 도메인**: {result.snippet}"
    
    def _format_code_template(self, result: SearchResult, user_query: str) -> str:
        """코드 템플릿 포맷팅"""
        language = result.metadata.get('language', 'Python')
        function_name = result.metadata.get('function_name', '함수')
        
        # 코드 스니펫 정리
        code_snippet = self._extract_code_snippet(result.content)
        
        return f"**{language} {function_name} 예시**:\n```python\n{code_snippet}\n```"
    
    def _format_workflow_guideline(self, result: SearchResult, user_query: str) -> str:
        """워크플로우 가이드라인 포맷팅"""
        step_number = result.metadata.get('step_number', '')
        step_prefix = f"단계 {step_number}: " if step_number else ""
        
        return f"**{step_prefix}워크플로우**: {result.snippet}"
    
    def _format_general_knowledge(self, result: SearchResult, user_query: str) -> str:
        """일반 지식 포맷팅"""
        return f"**관련 정보**: {result.snippet}"
    
    def _extract_code_snippet(self, content: str, max_lines: int = 10) -> str:
        """코드 스니펫 추출"""
        lines = content.split('\n')
        
        # 실제 코드 라인만 필터링 (주석과 빈 줄 제외)
        code_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('"""'):
                code_lines.append(line)
            if len(code_lines) >= max_lines:
                break
        
        return '\n'.join(code_lines)
    
    def _optimize_context_length(self,
                                context_sections: Dict[str, str],
                                template: ContextTemplate) -> Dict[str, str]:
        """컨텍스트 길이 최적화"""
        # 현재 길이 계산 (대략적인 토큰 수)
        current_length = sum(len(section.split()) for section in context_sections.values())
        
        if current_length <= template.max_length:
            return context_sections
        
        # 우선순위에 따라 섹션 축소
        optimized_sections = {}
        remaining_length = template.max_length
        
        # 우선순위 순으로 정렬
        sorted_sections = sorted(
            context_sections.items(),
            key=lambda x: template.priority_weights.get(x[0], 0.5),
            reverse=True
        )
        
        for section_name, section_content in sorted_sections:
            section_words = section_content.split()
            section_length = len(section_words)
            
            if section_length <= remaining_length:
                optimized_sections[section_name] = section_content
                remaining_length -= section_length
            else:
                # 섹션 축소
                if remaining_length > 50:  # 최소 50단어는 유지
                    truncated_content = ' '.join(section_words[:remaining_length-10]) + "..."
                    optimized_sections[section_name] = truncated_content
                break
        
        return optimized_sections
    
    def _integrate_additional_context(self,
                                    base_context: Dict[str, str],
                                    additional_context: Dict[str, Any]) -> Dict[str, str]:
        """추가 컨텍스트 통합"""
        integrated_context = base_context.copy()
        
        for key, value in additional_context.items():
            if key == 'data_info':
                integrated_context['data_context'] = self._format_data_info(value)
            elif key == 'methodology':
                if 'methodology' in integrated_context:
                    integrated_context['methodology'] += f"\n\n{value}"
                else:
                    integrated_context['methodology'] = value
            elif key == 'conversation_history':
                integrated_context['conversation_context'] = self._format_conversation_history(value)
            elif isinstance(value, str):
                integrated_context[key] = value
        
        return integrated_context
    
    def _add_context_metadata(self,
                             context_sections: Dict[str, str],
                             search_results: List[SearchResult],
                             template: ContextTemplate) -> Dict[str, Any]:
        """컨텍스트 메타데이터 추가"""
        # 포맷된 컨텍스트 생성
        formatted_context = self._format_final_context(context_sections, template.format_style)
        
        metadata = {
            'template_used': template.name,
            'sections_included': list(context_sections.keys()),
            'total_search_results': len(search_results),
            'context_length_words': len(formatted_context.split()),
            'build_timestamp': datetime.now().isoformat(),
            'knowledge_sources': list(set(result.source for result in search_results))
        }
        
        return {
            'formatted_context': formatted_context,
            'sections': context_sections,
            'metadata': metadata
        }
    
    def _format_final_context(self, sections: Dict[str, str], format_style: str) -> str:
        """최종 컨텍스트 포맷팅"""
        if format_style == "structured":
            return "\n\n".join(sections.values())
        elif format_style == "narrative":
            return "\n\n".join(sections.values())
        elif format_style == "procedural":
            formatted_sections = []
            for i, (name, content) in enumerate(sections.items(), 1):
                formatted_sections.append(f"{i}. {content}")
            return "\n\n".join(formatted_sections)
        elif format_style == "conversational":
            return " ".join(sections.values())
        else:  # informative
            return "\n\n".join(sections.values())
    
    def _extract_historical_context(self, 
                                   conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """대화 이력에서 관련 컨텍스트 추출"""
        historical_context = {
            'previous_questions': [],
            'previous_analyses': [],
            'user_preferences': {},
            'domain_focus': []
        }
        
        for turn in conversation_history[-5:]:  # 최근 5턴만 고려
            if 'user_message' in turn:
                historical_context['previous_questions'].append(turn['user_message'])
            
            if 'analysis_results' in turn:
                historical_context['previous_analyses'].append(turn['analysis_results'])
            
            if 'user_preferences' in turn:
                historical_context['user_preferences'].update(turn['user_preferences'])
            
            if 'domain' in turn:
                historical_context['domain_focus'].append(turn['domain'])
        
        return historical_context
    
    def _combine_with_history(self,
                             current_results: List[SearchResult],
                             historical_context: Dict[str, Any]) -> List[SearchResult]:
        """현재 검색 결과와 과거 컨텍스트 결합"""
        # 단순히 현재 결과를 반환 (실제로는 더 정교한 결합 로직 구현 가능)
        # 추후 개선: 과거 질문과 관련된 추가 검색 결과 생성
        return current_results
    
    def _build_conversation_sections(self,
                                   conversation_history: List[Dict[str, Any]],
                                   current_results: List[SearchResult]) -> Dict[str, str]:
        """대화 특화 섹션 구성"""
        sections = {}
        
        # 이전 질문 요약
        if conversation_history:
            recent_questions = [
                turn.get('user_message', '') 
                for turn in conversation_history[-3:]
                if 'user_message' in turn
            ]
            if recent_questions:
                sections['previous_context'] = "이전 질문들:\n" + "\n".join(
                    f"- {q}" for q in recent_questions if q
                )
        
        return sections
    
    def _filter_analysis_relevant(self,
                                 search_results: List[SearchResult],
                                 data_info: Dict[str, Any],
                                 analysis_goal: str) -> List[SearchResult]:
        """분석 관련 검색 결과 필터링"""
        # 데이터 타입과 분석 목표에 따른 필터링
        relevant_results = []
        
        data_types = data_info.get('column_types', {})
        has_numerical = any(dtype in ['int64', 'float64'] for dtype in data_types.values())
        has_categorical = any(dtype in ['object', 'category'] for dtype in data_types.values())
        
        for result in search_results:
            # 데이터 타입 관련성 확인
            content_lower = result.content.lower()
            is_relevant = False
            
            if has_numerical and any(term in content_lower for term in 
                                   ['numerical', 'continuous', 'regression', 't-test', 'correlation']):
                is_relevant = True
            
            if has_categorical and any(term in content_lower for term in 
                                     ['categorical', 'chi-square', 'anova', 'frequency']):
                is_relevant = True
            
            # 분석 목표 관련성 확인
            if analysis_goal.lower() in content_lower:
                is_relevant = True
            
            if is_relevant:
                relevant_results.append(result)
        
        return relevant_results
    
    def _build_data_context(self, data_info: Dict[str, Any]) -> str:
        """데이터 특성 기반 컨텍스트 구성"""
        context_parts = []
        
        # 데이터 크기
        if 'shape' in data_info:
            rows, cols = data_info['shape']
            context_parts.append(f"데이터 크기: {rows}행 {cols}열")
        
        # 변수 타입
        if 'column_types' in data_info:
            type_counts = {}
            for dtype in data_info['column_types'].values():
                type_counts[dtype] = type_counts.get(dtype, 0) + 1
            
            type_summary = ", ".join(f"{dtype}: {count}개" for dtype, count in type_counts.items())
            context_parts.append(f"변수 타입: {type_summary}")
        
        # 결측값 정보
        if 'missing_values' in data_info:
            missing_info = data_info['missing_values']
            if any(missing_info.values()):
                context_parts.append("결측값이 존재함")
        
        return "\n".join(context_parts)
    
    def _build_methodology_context(self, 
                                  analysis_goal: str,
                                  analysis_results: List[SearchResult]) -> str:
        """분석 목표 기반 방법론 컨텍스트"""
        methodology_parts = []
        
        # 분석 목표 기반 권장 방법
        goal_lower = analysis_goal.lower()
        
        if 'compare' in goal_lower or 'difference' in goal_lower:
            methodology_parts.append("그룹 비교 분석을 위한 방법론 고려")
        
        if 'relationship' in goal_lower or 'correlation' in goal_lower:
            methodology_parts.append("관계 분석을 위한 상관분석 또는 회귀분석 고려")
        
        if 'predict' in goal_lower or 'forecast' in goal_lower:
            methodology_parts.append("예측 모델링을 위한 회귀분석 또는 머신러닝 기법 고려")
        
        # 검색 결과에서 방법론 정보 추출
        for result in analysis_results[:3]:
            if result.knowledge_type == 'statistical_concepts':
                methodology_parts.append(f"참고 방법: {result.snippet}")
        
        return "\n".join(methodology_parts)
    
    def _format_data_info(self, data_info: Dict[str, Any]) -> str:
        """데이터 정보 포맷팅"""
        return self._build_data_context(data_info)
    
    def _format_conversation_history(self, history: Dict[str, Any]) -> str:
        """대화 이력 포맷팅"""
        context_parts = []
        
        if 'previous_questions' in history:
            questions = history['previous_questions'][-3:]  # 최근 3개
            if questions:
                context_parts.append("이전 질문:")
                for i, q in enumerate(questions, 1):
                    context_parts.append(f"{i}. {q}")
        
        if 'user_preferences' in history and history['user_preferences']:
            prefs = history['user_preferences']
            pref_strs = [f"{k}: {v}" for k, v in prefs.items()]
            context_parts.append(f"사용자 선호: {', '.join(pref_strs)}")
        
        return "\n".join(context_parts)
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """컨텍스트 빌더 통계 정보"""
        return {
            'max_context_length': self.max_context_length,
            'available_templates': list(self.templates.keys()),
            'section_priorities': self.section_priorities,
            'supported_formatters': list(self.formatters.keys())
        } 