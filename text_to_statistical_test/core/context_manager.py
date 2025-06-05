"""
ContextManager: Enhanced RAG 기반 Agent의 장기 기억 및 작업 메모리 관리

LLM과의 상호작용, 주요 분석 단계의 이력, Enhanced RAG 시스템의 비즈니스 컨텍스트 및 
DB 스키마 정보를 관리하고, 토큰 제한을 고려하여 이력을 요약하거나 필터링하여 
LLM에 전달할 컨텍스트를 최적화합니다.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class ContextManager:
    """
    Enhanced RAG 기반 Agent의 장기 기억 및 작업 메모리 관리자
    
    대화 이력, 분석 과정, 비즈니스 컨텍스트, DB 스키마 정보를 관리하며, 
    LLM 토큰 제한을 고려한 컨텍스트 최적화 기능을 제공합니다.
    """
    
    def __init__(self, llm_client, max_history_items: int = 20, 
                 summarization_trigger_count: int = 10, 
                 context_token_limit: int = 3000):
        """
        Enhanced RAG 기반 ContextManager 초기화
        
        Args:
            llm_client: 요약용 LLM 클라이언트
            max_history_items: 최대 저장할 상호작용 수
            summarization_trigger_count: 요약 트리거 수
            context_token_limit: LLM 전달 컨텍스트 토큰 제한
        """
        self.llm_client = llm_client
        self.max_history_items = max_history_items
        self.summarization_trigger_count = summarization_trigger_count
        self.context_token_limit = context_token_limit
        
        # 상호작용 이력 저장
        self._interaction_history: List[Dict] = []
        
        # Enhanced RAG 컨텍스트 저장
        self._business_context: Dict = {}
        self._schema_context: Dict = {}
        self._rag_search_history: List[Dict] = []
        
        # 분석 상태 컨텍스트
        self._analysis_context: Dict = {
            'natural_language_request': '',
            'selected_method': '',
            'data_summary': {},
            'ai_recommendations': []
        }
        
        # 요약 캐시
        self._summary_cache: str = ""
        self._last_summarized_index: int = 0
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
    
    def add_interaction(self, role: str, content: str, node_id: str, 
                       metadata: Optional[Dict] = None):
        """
        새로운 상호작용을 이력에 추가합니다.
        
        Args:
            role: 역할 ('user', 'assistant', 'system', 'rag')
            content: 상호작용 내용
            node_id: 현재 워크플로우 노드 ID
            metadata: 추가 메타데이터 (RAG 검색 결과 등)
        """
        interaction = {
            'role': role,
            'content': content,
            'node_id': node_id,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self._interaction_history.append(interaction)
        self.logger.debug(f"상호작용 추가: {role} at {node_id}")
        
        # 요약 트리거 확인
        if (len(self._interaction_history) - self._last_summarized_index 
            >= self.summarization_trigger_count):
            self._trigger_summarization()
        
        # 이력 정리
        if len(self._interaction_history) > self.max_history_items:
            self._prune_history()
    
    def add_business_context(self, context_data: Dict):
        """
        비즈니스 컨텍스트 정보를 추가합니다.
        
        Args:
            context_data: 비즈니스 컨텍스트 데이터
        """
        self._business_context.update(context_data)
        self.logger.debug("비즈니스 컨텍스트 업데이트됨")
        
        # RAG 검색 이력에 추가
        self._rag_search_history.append({
            'type': 'business_context',
            'timestamp': datetime.now().isoformat(),
            'data': context_data
        })
    
    def add_schema_context(self, schema_data: Dict):
        """
        DB 스키마 컨텍스트 정보를 추가합니다.
        
        Args:
            schema_data: DB 스키마 컨텍스트 데이터
        """
        self._schema_context.update(schema_data)
        self.logger.debug("스키마 컨텍스트 업데이트됨")
        
        # RAG 검색 이력에 추가
        self._rag_search_history.append({
            'type': 'schema_context',
            'timestamp': datetime.now().isoformat(),
            'data': schema_data
        })
    
    def update_analysis_context(self, key: str, value: Any):
        """
        분석 상태 컨텍스트를 업데이트합니다.
        
        Args:
            key: 컨텍스트 키
            value: 업데이트할 값
        """
        self._analysis_context[key] = value
        self.logger.debug(f"분석 컨텍스트 업데이트: {key}")
    
    def get_optimized_context(self, current_task_prompt: str, 
                            required_recent_interactions: int = 5,
                            include_rag_context: bool = True) -> str:
        """
        LLM에 전달할 최적화된 컨텍스트 문자열을 반환합니다.
        
        Args:
            current_task_prompt: 현재 작업 프롬프트
            required_recent_interactions: 포함할 최근 상호작용 수
            include_rag_context: RAG 컨텍스트 포함 여부
            
        Returns:
            str: 최적화된 컨텍스트 문자열
        """
        context_parts = []
        
        # 1. Enhanced RAG 컨텍스트 추가
        if include_rag_context:
            rag_context = self._build_rag_context()
            if rag_context:
                context_parts.append("=== Enhanced RAG 컨텍스트 ===")
                context_parts.append(rag_context)
                context_parts.append("")
        
        # 2. 요약된 이전 이력 추가
        if self._summary_cache:
            context_parts.append("=== 이전 분석 과정 요약 ===")
            context_parts.append(self._summary_cache)
            context_parts.append("")
        
        # 3. 최근 상호작용 추가
        recent_interactions = self._get_recent_interactions(required_recent_interactions)
        if recent_interactions:
            context_parts.append("=== 최근 상호작용 ===")
            for interaction in recent_interactions:
                formatted_interaction = self._format_interaction(interaction)
                context_parts.append(formatted_interaction)
            context_parts.append("")
        
        # 4. 현재 분석 상태 추가
        analysis_state = self._build_analysis_state_summary()
        if analysis_state:
            context_parts.append("=== 현재 분석 상태 ===")
            context_parts.append(analysis_state)
            context_parts.append("")
        
        # 5. 현재 작업 추가
        context_parts.append("=== 현재 작업 ===")
        context_parts.append(current_task_prompt)
        
        # 전체 컨텍스트 생성
        full_context = "\n".join(context_parts)
        
        # 토큰 제한 확인 및 조정
        optimized_context = self._optimize_for_token_limit(full_context)
        
        return optimized_context
    
    def get_rag_context_summary(self) -> Dict[str, Any]:
        """
        Enhanced RAG 컨텍스트 요약을 반환합니다.
        
        Returns:
            Dict: RAG 컨텍스트 요약 정보
        """
        return {
            'business_context_keys': list(self._business_context.keys()),
            'schema_context_keys': list(self._schema_context.keys()),
            'rag_searches_count': len(self._rag_search_history),
            'last_business_update': self._get_last_context_update('business_context'),
            'last_schema_update': self._get_last_context_update('schema_context')
        }
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        현재까지의 분석 과정 요약을 반환합니다.
        
        Returns:
            Dict: 분석 과정 요약 정보
        """
        total_interactions = len(self._interaction_history)
        
        # 역할별 상호작용 수 계산
        role_counts = {}
        node_visits = {}
        
        for interaction in self._interaction_history:
            role = interaction['role']
            node_id = interaction['node_id']
            
            role_counts[role] = role_counts.get(role, 0) + 1
            node_visits[node_id] = node_visits.get(node_id, 0) + 1
        
        # 분석 진행 단계 파악
        visited_nodes = list(node_visits.keys())
        
        return {
            'total_interactions': total_interactions,
            'role_distribution': role_counts,
            'visited_nodes': visited_nodes,
            'current_summary': self._summary_cache,
            'analysis_context': self._analysis_context,
            'rag_context_summary': self.get_rag_context_summary(),
            'analysis_start_time': (self._interaction_history[0]['timestamp'] 
                                  if self._interaction_history else None),
            'last_interaction_time': (self._interaction_history[-1]['timestamp'] 
                                    if self._interaction_history else None)
        }
    
    def _build_rag_context(self) -> str:
        """Enhanced RAG 컨텍스트를 구축합니다."""
        rag_parts = []
        
        # 비즈니스 컨텍스트
        if self._business_context:
            rag_parts.append("🏢 비즈니스 컨텍스트:")
            for key, value in self._business_context.items():
                if isinstance(value, dict) and 'summary' in value:
                    rag_parts.append(f"  • {key}: {value['summary']}")
                elif isinstance(value, str) and len(value) < 200:
                    rag_parts.append(f"  • {key}: {value}")
        
        # DB 스키마 컨텍스트
        if self._schema_context:
            if rag_parts:
                rag_parts.append("")
            rag_parts.append("🗄️ DB 스키마 컨텍스트:")
            for key, value in self._schema_context.items():
                if isinstance(value, dict) and 'summary' in value:
                    rag_parts.append(f"  • {key}: {value['summary']}")
                elif isinstance(value, str) and len(value) < 200:
                    rag_parts.append(f"  • {key}: {value}")
        
        return "\n".join(rag_parts)
    
    def _build_analysis_state_summary(self) -> str:
        """현재 분석 상태 요약을 구축합니다."""
        state_parts = []
        
        if self._analysis_context['natural_language_request']:
            state_parts.append(f"📝 사용자 요청: {self._analysis_context['natural_language_request']}")
        
        if self._analysis_context['selected_method']:
            state_parts.append(f"🔬 선택된 방법: {self._analysis_context['selected_method']}")
        
        if self._analysis_context['ai_recommendations']:
            rec_count = len(self._analysis_context['ai_recommendations'])
            state_parts.append(f"🤖 AI 추천: {rec_count}개 방법 제시됨")
        
        return "\n".join(state_parts)
    
    def _get_last_context_update(self, context_type: str) -> Optional[str]:
        """특정 컨텍스트 유형의 마지막 업데이트 시간을 반환합니다."""
        for search in reversed(self._rag_search_history):
            if search['type'] == context_type:
                return search['timestamp']
        return None
    
    def _get_recent_interactions(self, count: int) -> List[Dict]:
        """최근 상호작용들을 반환합니다."""
        if count <= 0:
            return []
        
        return self._interaction_history[-count:] if self._interaction_history else []
    
    def _format_interaction(self, interaction: Dict) -> str:
        """단일 상호작용을 형식화된 문자열로 변환합니다."""
        role = interaction['role']
        content = interaction['content']
        node_id = interaction['node_id']
        timestamp = interaction['timestamp']
        
        # 역할별 아이콘
        role_icons = {
            'user': '👤',
            'assistant': '🤖',
            'system': '⚙️'
        }
        
        icon = role_icons.get(role, '•')
        
        # 내용이 너무 길면 축약
        if len(content) > 200:
            content = content[:197] + "..."
        
        return f"{icon} [{node_id}] {role}: {content}"
    
    def _trigger_summarization(self):
        """이력 요약을 트리거합니다."""
        if len(self._interaction_history) <= self._last_summarized_index:
            return
        
        # 요약할 상호작용들 선택
        interactions_to_summarize = self._interaction_history[
            self._last_summarized_index:
            self._last_summarized_index + self.summarization_trigger_count
        ]
        
        if not interactions_to_summarize:
            return
        
        try:
            # 새로운 요약 생성
            new_summary = self._summarize_interactions(interactions_to_summarize)
            
            # 기존 요약과 결합
            if self._summary_cache:
                combined_summary = f"{self._summary_cache}\n\n{new_summary}"
                # 결합된 요약이 너무 길면 다시 요약
                if len(combined_summary) > 1000:
                    self._summary_cache = self._summarize_text(combined_summary)
                else:
                    self._summary_cache = combined_summary
            else:
                self._summary_cache = new_summary
            
            # 요약된 인덱스 업데이트
            self._last_summarized_index += len(interactions_to_summarize)
            
            self.logger.info(f"이력 요약 완료: {len(interactions_to_summarize)}개 상호작용")
            
        except Exception as e:
            self.logger.error(f"이력 요약 실패: {e}")
    
    def _summarize_interactions(self, interactions: List[Dict]) -> str:
        """주어진 상호작용들을 요약합니다."""
        # 상호작용들을 텍스트로 변환
        interaction_texts = []
        for interaction in interactions:
            formatted = self._format_interaction(interaction)
            interaction_texts.append(formatted)
        
        interactions_text = "\n".join(interaction_texts)
        
        # 요약 프롬프트 생성
        summary_prompt = f"""
다음은 통계 분석 워크플로우의 상호작용 기록입니다. 
주요 내용과 결정사항을 간결하게 요약해주세요:

{interactions_text}

요약:
"""
        
        try:
            # LLM을 사용한 요약
            summary = self.llm_client.generate_text(
                prompt=summary_prompt,
                temperature=0.3
            )
            return summary.strip()
            
        except Exception as e:
            self.logger.error(f"LLM 요약 실패: {e}")
            # 폴백: 단순 요약
            return self._simple_summarize(interactions)
    
    def _summarize_text(self, text: str) -> str:
        """긴 텍스트를 요약합니다."""
        summary_prompt = f"""
다음 텍스트를 핵심 내용만 포함하여 간결하게 요약해주세요:

{text}

요약:
"""
        
        try:
            summary = self.llm_client.generate_text(
                prompt=summary_prompt,
                temperature=0.3
            )
            return summary.strip()
        except Exception as e:
            self.logger.error(f"텍스트 요약 실패: {e}")
            # 폴백: 단순 축약
            return text[:500] + "..." if len(text) > 500 else text
    
    def _simple_summarize(self, interactions: List[Dict]) -> str:
        """간단한 요약 (LLM 실패시 폴백)"""
        if not interactions:
            return ""
        
        # 노드별 그룹화
        nodes = set(interaction['node_id'] for interaction in interactions)
        
        summary_parts = []
        summary_parts.append(f"처리된 노드: {', '.join(sorted(nodes))}")
        summary_parts.append(f"총 상호작용: {len(interactions)}개")
        
        # 주요 결정사항 추출
        decisions = []
        for interaction in interactions:
            if interaction['role'] == 'user' and len(interaction['content']) < 50:
                decisions.append(interaction['content'])
        
        if decisions:
            summary_parts.append(f"주요 결정: {', '.join(decisions[:3])}")
        
        return " | ".join(summary_parts)
    
    def _optimize_for_token_limit(self, context: str) -> str:
        """토큰 제한에 맞게 컨텍스트를 최적화합니다."""
        # 간단한 토큰 추정 (실제로는 tokenizer 사용 권장)
        estimated_tokens = len(context.split()) * 1.3
        
        if estimated_tokens <= self.context_token_limit:
            return context
        
        # 토큰 제한 초과시 축약
        self.logger.warning(f"컨텍스트 토큰 제한 초과: {estimated_tokens:.0f} > {self.context_token_limit}")
        
        # 최근 상호작용 수를 줄여서 재생성
        lines = context.split('\n')
        
        # 현재 작업 부분은 유지
        current_task_index = -1
        for i, line in enumerate(lines):
            if "=== 현재 작업 ===" in line:
                current_task_index = i
                break
        
        if current_task_index > 0:
            # 현재 작업 이전 부분을 축약
            before_current = lines[:current_task_index]
            current_and_after = lines[current_task_index:]
            
            # 이전 부분을 절반으로 축약
            truncated_before = before_current[:len(before_current)//2]
            truncated_before.append("... (중간 내용 생략) ...")
            
            optimized_lines = truncated_before + current_and_after
            return '\n'.join(optimized_lines)
        
        # 폴백: 전체를 절반으로 축약
        return '\n'.join(lines[:len(lines)//2]) + "\n... (내용 생략) ..."
    
    def _prune_history(self):
        """오래된 상호작용 기록을 정리합니다."""
        if len(self._interaction_history) <= self.max_history_items:
            return
        
        # 가장 오래된 상호작용들 제거 (요약된 부분 제외)
        items_to_remove = len(self._interaction_history) - self.max_history_items
        
        # 요약되지 않은 부분부터 제거
        removal_start = max(0, self._last_summarized_index)
        removal_end = min(removal_start + items_to_remove, len(self._interaction_history))
        
        if removal_end > removal_start:
            removed_items = self._interaction_history[removal_start:removal_end]
            self._interaction_history = (
                self._interaction_history[:removal_start] + 
                self._interaction_history[removal_end:]
            )
            
            # 인덱스 조정
            self._last_summarized_index = max(0, self._last_summarized_index - len(removed_items))
            
            self.logger.info(f"이력 정리 완료: {len(removed_items)}개 상호작용 제거")
    
    def reset_context(self):
        """컨텍스트를 초기화합니다."""
        self._interaction_history.clear()
        self._summary_cache = ""
        self._last_summarized_index = 0
        self.logger.info("컨텍스트 초기화 완료") 