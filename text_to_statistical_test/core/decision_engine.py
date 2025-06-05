"""
DecisionEngine: Enhanced RAG 기반 워크플로우 네비게이터

현재 노드의 처리 결과와 전환 규칙을 비교하여 Enhanced RAG 시스템을 포함한
Multi-turn 워크플로우에서 다음으로 진행할 노드 ID를 결정합니다.
"""

import logging
import re
from typing import Dict, Any, Optional, List


class DecisionEngine:
    """
    Enhanced RAG 기반 워크플로우 네비게이터
    
    노드 처리 결과를 기반으로 Enhanced RAG 시스템을 포함한 워크플로우의 
    다음 단계를 결정하는 의사결정 엔진입니다.
    """
    
    def __init__(self):
        """DecisionEngine 초기화"""
        self.logger = logging.getLogger(__name__)
    
    def determine_next_node(self, current_node_details: Dict, execution_outcome: Any, 
                          user_response: str = None) -> Optional[str]:
        """
        현재 노드의 처리 결과를 바탕으로 다음 노드를 결정합니다.
        
        Args:
            current_node_details: 현재 노드의 전체 정보
            execution_outcome: 가장 최근 작업의 결과
            user_response: 사용자가 제공한 응답
            
        Returns:
            str: 다음 노드 ID, 조건에 맞는 노드가 없으면 None
        """
        transitions = current_node_details.get('transitions', [])
        
        if not transitions:
            self.logger.info("전환 규칙이 없는 노드입니다.")
            return None
        
        # 각 전환 조건을 순서대로 평가
        for transition in transitions:
            condition = transition.get('condition', '')
            next_node = transition.get('next', '')
            
            self.logger.debug(f"조건 평가 중: '{condition}' -> '{next_node}'")
            
            if self._evaluate_condition(condition, execution_outcome, user_response):
                self.logger.info(f"조건 만족: '{condition}' -> 다음 노드: '{next_node}'")
                return next_node
        
        # 만족하는 조건이 없는 경우
        self.logger.warning("만족하는 전환 조건이 없습니다.")
        return None
    
    def _evaluate_condition(self, condition_string: str, outcome: Any, user_input: str) -> bool:
        """
        조건 문자열을 파싱하고 평가합니다.
        
        Args:
            condition_string: 평가할 조건 문자열
            outcome: 노드 처리 결과
            user_input: 사용자 입력
            
        Returns:
            bool: 조건 만족 여부
        """
        if not condition_string:
            return False
        
        condition_lower = condition_string.lower().strip()
        
        # 1. 자동 진행 조건 (Enhanced RAG 워크플로우)
        if self._is_auto_condition(condition_lower):
            return True
        
        # 2. 사용자 응답 기반 조건
        if user_input and self._is_user_response_condition(condition_lower, user_input):
            return True
        
        # 3. RAG 시스템 관련 조건
        if self._is_rag_condition(condition_lower, outcome):
            return True
        
        # 4. AI 추천 관련 조건
        if self._is_ai_recommendation_condition(condition_lower, outcome):
            return True
        
        # 5. 데이터 상태 기반 조건
        if self._is_data_state_condition(condition_lower, outcome):
            return True
        
        # 6. 분석 결과 기반 조건
        if self._is_analysis_result_condition(condition_lower, outcome):
            return True
        
        # 7. 복합 조건 (AND, OR 포함)
        if self._is_complex_condition(condition_lower, outcome, user_input):
            return True
        
        return False
    
    def _is_auto_condition(self, condition: str) -> bool:
        """자동 진행 조건인지 확인"""
        auto_keywords = [
            '시스템 준비 완료', '데이터 선택 및 로딩 완료', '데이터 검증 통과',
            '유효한 자연어 요청 입력 완료', 'rag 검색 준비 완료', 'rag 검색 완료',
            '비즈니스 지식 검색 완료', '스키마 구조 검색 완료', 'ai 추천 생성 완료',
            '방법 확정 완료', '전처리 완료', '분석 실행 성공', '해석 완료', 
            '보고서 생성 완료', '자동 시작', '진행', '완료', '자동', 'auto', 'proceed'
        ]
        return any(keyword in condition for keyword in auto_keywords)
    
    def _is_user_response_condition(self, condition: str, user_input: str) -> bool:
        """사용자 응답 기반 조건인지 확인"""
        if not user_input:
            return False
        
        user_input_lower = user_input.lower().strip()
        
        # 예/아니오 응답 확인
        if '사용자' in condition:
            # 추천 방법 선택
            if '추천 방법 선택' in condition:
                try:
                    selection = int(user_input_lower)
                    return 1 <= selection <= 3  # 1-3 범위의 선택
                except ValueError:
                    return False
            
            # 추가 분석 요청
            if '추가 분석 요청' in condition:
                return any(yes_word in user_input_lower 
                          for yes_word in ['예', 'yes', 'y', '네', '계속', '추가'])
            
            # 다른 데이터로 분석 요청
            if '다른 데이터로 분석 요청' in condition:
                return any(other_word in user_input_lower 
                          for other_word in ['다른', 'other', '새로운', 'new', '다시'])
            
            # 세션 종료 요청
            if '세션 종료 요청' in condition:
                return any(end_word in user_input_lower 
                          for end_word in ['아니오', 'no', 'n', '종료', 'exit', '끝'])
            
            # 추가 설명 요청
            if '추가 설명 요청' in condition:
                return any(explain_word in user_input_lower 
                          for explain_word in ['설명', 'explain', '자세히', '왜', 'why'])
        
        return False
    
    def _is_rag_condition(self, condition: str, outcome: Any) -> bool:
        """RAG 시스템 관련 조건인지 확인"""
        rag_keywords = [
            'rag 검색', '비즈니스 지식', '스키마 구조', 'ai 추천',
            'business knowledge', 'schema structure', 'ai recommendation'
        ]
        
        if any(keyword in condition for keyword in rag_keywords):
            # RAG 검색 완료 확인
            if '완료' in condition or 'complete' in condition:
                return self._check_rag_completion(outcome)
            
            # RAG 검색 실패 확인
            if '실패' in condition or 'fail' in condition:
                return self._check_rag_failure(outcome)
        
        return False
    
    def _is_ai_recommendation_condition(self, condition: str, outcome: Any) -> bool:
        """AI 추천 관련 조건인지 확인"""
        if 'ai 추천' in condition or 'recommendation' in condition:
            if '생성 완료' in condition:
                return self._check_ai_recommendations_generated(outcome)
            elif '생성 실패' in condition:
                return not self._check_ai_recommendations_generated(outcome)
        
        return False
    
    def _is_data_state_condition(self, condition: str, outcome: Any) -> bool:
        """데이터 상태 기반 조건인지 확인"""
        # 데이터 로딩 관련
        if '로딩' in condition:
            if '완료' in condition or 'success' in condition:
                return self._check_data_loading_success(outcome)
            elif '실패' in condition or 'fail' in condition:
                return self._check_data_loading_failure(outcome)
        
        # 데이터 품질 관련
        if '데이터 품질' in condition or 'data quality' in condition:
            if '문제' in condition or 'issue' in condition:
                return self._check_data_quality_issues(outcome)
        
        # 전제조건 관련
        if '전제조건' in condition or 'assumption' in condition:
            if '충족' in condition or 'met' in condition:
                return self._check_assumptions_met(outcome)
            elif '위배' in condition or 'violation' in condition:
                return self._check_assumption_violations(outcome)
        
        return False
    
    def _is_analysis_result_condition(self, condition: str, outcome: Any) -> bool:
        """분석 결과 기반 조건인지 확인"""
        # 분석 실행 관련
        if '분석 실행' in condition or 'analysis execution' in condition:
            if '성공' in condition or 'success' in condition:
                return self._check_analysis_success(outcome)
            elif '오류' in condition or 'error' in condition:
                return self._check_analysis_error(outcome)
        
        # 분석 불가능 관련
        if '분석 불가능' in condition or 'analysis impossible' in condition:
            return self._check_analysis_impossible(outcome)
        
        return False
    
    def _is_complex_condition(self, condition: str, outcome: Any, user_input: str) -> bool:
        """복합 조건 (AND, OR) 평가"""
        # AND 조건
        if ' and ' in condition.lower() or ' 그리고 ' in condition:
            parts = re.split(r'\s+(?:and|그리고)\s+', condition, flags=re.IGNORECASE)
            return all(self._evaluate_simple_condition(part.strip(), outcome, user_input) 
                      for part in parts)
        
        # OR 조건
        if ' or ' in condition.lower() or ' 또는 ' in condition:
            parts = re.split(r'\s+(?:or|또는)\s+', condition, flags=re.IGNORECASE)
            return any(self._evaluate_simple_condition(part.strip(), outcome, user_input) 
                      for part in parts)
        
        return False
    
    def _evaluate_simple_condition(self, condition: str, outcome: Any, user_input: str) -> bool:
        """단순 조건을 평가합니다."""
        return self._evaluate_condition(condition, outcome, user_input)
    
    # RAG 시스템 관련 체크 메서드
    def _check_rag_completion(self, outcome: Any) -> bool:
        """RAG 검색 완료 여부를 확인합니다."""
        if isinstance(outcome, dict):
            return (outcome.get('business_context') is not None and 
                   outcome.get('schema_context') is not None)
        return False
    
    def _check_rag_failure(self, outcome: Any) -> bool:
        """RAG 검색 실패 여부를 확인합니다."""
        if isinstance(outcome, dict):
            return 'rag_error' in outcome or 'search_error' in outcome
        return False
    
    def _check_ai_recommendations_generated(self, outcome: Any) -> bool:
        """AI 추천 생성 여부를 확인합니다."""
        if isinstance(outcome, dict):
            recommendations = outcome.get('ai_recommendations', [])
            return len(recommendations) > 0
        elif isinstance(outcome, list):
            return len(outcome) > 0
        return False
    
    # 데이터 상태 관련 체크 메서드
    def _check_data_loading_success(self, outcome: Any) -> bool:
        """데이터 로딩 성공 여부를 확인합니다."""
        if isinstance(outcome, dict):
            return outcome.get('data_loaded', False)
        return False
    
    def _check_data_loading_failure(self, outcome: Any) -> bool:
        """데이터 로딩 실패 여부를 확인합니다."""
        if isinstance(outcome, dict):
            return 'data_error' in outcome or 'loading_error' in outcome
        return False
    
    def _check_data_quality_issues(self, outcome: Any) -> bool:
        """데이터 품질 문제 여부를 확인합니다."""
        if isinstance(outcome, dict):
            return outcome.get('quality_issues', []) or outcome.get('data_problems', False)
        return False
    
    def _check_assumptions_met(self, outcome: Any) -> bool:
        """통계적 가정 충족 여부를 확인합니다."""
        if isinstance(outcome, dict):
            assumptions = outcome.get('assumptions', {})
            return all(assumptions.values()) if assumptions else True
        return True
    
    def _check_assumption_violations(self, outcome: Any) -> bool:
        """통계적 가정 위배 여부를 확인합니다."""
        return not self._check_assumptions_met(outcome)
    
    # 분석 결과 관련 체크 메서드
    def _check_analysis_success(self, outcome: Any) -> bool:
        """분석 실행 성공 여부를 확인합니다."""
        if isinstance(outcome, dict):
            return outcome.get('analysis_success', False)
        return False
    
    def _check_analysis_error(self, outcome: Any) -> bool:
        """분석 실행 오류 여부를 확인합니다."""
        if isinstance(outcome, dict):
            return 'analysis_error' in outcome or 'execution_error' in outcome
        return False
    
    def _check_analysis_impossible(self, outcome: Any) -> bool:
        """분석 불가능 여부를 확인합니다."""
        if isinstance(outcome, dict):
            return outcome.get('analysis_impossible', False)
        return False
    
    def get_possible_next_nodes(self, current_node_details: Dict) -> List[str]:
        """
        현재 노드에서 가능한 모든 다음 노드들을 반환합니다.
        
        Args:
            current_node_details: 현재 노드의 전체 정보
            
        Returns:
            List[str]: 가능한 다음 노드 ID 목록
        """
        transitions = current_node_details.get('transitions', [])
        return [transition.get('next', '') for transition in transitions 
                if transition.get('next')]
    
    def explain_transition_logic(self, current_node_details: Dict, 
                               chosen_next_node: str) -> str:
        """
        선택된 전환 로직에 대한 설명을 제공합니다.
        
        Args:
            current_node_details: 현재 노드의 전체 정보
            chosen_next_node: 선택된 다음 노드 ID
            
        Returns:
            str: 전환 로직 설명
        """
        transitions = current_node_details.get('transitions', [])
        
        for transition in transitions:
            if transition.get('next') == chosen_next_node:
                condition = transition.get('condition', '')
                return f"조건 '{condition}'이 만족되어 노드 '{chosen_next_node}'로 전환합니다."
        
        return f"노드 '{chosen_next_node}'로의 전환 조건을 찾을 수 없습니다." 