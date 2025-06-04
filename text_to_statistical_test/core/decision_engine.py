"""
DecisionEngine: 워크플로우 네비게이터

현재 노드의 처리 결과와 전환 규칙을 비교하여 
다음으로 진행할 노드 ID를 결정합니다.
"""

import logging
import re
from typing import Dict, Any, Optional, List


class DecisionEngine:
    """
    워크플로우 네비게이터
    
    노드 처리 결과를 기반으로 워크플로우의 다음 단계를 결정하는 의사결정 엔진입니다.
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
        
        # 1. 자동 진행 조건
        if self._is_auto_condition(condition_lower):
            return True
        
        # 2. 사용자 응답 기반 조건
        if user_input and self._is_user_response_condition(condition_lower, user_input):
            return True
        
        # 3. 결과 기반 조건
        if self._is_outcome_based_condition(condition_lower, outcome):
            return True
        
        # 4. 데이터 상태 기반 조건
        if self._is_data_state_condition(condition_lower, outcome):
            return True
        
        # 5. 복합 조건 (AND, OR 포함)
        if self._is_complex_condition(condition_lower, outcome, user_input):
            return True
        
        return False
    
    def _is_auto_condition(self, condition: str) -> bool:
        """자동 진행 조건인지 확인"""
        auto_keywords = [
            '자동 시작', '진행', '완료', '자동', 'auto', 'proceed', 
            '다음', 'next', '계속', 'continue'
        ]
        return any(keyword in condition for keyword in auto_keywords)
    
    def _is_user_response_condition(self, condition: str, user_input: str) -> bool:
        """사용자 응답 기반 조건인지 확인"""
        if not user_input:
            return False
        
        user_input_lower = user_input.lower().strip()
        
        # 예/아니오 응답 확인
        if '사용자' in condition:
            if '예' in condition and any(yes_word in user_input_lower 
                                       for yes_word in ['예', 'yes', 'y', '네', '확인']):
                return True
            
            if '아니오' in condition and any(no_word in user_input_lower 
                                          for no_word in ['아니오', 'no', 'n', '아니', '거부']):
                return True
            
            if '수정' in condition and any(modify_word in user_input_lower 
                                        for modify_word in ['수정', 'modify', 'change', '변경']):
                return True
        
        # 직접 매칭
        if any(keyword in user_input_lower for keyword in condition.split()):
            return True
        
        return False
    
    def _is_outcome_based_condition(self, condition: str, outcome: Any) -> bool:
        """결과 기반 조건인지 확인"""
        if outcome is None:
            return False
        
        # 문자열 결과 분석
        if isinstance(outcome, str):
            outcome_lower = outcome.lower()
            
            # 에러/실패 조건
            if any(error_word in condition for error_word in ['오류', 'error', '실패', 'fail']):
                return any(error_word in outcome_lower for error_word in ['error', 'fail', '오류', '실패'])
            
            # 성공 조건
            if any(success_word in condition for success_word in ['성공', 'success', '완료', 'complete']):
                return any(success_word in outcome_lower for success_word in ['success', 'complete', '성공', '완료'])
        
        # 딕셔너리 결과 분석
        if isinstance(outcome, dict):
            if 'error' in outcome and any(error_word in condition for error_word in ['오류', 'error']):
                return True
            
            if 'success' in outcome and any(success_word in condition for success_word in ['성공', 'success']):
                return True
        
        return False
    
    def _is_data_state_condition(self, condition: str, outcome: Any) -> bool:
        """데이터 상태 기반 조건인지 확인"""
        # 독립성 관련 조건
        if '독립성' in condition:
            if '확보' in condition:
                # 독립성이 확보된 경우를 확인하는 로직
                return not self._has_independence_violation(outcome)
            elif '위배' in condition:
                # 독립성이 위배된 경우를 확인하는 로직
                return self._has_independence_violation(outcome)
        
        # 샘플 크기 조건
        if '샘플' in condition or 'sample' in condition.lower():
            if '충족' in condition:
                return self._check_sample_size_adequate(outcome)
            elif '미달' in condition:
                return not self._check_sample_size_adequate(outcome)
        
        # 전제조건 관련
        if '전제조건' in condition:
            if '충족' in condition:
                return self._check_assumptions_met(outcome)
            elif '위배' in condition:
                return not self._check_assumptions_met(outcome)
        
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
        """단순 조건 평가 (복합 조건의 구성 요소)"""
        return (self._is_auto_condition(condition) or
                self._is_user_response_condition(condition, user_input) or
                self._is_outcome_based_condition(condition, outcome) or
                self._is_data_state_condition(condition, outcome))
    
    def _has_independence_violation(self, outcome: Any) -> bool:
        """독립성 위배 여부 확인"""
        # 실제 구현에서는 outcome의 데이터를 분석하여 독립성 위배 여부를 판단
        # 현재는 간단한 구현
        if isinstance(outcome, dict):
            return outcome.get('independence_violation', False)
        return False
    
    def _check_sample_size_adequate(self, outcome: Any) -> bool:
        """샘플 크기 적절성 확인"""
        # 실제 구현에서는 outcome의 데이터를 분석하여 샘플 크기를 확인
        if isinstance(outcome, dict):
            sample_size = outcome.get('sample_size', 0)
            return sample_size >= 30  # 기본적인 중심극한정리 기준
        return True  # 기본값은 적절하다고 가정
    
    def _check_assumptions_met(self, outcome: Any) -> bool:
        """통계적 전제조건 만족 여부 확인"""
        # 실제 구현에서는 정규성, 등분산성 등의 전제조건을 확인
        if isinstance(outcome, dict):
            normality = outcome.get('normality_test', True)
            homoscedasticity = outcome.get('homoscedasticity_test', True)
            return normality and homoscedasticity
        return True  # 기본값은 만족한다고 가정
    
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