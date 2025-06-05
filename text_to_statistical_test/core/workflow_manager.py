"""
WorkflowManager: Enhanced RAG 기반 워크플로우 그래프 관리

workflow_graph.json 파일을 로드하고 Enhanced RAG 시스템을 포함한 
Multi-turn 대화형 워크플로우 노드 관리 기능을 제공합니다.
"""

import json
import logging
from typing import Dict, Optional, List
from pathlib import Path


class WorkflowManager:
    """
    Enhanced RAG 기반 Multi-turn 대화형 워크플로우 정의서 관리자
    
    JSON 형식의 워크플로우 그래프를 로드하고, Enhanced RAG 시스템을 포함한
    Multi-turn 세션의 각 노드 정보와 전환 규칙을 제공하는 인터페이스를 제공합니다.
    """
    
    def __init__(self, workflow_file_path: str):
        """
        WorkflowManager 초기화
        
        Args:
            workflow_file_path: 워크플로우 JSON 파일 경로
        """
        self.workflow_file_path = Path(workflow_file_path)
        self._workflow_definition: Dict = {}
        self.logger = logging.getLogger(__name__)
        
        # 워크플로우 파일 로드
        self._load_workflow()
    
    def _load_workflow(self):
        """워크플로우 JSON 파일을 로드합니다."""
        try:
            if not self.workflow_file_path.exists():
                raise FileNotFoundError(f"워크플로우 파일을 찾을 수 없습니다: {self.workflow_file_path}")
            
            with open(self.workflow_file_path, 'r', encoding='utf-8') as f:
                self._workflow_definition = json.load(f)
            
            self.logger.info(f"워크플로우 파일 로드 완료: {self.workflow_file_path}")
            
            # 기본 구조 검증
            if 'nodes' not in self._workflow_definition:
                raise ValueError("워크플로우 파일에 'nodes' 섹션이 없습니다.")
                
        except Exception as e:
            self.logger.error(f"워크플로우 파일 로드 실패: {e}")
            raise
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        """
        특정 ID의 노드 정보를 반환합니다.
        
        Args:
            node_id: 노드 ID
            
        Returns:
            Dict: 노드 정보 (description, purpose, subtasks, transitions 등)
            None: 해당 노드가 존재하지 않는 경우
        """
        nodes = self._workflow_definition.get('nodes', {})
        return nodes.get(node_id)
    
    def get_initial_node_id(self) -> str:
        """
        워크플로우 시작 노드 ID를 반환합니다.
        
        Returns:
            str: 시작 노드 ID (Enhanced RAG 워크플로우에서는 "start")
        """
        # start 노드가 있는지 확인
        if 'start' in self._workflow_definition.get('nodes', {}):
            return 'start'
        
        # start 노드가 없으면 첫 번째 노드 반환
        nodes = self._workflow_definition.get('nodes', {})
        if nodes:
            return list(nodes.keys())[0]
        
        raise ValueError("워크플로우에 노드가 정의되지 않았습니다.")
    
    def is_terminal_node(self, node_id: str) -> bool:
        """
        해당 노드가 종료 노드인지 확인합니다.
        
        Args:
            node_id: 확인할 노드 ID
            
        Returns:
            bool: 종료 노드인 경우 True, 아니면 False
        """
        node = self.get_node(node_id)
        if node is None:
            return True  # 존재하지 않는 노드는 종료로 간주
        
        # transitions가 없거나 비어있으면 종료 노드
        transitions = node.get('transitions', [])
        if not transitions:
            return True
        
        # Enhanced RAG 워크플로우의 종료 노드들
        terminal_nodes = ['session_end', 'error_handling', 'analysis_impossible']
        if node_id in terminal_nodes:
            return True
            
        return False
    
    def is_loop_node(self, node_id: str) -> bool:
        """
        해당 노드가 Multi-turn 루프 노드인지 확인합니다.
        
        Args:
            node_id: 확인할 노드 ID
            
        Returns:
            bool: 루프 노드인 경우 True, 아니면 False
        """
        loop_nodes = [
            'session_continuation', 'natural_language_request', 
            'recommendation_display', 'multi_turn_interaction'
        ]
        return node_id in loop_nodes
    
    def is_interactive_node(self, node_id: str) -> bool:
        """
        해당 노드가 사용자 상호작용이 필요한 노드인지 확인합니다.
        
        Args:
            node_id: 확인할 노드 ID
            
        Returns:
            bool: 상호작용 노드인 경우 True, 아니면 False
        """
        interactive_nodes = [
            'data_selection', 'natural_language_request', 'request_clarification',
            'recommendation_display', 'recommendation_explanation', 'method_confirmation',
            'session_continuation', 'data_quality_issues', 'assumption_violation_handling'
        ]
        return node_id in interactive_nodes
    
    def is_rag_node(self, node_id: str) -> bool:
        """
        해당 노드가 Enhanced RAG 시스템 관련 노드인지 확인합니다.
        
        Args:
            node_id: 확인할 노드 ID
            
        Returns:
            bool: RAG 노드인 경우 True, 아니면 False
        """
        rag_nodes = [
            'rag_system_activation', 'business_knowledge_search', 
            'schema_structure_search', 'ai_recommendation_generation'
        ]
        return node_id in rag_nodes
    
    def is_automated_node(self, node_id: str) -> bool:
        """
        해당 노드가 자동 처리 노드인지 확인합니다.
        
        Args:
            node_id: 확인할 노드 ID
            
        Returns:
            bool: 자동 처리 노드인 경우 True, 아니면 False
        """
        automated_nodes = [
            'data_overview', 'rag_system_activation', 'business_knowledge_search',
            'schema_structure_search', 'ai_recommendation_generation',
            'automated_preprocessing', 'automated_assumption_testing',
            'statistical_analysis_execution', 'results_interpretation',
            'report_generation'
        ]
        return node_id in automated_nodes
    
    def get_node_transitions(self, node_id: str) -> List[Dict]:
        """
        특정 노드의 전환 규칙을 반환합니다.
        
        Args:
            node_id: 노드 ID
            
        Returns:
            List[Dict]: 전환 규칙 목록
        """
        node = self.get_node(node_id)
        if node is None:
            return []
        
        return node.get('transitions', [])
    
    def get_node_description(self, node_id: str) -> str:
        """
        특정 노드의 설명을 반환합니다.
        
        Args:
            node_id: 노드 ID
            
        Returns:
            str: 노드 설명
        """
        node = self.get_node(node_id)
        if node is None:
            return f"알 수 없는 노드: {node_id}"
        
        return node.get('description', '')
    
    def get_node_purpose(self, node_id: str) -> str:
        """
        특정 노드의 목적을 반환합니다.
        
        Args:
            node_id: 노드 ID
            
        Returns:
            str: 노드 목적
        """
        node = self.get_node(node_id)
        if node is None:
            return f"알 수 없는 노드: {node_id}"
        
        return node.get('purpose', '')
    
    def get_node_subtasks(self, node_id: str) -> List[Dict]:
        """
        특정 노드의 하위 작업 목록을 반환합니다.
        
        Args:
            node_id: 노드 ID
            
        Returns:
            List[Dict]: 하위 작업 목록
        """
        node = self.get_node(node_id)
        if node is None:
            return []
        
        return node.get('subtasks', [])
    
    def get_workflow_metadata(self) -> Dict:
        """
        워크플로우 메타데이터를 반환합니다.
        
        Returns:
            Dict: 워크플로우 메타데이터 (버전, 설명, 주요 기능 등)
        """
        return self._workflow_definition.get('workflow_metadata', {})
    
    def get_all_node_ids(self) -> List[str]:
        """
        모든 노드 ID 목록을 반환합니다.
        
        Returns:
            List[str]: 노드 ID 목록
        """
        return list(self._workflow_definition.get('nodes', {}).keys())
    
    def get_rag_workflow_sequence(self) -> List[str]:
        """
        Enhanced RAG 시스템 워크플로우 시퀀스를 반환합니다.
        
        Returns:
            List[str]: RAG 워크플로우 노드 시퀀스
        """
        return [
            'natural_language_request',
            'rag_system_activation', 
            'business_knowledge_search',
            'schema_structure_search',
            'ai_recommendation_generation',
            'recommendation_display'
        ]
    
    def get_automation_level(self) -> str:
        """
        워크플로우의 자동화 수준을 반환합니다.
        
        Returns:
            str: 자동화 수준 ('high', 'medium', 'low')
        """
        metadata = self.get_workflow_metadata()
        return metadata.get('automation_level', 'medium')
    
    def get_user_intervention_points(self) -> List[str]:
        """
        사용자 개입이 필요한 지점들을 반환합니다.
        
        Returns:
            List[str]: 사용자 개입 필요 지점 목록
        """
        metadata = self.get_workflow_metadata()
        return metadata.get('user_intervention_points', [])
    
    def get_context_management_rules(self) -> List[Dict]:
        """
        Multi-turn 컨텍스트 관리 규칙을 반환합니다.
        
        Returns:
            List[Dict]: 컨텍스트 관리 규칙 목록
        """
        context_mgmt = self._workflow_definition.get('context_management', {})
        return context_mgmt.get('rules', [])
    
    def get_recommendation_criteria(self) -> List[Dict]:
        """
        AI 추천 엔진의 평가 기준을 반환합니다.
        
        Returns:
            List[Dict]: 추천 평가 기준 목록
        """
        recommendation_engine = self._workflow_definition.get('recommendation_engine', {})
        return recommendation_engine.get('criteria', [])
    
    def get_confidence_scoring_rules(self) -> Dict:
        """
        신뢰도 점수 기준을 반환합니다.
        
        Returns:
            Dict: 신뢰도 점수 기준
        """
        recommendation_engine = self._workflow_definition.get('recommendation_engine', {})
        return recommendation_engine.get('confidence_scoring', {})
    
    def validate_workflow(self) -> bool:
        """
        Multi-turn 워크플로우 정의의 유효성을 검사합니다.
        
        Returns:
            bool: 유효한 경우 True, 아니면 False
        """
        try:
            # 기본 구조 확인
            if 'nodes' not in self._workflow_definition:
                self.logger.error("워크플로우에 'nodes' 섹션이 없습니다.")
                return False
            
            nodes = self._workflow_definition['nodes']
            
            # 시작 노드 확인
            if 'start' not in nodes:
                self.logger.error("Multi-turn 워크플로우에 'start' 노드가 필수입니다.")
                return False
            
            # Multi-turn 필수 노드들 확인
            required_nodes = [
                'data_selection', 'ai_recommendation', 'multi_turn_loop', 
                'session_end'
            ]
            for required_node in required_nodes:
                if required_node not in nodes:
                    self.logger.error(f"Multi-turn 워크플로우에 필수 노드 '{required_node}'가 없습니다.")
                    return False
            
            # 각 노드의 전환 규칙 검증
            for node_id, node_data in nodes.items():
                transitions = node_data.get('transitions', [])
                
                for transition in transitions:
                    if 'condition' not in transition:
                        self.logger.error(f"노드 {node_id}의 전환에 'condition'이 없습니다.")
                        return False
                    
                    if 'next' not in transition:
                        self.logger.error(f"노드 {node_id}의 전환에 'next'가 없습니다.")
                        return False
                    
                    # 참조된 노드가 존재하는지 확인
                    next_node = transition['next']
                    if next_node not in nodes:
                        self.logger.warning(f"노드 {node_id}에서 참조하는 노드 {next_node}가 존재하지 않습니다.")
            
            # 메타데이터 확인
            metadata = self._workflow_definition.get('metadata', {})
            if metadata.get('workflow_type') != 'multi_turn_conversational':
                self.logger.warning("워크플로우 타입이 'multi_turn_conversational'이 아닙니다.")
            
            self.logger.info("Multi-turn 워크플로우 유효성 검사 통과")
            return True
            
        except Exception as e:
            self.logger.error(f"워크플로우 유효성 검사 실패: {e}")
            return False
    
    def get_workflow_summary(self) -> Dict:
        """
        Multi-turn 워크플로우 요약 정보를 반환합니다.
        
        Returns:
            Dict: 워크플로우 요약 정보
        """
        nodes = self._workflow_definition.get('nodes', {})
        metadata = self._workflow_definition.get('metadata', {})
        
        # 노드 유형별 분류
        terminal_nodes = []
        loop_nodes = []
        interactive_nodes = []
        other_nodes = []
        
        for node_id in nodes.keys():
            if self.is_terminal_node(node_id):
                terminal_nodes.append(node_id)
            elif self.is_loop_node(node_id):
                loop_nodes.append(node_id)
            elif self.is_interactive_node(node_id):
                interactive_nodes.append(node_id)
            else:
                other_nodes.append(node_id)
        
        return {
            'metadata': metadata,
            'total_nodes': len(nodes),
            'terminal_nodes': terminal_nodes,
            'loop_nodes': loop_nodes,
            'interactive_nodes': interactive_nodes,
            'other_nodes': other_nodes,
            'start_node': self.get_initial_node_id(),
            'context_rules_count': len(self.get_context_management_rules()),
            'recommendation_criteria_count': len(self.get_recommendation_criteria())
        } 