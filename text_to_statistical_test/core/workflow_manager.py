"""
WorkflowManager: 워크플로우 그래프 관리

workflow_graph.json 파일을 로드하고 워크플로우 노드 관리 기능을 제공합니다.
"""

import json
import logging
from typing import Dict, Optional, List
from pathlib import Path


class WorkflowManager:
    """
    워크플로우 정의서 관리자
    
    JSON 형식의 워크플로우 그래프를 로드하고, 각 노드의 정보와 
    전환 규칙을 제공하는 인터페이스를 제공합니다.
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
            Dict: 노드 정보 (description, subtasks, transitions 등)
            None: 해당 노드가 존재하지 않는 경우
        """
        nodes = self._workflow_definition.get('nodes', {})
        return nodes.get(node_id)
    
    def get_initial_node_id(self) -> str:
        """
        워크플로우 시작 노드 ID를 반환합니다.
        
        Returns:
            str: 시작 노드 ID (일반적으로 "start")
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
        
        # 특정 종료 노드들 확인
        terminal_nodes = ['7', '8', '9']  # workflow_graph.json 기준
        if node_id in terminal_nodes:
            return True
        
        # 7-3 (종료) 노드 확인
        if node_id == '7-3':
            return True
            
        return False
    
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
    
    def get_all_node_ids(self) -> List[str]:
        """
        모든 노드 ID 목록을 반환합니다.
        
        Returns:
            List[str]: 노드 ID 목록
        """
        return list(self._workflow_definition.get('nodes', {}).keys())
    
    def validate_workflow(self) -> bool:
        """
        워크플로우 정의의 유효성을 검사합니다.
        
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
                self.logger.warning("'start' 노드가 정의되지 않았습니다.")
            
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
            
            self.logger.info("워크플로우 유효성 검사 통과")
            return True
            
        except Exception as e:
            self.logger.error(f"워크플로우 유효성 검사 실패: {e}")
            return False
    
    def get_workflow_summary(self) -> Dict:
        """
        워크플로우 요약 정보를 반환합니다.
        
        Returns:
            Dict: 워크플로우 요약 정보
        """
        nodes = self._workflow_definition.get('nodes', {})
        
        # 노드 유형별 분류
        terminal_nodes = []
        intermediate_nodes = []
        
        for node_id in nodes.keys():
            if self.is_terminal_node(node_id):
                terminal_nodes.append(node_id)
            else:
                intermediate_nodes.append(node_id)
        
        return {
            'total_nodes': len(nodes),
            'terminal_nodes': terminal_nodes,
            'intermediate_nodes': intermediate_nodes,
            'start_node': self.get_initial_node_id()
        } 