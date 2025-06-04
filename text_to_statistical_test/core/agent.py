"""
LLMAgent: 통계 분석 워크플로우 오케스트레이션 및 상태 관리

전체 통계 분석 프로세스의 중앙 컨트롤 타워 역할을 하며,
워크플로우의 각 단계를 실행하고 상태를 관리합니다.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime


class LLMAgent:
    """
    LLM Agent 기반 통계 검정 자동화 시스템의 핵심 클래스
    
    워크플로우 상태 기계의 실행자 역할을 하며, 각 노드별 작업을 처리하고
    전체 분석 과정을 관리합니다.
    """
    
    def __init__(self, workflow_manager, decision_engine, context_manager, 
                 llm_client, prompt_crafter, data_loader, code_retriever, 
                 safe_code_executor, report_generator):
        """
        LLMAgent 초기화
        
        Args:
            workflow_manager: 워크플로우 관리자
            decision_engine: 의사결정 엔진
            context_manager: 컨텍스트 관리자
            llm_client: LLM 클라이언트
            prompt_crafter: 프롬프트 생성기
            data_loader: 데이터 로더
            code_retriever: 코드 검색기
            safe_code_executor: 안전 코드 실행기
            report_generator: 보고서 생성기
        """
        self.workflow_manager = workflow_manager
        self.decision_engine = decision_engine
        self.context_manager = context_manager
        self.llm_client = llm_client
        self.prompt_crafter = prompt_crafter
        self.data_loader = data_loader
        self.code_retriever = code_retriever
        self.safe_code_executor = safe_code_executor
        self.report_generator = report_generator
        
        # 상태 관리
        self.current_node_id = "start"
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.analysis_parameters: Dict[str, Any] = {}
        self.user_interaction_history: list = []
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
    def run(self, input_data_path: Optional[str] = None) -> str:
        """
        전체 분석 프로세스 시작
        
        Args:
            input_data_path: 입력 데이터 파일 경로
            
        Returns:
            str: 생성된 보고서 파일 경로
        """
        self.logger.info("LLM Agent 워크플로우 시작")
        
        # 초기 설정
        if input_data_path:
            self._load_initial_data(input_data_path)
            
        # 시작 노드 설정
        self.current_node_id = self.workflow_manager.get_initial_node_id()
        
        # 메인 루프 실행
        self._main_loop()
        
        # 최종 보고서 생성
        report_path = self._generate_final_report()
        
        self.logger.info(f"워크플로우 완료. 보고서: {report_path}")
        return report_path
    
    def _load_initial_data(self, data_path: str):
        """초기 데이터 로딩"""
        try:
            self.raw_data = self.data_loader.load_data(data_path)
            self.logger.info(f"데이터 로딩 완료: {self.raw_data.shape}")
            
            # 컨텍스트에 추가
            self.context_manager.add_interaction(
                role="system",
                content=f"데이터 로딩 완료: {self.raw_data.shape[0]}행 {self.raw_data.shape[1]}열",
                node_id="data_loading"
            )
        except Exception as e:
            self.logger.error(f"데이터 로딩 실패: {e}")
            raise
    
    def _main_loop(self):
        """
        메인 워크플로우 루프
        
        현재 노드 처리 -> 다음 노드 결정 -> 상태 전이를 반복하며
        워크플로우 종료 조건 만족시까지 실행합니다.
        """
        max_iterations = 100  # 무한 루프 방지
        iteration_count = 0
        
        while (not self.workflow_manager.is_terminal_node(self.current_node_id) 
               and iteration_count < max_iterations):
            
            self.logger.info(f"현재 노드 처리: {self.current_node_id}")
            
            # 현재 노드 처리
            execution_result = self._process_current_node()
            
            # 다음 노드 결정
            next_node_id = self._determine_next_node(execution_result)
            
            if next_node_id is None:
                self.logger.warning("다음 노드를 결정할 수 없습니다. 워크플로우 종료.")
                break
                
            # 상태 전이 로깅
            self._log_state_transition(self.current_node_id, next_node_id, str(execution_result))
            
            # 다음 노드로 이동
            self.current_node_id = next_node_id
            iteration_count += 1
        
        if iteration_count >= max_iterations:
            self.logger.warning("최대 반복 횟수 도달. 워크플로우 강제 종료.")
    
    def _process_current_node(self) -> Any:
        """
        현재 노드의 작업을 수행
        
        Returns:
            Any: 노드 처리 결과
        """
        current_node = self.workflow_manager.get_node(self.current_node_id)
        
        if current_node is None:
            raise ValueError(f"존재하지 않는 노드: {self.current_node_id}")
        
        node_description = current_node.get('description', '')
        self.logger.info(f"노드 처리 중: {node_description}")
        
        # 노드 타입에 따른 처리 분기
        if self._is_llm_node(current_node):
            return self._handle_llm_interaction(current_node)
        elif self._is_user_input_node(current_node):
            return self._handle_user_confirmation(current_node)
        elif self._is_data_processing_node(current_node):
            return self._handle_data_processing(current_node)
        elif self._is_code_execution_node(current_node):
            return self._handle_code_execution(current_node)
        else:
            # 기본 처리
            return self._handle_default_node(current_node)
    
    def _is_llm_node(self, node: Dict) -> bool:
        """LLM 처리가 필요한 노드인지 판단"""
        description = node.get('description', '').lower()
        return any(keyword in description for keyword in 
                  ['llm', '분석', '판단', '확인', '해석', '추천'])
    
    def _is_user_input_node(self, node: Dict) -> bool:
        """사용자 입력이 필요한 노드인지 판단"""
        description = node.get('description', '').lower()
        return '사용자' in description and ('확인' in description or '입력' in description)
    
    def _is_data_processing_node(self, node: Dict) -> bool:
        """데이터 처리 노드인지 판단"""
        description = node.get('description', '').lower()
        return any(keyword in description for keyword in 
                  ['데이터', '로딩', '전처리', '변환', '정제'])
    
    def _is_code_execution_node(self, node: Dict) -> bool:
        """코드 실행 노드인지 판단"""
        description = node.get('description', '').lower()
        return any(keyword in description for keyword in 
                  ['검정 수행', '코드', '실행', '계산'])
    
    def _handle_llm_interaction(self, node_details: Dict) -> str:
        """LLM과의 상호작용 처리"""
        # 현재 컨텍스트 준비
        context_summary = self.context_manager.get_optimized_context(
            current_task_prompt=node_details.get('description', ''),
            required_recent_interactions=3
        )
        
        # 프롬프트 생성
        prompt = self.prompt_crafter.get_prompt_for_node(
            node_id=self.current_node_id,
            dynamic_data={
                'node_description': node_details.get('description', ''),
                'analysis_parameters': self.analysis_parameters,
                'data_info': self._get_data_summary() if self.raw_data is not None else None
            },
            agent_context_summary=context_summary
        )
        
        # LLM 호출
        response = self.llm_client.generate_text(prompt)
        
        # 응답을 컨텍스트에 추가
        self.context_manager.add_interaction(
            role="assistant",
            content=response,
            node_id=self.current_node_id
        )
        
        # 분석 파라미터 업데이트
        self._update_analysis_parameters_from_response(response)
        
        return response
    
    def _handle_user_confirmation(self, node_details: Dict) -> str:
        """사용자 확인 처리"""
        description = node_details.get('description', '')
        print(f"\n🤖 시스템: {description}")
        
        # 현재 분석 상태 출력
        if self.analysis_parameters:
            print("\n현재 분석 상태:")
            for key, value in self.analysis_parameters.items():
                print(f"  • {key}: {value}")
        
        user_input = input("\n👤 응답을 입력하세요 (예/아니오/수정): ").strip()
        
        # 사용자 입력을 컨텍스트에 추가
        self.context_manager.add_interaction(
            role="user", 
            content=user_input,
            node_id=self.current_node_id
        )
        
        return user_input
    
    def _handle_data_processing(self, node_details: Dict) -> Dict:
        """데이터 처리 작업"""
        if self.raw_data is None:
            return {"error": "데이터가 로드되지 않았습니다."}
        
        # 데이터 프로파일링
        data_profile = self.data_loader.get_data_profile(self.raw_data)
        
        # 분석 파라미터에 추가
        self.analysis_parameters.update({
            'data_profile': data_profile,
            'data_shape': self.raw_data.shape
        })
        
        return data_profile
    
    def _handle_code_execution(self, node_details: Dict) -> Dict:
        """통계 코드 실행 처리"""
        # 적합한 코드 스니펫 검색
        query_description = self._build_code_search_query()
        code_snippets = self.code_retriever.find_relevant_code_snippets(
            query_description=query_description,
            required_variables=self.analysis_parameters.get('variables', [])
        )
        
        if not code_snippets:
            return {"error": "적합한 코드 스니펫을 찾을 수 없습니다."}
        
        # 가장 관련성 높은 코드 실행
        best_code = code_snippets[0]['content']
        
        execution_result = self.safe_code_executor.execute_code(
            code_string=best_code,
            input_dataframe=self.processed_data or self.raw_data,
            parameters=self.analysis_parameters
        )
        
        return execution_result
    
    def _handle_default_node(self, node_details: Dict) -> str:
        """기본 노드 처리"""
        return "processed"
    
    def _determine_next_node(self, execution_result: Any) -> Optional[str]:
        """다음 노드 결정"""
        current_node = self.workflow_manager.get_node(self.current_node_id)
        
        next_node_id = self.decision_engine.determine_next_node(
            current_node_details=current_node,
            execution_outcome=execution_result,
            user_response=execution_result if isinstance(execution_result, str) else None
        )
        
        return next_node_id
    
    def _update_analysis_parameters_from_response(self, response: str):
        """LLM 응답에서 분석 파라미터 추출 및 업데이트"""
        # 간단한 키워드 기반 파라미터 추출 (추후 더 정교하게 구현)
        if '종속변수' in response or 'dependent' in response.lower():
            # 종속변수 추출 로직
            pass
        if '독립변수' in response or 'independent' in response.lower():
            # 독립변수 추출 로직
            pass
    
    def _get_data_summary(self) -> Dict:
        """현재 데이터 요약 정보 반환"""
        if self.raw_data is None:
            return {}
        
        return {
            'shape': self.raw_data.shape,
            'columns': list(self.raw_data.columns),
            'dtypes': self.raw_data.dtypes.to_dict(),
            'missing_values': self.raw_data.isnull().sum().to_dict()
        }
    
    def _build_code_search_query(self) -> str:
        """코드 검색을 위한 쿼리 구성"""
        query_parts = []
        
        if 'test_type' in self.analysis_parameters:
            query_parts.append(self.analysis_parameters['test_type'])
        
        if 'variables' in self.analysis_parameters:
            query_parts.append("변수 분석")
        
        return " ".join(query_parts) if query_parts else "통계 검정"
    
    def _log_state_transition(self, from_node: str, to_node: str, reason: str):
        """상태 전이 로깅"""
        self.logger.info(f"상태 전이: {from_node} -> {to_node} (이유: {reason})")
        
        # 사용자 상호작용 이력에 추가
        self.user_interaction_history.append({
            'timestamp': datetime.now().isoformat(),
            'from_node': from_node,
            'to_node': to_node,
            'reason': reason
        })
    
    def _generate_final_report(self) -> str:
        """최종 보고서 생성"""
        # 전체 이력 수집
        full_history = self.context_manager.get_full_history_for_report()
        
        # 최종 상태 정보
        agent_final_state = {
            'analysis_parameters': self.analysis_parameters,
            'data_summary': self._get_data_summary(),
            'final_node': self.current_node_id,
            'interaction_history': self.user_interaction_history
        }
        
        # 보고서 생성
        report_path = self.report_generator.generate_report(
            agent_final_state=agent_final_state,
            full_interaction_history=full_history,
            data_profile=self.analysis_parameters.get('data_profile', {}),
            workflow_graph_info={'final_node': self.current_node_id}
        )
        
        return report_path 