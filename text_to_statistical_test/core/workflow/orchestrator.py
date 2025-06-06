"""
Workflow Orchestrator

8단계 파이프라인의 순차적/조건부 실행을 관리하는 오케스트레이터
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from core.workflow.state_manager import StateManager
from core.workflow.conversation_history import ConversationHistory
from utils.error_handler import ErrorHandler, PipelineError
from utils.global_cache import get_global_cache

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    파이프라인 실행을 관리하는 오케스트레이터
    """
    
    def __init__(self, state_manager: Optional[StateManager] = None):
        """
        오케스트레이터 초기화
        
        Args:
            state_manager: 상태 관리자 (None이면 새로 생성)
        """
        self.state_manager = state_manager or StateManager()
        self.conversation_history = ConversationHistory()  # 대화 이력 관리자
        self.error_handler = ErrorHandler()
        self.cache = get_global_cache()
        
        # 파이프라인 단계들은 지연 로딩으로 처리 (순환 임포트 방지)
        self.pipeline_steps = {}
        self._step_classes = {
            1: ('core.pipeline.data_selection', 'DataSelectionStep'),
            2: ('core.pipeline.user_request', 'UserRequestStep'),
            3: ('core.pipeline.data_summary', 'DataSummaryStep'),
            4: ('core.pipeline.analysis_proposal', 'AnalysisProposalStep'),
            5: ('core.pipeline.user_selection', 'UserSelectionStep'),
            6: ('core.pipeline.agent_analysis', 'AgentAnalysisStep'),
            7: ('core.pipeline.agent_execution', 'AgentExecutionStep'),
            8: ('core.pipeline.agent_reporting', 'AgentReportingStep')
        }
        
        logger.info("오케스트레이터가 초기화되었습니다.")
    
    def _get_pipeline_step(self, step_num: int):
        """지연 로딩으로 파이프라인 단계 클래스 가져오기"""
        if step_num not in self.pipeline_steps:
            if step_num in self._step_classes:
                module_name, class_name = self._step_classes[step_num]
                try:
                    import importlib
                    module = importlib.import_module(module_name)
                    step_class = getattr(module, class_name)
                    
                    # 단계별로 필요한 의존성 주입
                    if step_num == 5:  # UserSelectionStep
                        self.pipeline_steps[step_num] = step_class(
                            conversation_history=self.conversation_history
                        )
                    else:
                        self.pipeline_steps[step_num] = step_class()
                    
                    logger.debug(f"단계 {step_num} 클래스 로드됨: {class_name}")
                except Exception as e:
                    logger.error(f"단계 {step_num} 클래스 로드 실패: {e}")
                    return None
            else:
                logger.warning(f"단계 {step_num}가 정의되지 않았습니다.")
                return None
        
        return self.pipeline_steps.get(step_num)
    
    async def execute_pipeline(self, start_stage: int = 1, 
                             initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        파이프라인 실행
        
        Args:
            start_stage: 시작 단계 (1-8)
            initial_context: 초기 컨텍스트
            
        Returns:
            Dict[str, Any]: 실행 결과
        """
        try:
            # 세션 생성
            execution_id = self.state_manager.create_session()
            logger.info(f"파이프라인 실행 시작 - ID: {execution_id}, 시작 단계: {start_stage}")
            
            # 초기 컨텍스트 설정
            if initial_context:
                self.state_manager.update_context(execution_id, initial_context)
            
            # 현재 데이터를 초기 컨텍스트로 설정
            current_data = initial_context or {}
            current_data['session_id'] = execution_id  # 세션 ID 추가
            
            # 건너뛸 단계 처리
            skip_stages = current_data.get('skip_stages', [])
            
            # 각 단계 순차 실행 (구현된 단계들만)
            max_stage = min(8, max(self.pipeline_steps.keys()))
            
            for stage_num in range(start_stage, max_stage + 1):
                if stage_num in skip_stages:
                    logger.info(f"단계 {stage_num} 건너뛰기")
                    continue
                
                # 구현되지 않은 단계는 건너뛰기
                if stage_num not in self.pipeline_steps:
                    logger.warning(f"단계 {stage_num}가 아직 구현되지 않았습니다. 건너뛰기")
                    continue
                
                try:
                    # 단계 실행
                    step_result = await self._execute_step(stage_num, current_data)
                    
                    if step_result.get('error'):
                        # 오류 처리
                        error_msg = step_result.get('error_message', '알 수 없는 오류')
                        logger.error(f"단계 {stage_num} 실행 실패: {error_msg}")
                        
                        # 복구 시도
                        recovery_result = await self._attempt_recovery(stage_num, step_result, current_data)
                        if recovery_result.get('success'):
                            step_result = recovery_result['result']
                        else:
                            return self._create_error_result(stage_num, error_msg, execution_id)
                    
                    # 상태 업데이트
                    self.state_manager.update_stage_result(execution_id, stage_num, step_result)
                    
                    # 다음 단계를 위한 데이터 준비
                    current_data = self._prepare_next_stage_data(current_data, step_result)
                    current_data['session_id'] = execution_id  # 세션 ID 유지
                    
                    # 조건부 실행 체크
                    if self._should_stop_pipeline(stage_num, step_result):
                        logger.info(f"단계 {stage_num}에서 파이프라인 중단")
                        break
                    
                    logger.info(f"단계 {stage_num} 완료")
                    
                except Exception as e:
                    error_msg = f"단계 {stage_num} 실행 중 예외: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    return self._create_error_result(stage_num, error_msg, execution_id)
            
            # 최종 결과 생성
            final_result = self._create_final_result(current_data, execution_id)
            
            # 세션 완료 처리
            self.state_manager.complete_session(execution_id)
            
            logger.info("파이프라인 실행 완료")
            return final_result
            
        except Exception as e:
            error_msg = f"파이프라인 실행 중 예외: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return self._create_error_result(0, error_msg, "unknown")
    
    async def _execute_step(self, stage_num: int, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        단일 단계 실행
        
        Args:
            stage_num: 단계 번호
            input_data: 입력 데이터
            
        Returns:
            Dict[str, Any]: 단계 실행 결과
        """
        step = self._get_pipeline_step(stage_num)
        if not step:
            raise PipelineError(f"단계 {stage_num}를 찾을 수 없습니다.")
        
        logger.info(f"단계 {stage_num} ({step.step_name}) 실행 시작")
        
        try:
            # 비동기 실행 지원
            if hasattr(step, 'execute_async'):
                result = await step.execute_async(input_data)
            else:
                # 동기 실행을 비동기로 래핑
                result = await asyncio.get_event_loop().run_in_executor(
                    None, step.run, input_data
                )
            
            return result
            
        except Exception as e:
            logger.error(f"단계 {stage_num} 실행 중 오류: {str(e)}", exc_info=True)
            return {
                'error': True,
                'error_message': str(e),
                'step_number': stage_num,
                'step_name': step.step_name,
                'exception_type': type(e).__name__
            }
    
    async def _attempt_recovery(self, stage_num: int, error_result: Dict[str, Any], 
                              input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        오류 복구 시도
        
        Args:
            stage_num: 오류가 발생한 단계
            error_result: 오류 결과
            input_data: 입력 데이터
            
        Returns:
            Dict[str, Any]: 복구 결과
        """
        logger.info(f"단계 {stage_num} 오류 복구 시도")
        
        try:
            # 오류 유형별 복구 전략
            error_type = error_result.get('exception_type', 'Unknown')
            
            if error_type in ['ValidationError', 'InputValidationError']:
                # 입력 검증 오류 - 기본값 사용 또는 재시도
                logger.info("입력 검증 오류 복구 시도")
                return await self._recover_validation_error(stage_num, input_data)
                
            elif error_type in ['LLMError', 'APIError']:
                # LLM API 오류 - 재시도 또는 폴백
                logger.info("LLM API 오류 복구 시도")
                return await self._recover_llm_error(stage_num, input_data)
                
            elif error_type in ['DataProcessingError']:
                # 데이터 처리 오류 - 대안 방법 시도
                logger.info("데이터 처리 오류 복구 시도")
                return await self._recover_data_error(stage_num, input_data)
            
            else:
                # 일반적인 복구 시도
                return await self._general_recovery(stage_num, input_data)
                
        except Exception as e:
            logger.error(f"복구 시도 중 오류: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _recover_validation_error(self, stage_num: int, 
                                      input_data: Dict[str, Any]) -> Dict[str, Any]:
        """검증 오류 복구"""
        # 기본값으로 재시도
        simplified_data = self._simplify_input_data(input_data)
        retry_result = await self._execute_step(stage_num, simplified_data)
        
        if not retry_result.get('error'):
            return {'success': True, 'result': retry_result}
        else:
            return {'success': False, 'error': '복구 실패'}
    
    async def _recover_llm_error(self, stage_num: int, 
                               input_data: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 오류 복구"""
        # 단순화된 프롬프트로 재시도
        input_data['simplified_mode'] = True
        retry_result = await self._execute_step(stage_num, input_data)
        
        if not retry_result.get('error'):
            return {'success': True, 'result': retry_result}
        else:
            return {'success': False, 'error': 'LLM 복구 실패'}
    
    async def _recover_data_error(self, stage_num: int, 
                                input_data: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 처리 오류 복구"""
        # 대안 데이터 처리 방법 시도
        input_data['fallback_mode'] = True
        retry_result = await self._execute_step(stage_num, input_data)
        
        if not retry_result.get('error'):
            return {'success': True, 'result': retry_result}
        else:
            return {'success': False, 'error': '데이터 처리 복구 실패'}
    
    async def _general_recovery(self, stage_num: int, 
                              input_data: Dict[str, Any]) -> Dict[str, Any]:
        """일반 복구"""
        # 최소한의 데이터로 재시도
        minimal_data = {
            'stage': stage_num,
            'recovery_mode': True,
            'session_id': input_data.get('session_id'),
            **{k: v for k, v in input_data.items() if k in ['data', 'file_path', 'columns']}
        }
        
        retry_result = await self._execute_step(stage_num, minimal_data)
        
        if not retry_result.get('error'):
            return {'success': True, 'result': retry_result}
        else:
            return {'success': False, 'error': '일반 복구 실패'}
    
    def _simplify_input_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """입력 데이터 단순화"""
        essential_keys = ['data', 'file_path', 'columns', 'user_request', 'analysis_type', 'session_id']
        return {k: v for k, v in data.items() if k in essential_keys}
    
    def _prepare_next_stage_data(self, current_data: Dict[str, Any], 
                               step_result: Dict[str, Any]) -> Dict[str, Any]:
        """다음 단계를 위한 데이터 준비"""
        next_data = current_data.copy()
        
        # 단계 결과를 다음 단계 입력에 추가
        if not step_result.get('error'):
            # 메타데이터 제외하고 실제 결과만 추가
            result_data = {k: v for k, v in step_result.items() if not k.startswith('_')}
            next_data.update(result_data)
        
        return next_data
    
    def _should_stop_pipeline(self, stage_num: int, step_result: Dict[str, Any]) -> bool:
        """파이프라인 중단 여부 결정"""
        # 사용자가 중단을 요청한 경우
        if step_result.get('user_stop_requested'):
            return True
        
        # 치명적 오류가 발생한 경우
        if step_result.get('critical_error'):
            return True
        
        # 특정 단계에서 중단 조건 체크
        if stage_num == 2 and not step_result.get('user_request'):
            # 사용자 요청이 없으면 중단
            return True
        
        return False
    
    def _create_final_result(self, data: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        """최종 결과 생성"""
        return {
            'success': True,
            'execution_id': execution_id,
            'execution_time': datetime.now().isoformat(),
            'pipeline_completed': True,
            'final_report': data.get('final_report'),
            'analysis_results': data.get('analysis_results'),
            'visualizations': data.get('visualizations'),
            'output_files': data.get('output_files', []),
            'session_summary': self.state_manager.get_session_summary(execution_id)
        }
    
    def _create_error_result(self, stage_num: int, error_msg: str, 
                           execution_id: str) -> Dict[str, Any]:
        """오류 결과 생성"""
        return {
            'success': False,
            'execution_id': execution_id,
            'error_stage': stage_num,
            'error_message': error_msg,
            'execution_time': datetime.now().isoformat(),
            'pipeline_completed': False
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """파이프라인 상태 조회"""
        return {
            'total_steps': len(self.pipeline_steps),
            'registered_steps': list(self.pipeline_steps.keys()),
            'state_manager_status': self.state_manager.get_status(),
            'cache_stats': self.cache.get_stats()
        } 