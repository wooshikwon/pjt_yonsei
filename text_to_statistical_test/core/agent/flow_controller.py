"""
Flow Controller

Agentic Flow 제어 (상태, 전환, 도구 사용 관리)
- Agent 상태 관리 및 전환 제어
- 도구 사용 순서 및 조건 관리
- 실행 흐름 최적화
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timedelta
import json

from .autonomous_agent import AgentState, ActionType, AgentAction
from .decision_tree import DecisionTree, DecisionNode
from .tool_registry import ToolRegistry
from utils.error_handler import handle_error, StatisticalException


class FlowState(Enum):
    """Flow 상태"""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    ADAPTING = "adapting"
    COMPLETING = "completing"
    ERROR_HANDLING = "error_handling"
    PAUSED = "paused"


class TransitionCondition(Enum):
    """전환 조건"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    USER_INPUT = "user_input"
    THRESHOLD_MET = "threshold_met"
    MANUAL_TRIGGER = "manual_trigger"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class FlowTransition:
    """Flow 전환"""
    from_state: FlowState
    to_state: FlowState
    condition: TransitionCondition
    condition_params: Dict[str, Any] = field(default_factory=dict)
    action_required: Optional[ActionType] = None
    priority: int = 1
    timeout_seconds: Optional[int] = None


@dataclass
class FlowContext:
    """Flow 컨텍스트"""
    current_state: FlowState
    previous_states: List[FlowState] = field(default_factory=list)
    state_data: Dict[str, Any] = field(default_factory=dict)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    error_count: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_transition_time: datetime = field(default_factory=datetime.now)


@dataclass
class FlowMetrics:
    """Flow 메트릭"""
    total_transitions: int = 0
    successful_transitions: int = 0
    failed_transitions: int = 0
    average_state_duration: float = 0.0
    error_rate: float = 0.0
    efficiency_score: float = 0.0


class FlowController:
    """Agentic Flow 제어 (상태, 전환, 도구 사용 관리)"""
    
    def __init__(self, decision_tree: DecisionTree = None, tool_registry: ToolRegistry = None):
        """
        FlowController 초기화
        
        Args:
            decision_tree: 의사결정 트리
            tool_registry: 도구 레지스트리
        """
        self.logger = logging.getLogger(__name__)
        
        # 핵심 컴포넌트
        self.decision_tree = decision_tree or DecisionTree()
        self.tool_registry = tool_registry or ToolRegistry()
        
        # Flow 상태 관리
        self.context = FlowContext(current_state=FlowState.INITIALIZING)
        self.metrics = FlowMetrics()
        
        # 전환 규칙 정의
        self.transitions = self._define_default_transitions()
        
        # 상태별 핸들러
        self.state_handlers = self._setup_state_handlers()
        
        # 설정
        self.max_error_count = 5
        self.default_timeout = 300  # 5분
        self.monitoring_interval = 1.0  # 1초
        
        # 콜백 함수들
        self.state_change_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        self.logger.info("FlowController 초기화 완료")
    
    async def start_flow(self, initial_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Flow 시작
        
        Args:
            initial_context: 초기 컨텍스트
            
        Returns:
            Dict: Flow 실행 결과
        """
        self.logger.info("Agentic Flow 시작")
        
        try:
            # 초기 컨텍스트 설정
            if initial_context:
                self.context.state_data.update(initial_context)
            
            # 초기화 상태로 전환
            await self._transition_to_state(FlowState.INITIALIZING)
            
            # Flow 실행 루프
            result = await self._execute_flow_loop()
            
            self.logger.info("Agentic Flow 완료")
            return result
            
        except Exception as e:
            self.logger.error(f"Flow 실행 중 오류: {e}")
            await self._handle_error(e)
            return {
                'success': False,
                'error': str(e),
                'final_state': self.context.current_state.value,
                'metrics': self._get_metrics_summary()
            }
    
    async def _execute_flow_loop(self) -> Dict[str, Any]:
        """Flow 실행 루프"""
        max_iterations = 100
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            self.logger.debug(f"Flow 반복 {iteration}, 현재 상태: {self.context.current_state.value}")
            
            try:
                # 현재 상태 처리
                state_result = await self._handle_current_state()
                
                # 상태 결과 기록
                self.context.execution_history.append({
                    'iteration': iteration,
                    'state': self.context.current_state.value,
                    'result': state_result,
                    'timestamp': datetime.now()
                })
                
                # 완료 조건 확인
                if self.context.current_state == FlowState.COMPLETING:
                    break
                
                # 다음 전환 결정
                next_transition = await self._decide_next_transition(state_result)
                
                if next_transition:
                    await self._execute_transition(next_transition)
                else:
                    self.logger.warning("다음 전환을 결정할 수 없습니다.")
                    break
                
                # 모니터링 간격 대기
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Flow 루프 오류: {e}")
                await self._handle_error(e)
                
                if self.context.error_count >= self.max_error_count:
                    self.logger.error("최대 오류 횟수 초과")
                    break
        
        return await self._finalize_flow()
    
    async def _handle_current_state(self) -> Dict[str, Any]:
        """현재 상태 처리"""
        handler = self.state_handlers.get(self.context.current_state)
        
        if handler:
            return await handler()
        else:
            self.logger.warning(f"상태 {self.context.current_state.value}에 대한 핸들러가 없습니다.")
            return {'success': False, 'error': 'No handler for current state'}
    
    async def _decide_next_transition(self, state_result: Dict[str, Any]) -> Optional[FlowTransition]:
        """다음 전환 결정"""
        current_state = self.context.current_state
        
        # 현재 상태에서 가능한 전환들 찾기
        possible_transitions = [
            t for t in self.transitions 
            if t.from_state == current_state
        ]
        
        if not possible_transitions:
            return None
        
        # 조건에 맞는 전환 찾기
        for transition in sorted(possible_transitions, key=lambda t: t.priority, reverse=True):
            if await self._check_transition_condition(transition, state_result):
                return transition
        
        return None
    
    async def _check_transition_condition(self, transition: FlowTransition, 
                                        state_result: Dict[str, Any]) -> bool:
        """전환 조건 확인"""
        condition = transition.condition
        
        if condition == TransitionCondition.SUCCESS:
            return state_result.get('success', False)
        
        elif condition == TransitionCondition.FAILURE:
            return not state_result.get('success', True)
        
        elif condition == TransitionCondition.TIMEOUT:
            timeout = transition.condition_params.get('timeout', self.default_timeout)
            elapsed = (datetime.now() - self.context.last_transition_time).total_seconds()
            return elapsed >= timeout
        
        elif condition == TransitionCondition.THRESHOLD_MET:
            threshold_key = transition.condition_params.get('key')
            threshold_value = transition.condition_params.get('value')
            current_value = state_result.get(threshold_key)
            return current_value is not None and current_value >= threshold_value
        
        elif condition == TransitionCondition.ERROR_OCCURRED:
            return state_result.get('error') is not None
        
        elif condition == TransitionCondition.USER_INPUT:
            return self.context.state_data.get('user_input_received', False)
        
        elif condition == TransitionCondition.MANUAL_TRIGGER:
            return self.context.state_data.get('manual_trigger', False)
        
        return False
    
    async def _execute_transition(self, transition: FlowTransition):
        """전환 실행"""
        self.logger.info(f"상태 전환: {transition.from_state.value} -> {transition.to_state.value}")
        
        try:
            # 전환 전 작업
            await self._pre_transition_actions(transition)
            
            # 상태 전환
            await self._transition_to_state(transition.to_state)
            
            # 전환 후 작업
            await self._post_transition_actions(transition)
            
            # 메트릭 업데이트
            self.metrics.total_transitions += 1
            self.metrics.successful_transitions += 1
            
        except Exception as e:
            self.logger.error(f"전환 실행 오류: {e}")
            self.metrics.failed_transitions += 1
            raise
    
    async def _transition_to_state(self, new_state: FlowState):
        """상태 전환"""
        old_state = self.context.current_state
        
        # 상태 변경
        self.context.previous_states.append(old_state)
        self.context.current_state = new_state
        self.context.last_transition_time = datetime.now()
        
        # 콜백 실행
        for callback in self.state_change_callbacks:
            try:
                await callback(old_state, new_state, self.context)
            except Exception as e:
                self.logger.error(f"상태 변경 콜백 오류: {e}")
        
        self.logger.info(f"상태 전환 완료: {old_state.value} -> {new_state.value}")
    
    async def _pre_transition_actions(self, transition: FlowTransition):
        """전환 전 작업"""
        # 필요한 액션 실행
        if transition.action_required:
            action_result = await self._execute_required_action(transition.action_required)
            self.context.state_data['last_action_result'] = action_result
    
    async def _post_transition_actions(self, transition: FlowTransition):
        """전환 후 작업"""
        # 상태 데이터 정리
        self._cleanup_state_data(transition)
        
        # 타임아웃 설정
        if transition.timeout_seconds:
            self.context.state_data['timeout'] = datetime.now() + timedelta(seconds=transition.timeout_seconds)
    
    async def _execute_required_action(self, action_type: ActionType) -> Dict[str, Any]:
        """필수 액션 실행"""
        try:
            # 도구 레지스트리에서 적절한 도구 찾기
            tool = self.tool_registry.get_tool_for_action(action_type)
            
            if tool:
                return await tool.execute(self.context.state_data)
            else:
                return {'success': False, 'error': f'No tool found for action: {action_type.value}'}
                
        except Exception as e:
            self.logger.error(f"액션 실행 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    def _cleanup_state_data(self, transition: FlowTransition):
        """상태 데이터 정리"""
        # 이전 상태의 임시 데이터 제거
        keys_to_remove = [
            'manual_trigger', 'user_input_received', 'temporary_data'
        ]
        
        for key in keys_to_remove:
            self.context.state_data.pop(key, None)
    
    async def _handle_error(self, error: Exception):
        """오류 처리"""
        self.context.error_count += 1
        
        # 오류 정보 기록
        error_info = {
            'error': str(error),
            'error_type': type(error).__name__,
            'state': self.context.current_state.value,
            'timestamp': datetime.now(),
            'error_count': self.context.error_count
        }
        
        self.context.state_data['last_error'] = error_info
        
        # 오류 콜백 실행
        for callback in self.error_callbacks:
            try:
                await callback(error, self.context)
            except Exception as e:
                self.logger.error(f"오류 콜백 실행 중 오류: {e}")
        
        # 오류 처리 상태로 전환
        if self.context.current_state != FlowState.ERROR_HANDLING:
            await self._transition_to_state(FlowState.ERROR_HANDLING)
    
    async def _finalize_flow(self) -> Dict[str, Any]:
        """Flow 완료 처리"""
        end_time = datetime.now()
        total_duration = (end_time - self.context.start_time).total_seconds()
        
        # 메트릭 계산
        self._calculate_final_metrics(total_duration)
        
        # 최종 결과 구성
        result = {
            'success': self.context.current_state == FlowState.COMPLETING,
            'final_state': self.context.current_state.value,
            'total_duration': total_duration,
            'total_transitions': self.metrics.total_transitions,
            'error_count': self.context.error_count,
            'execution_history': self.context.execution_history,
            'metrics': self._get_metrics_summary(),
            'final_context': self.context.state_data
        }
        
        return result
    
    def _calculate_final_metrics(self, total_duration: float):
        """최종 메트릭 계산"""
        if self.metrics.total_transitions > 0:
            self.metrics.average_state_duration = total_duration / self.metrics.total_transitions
            self.metrics.error_rate = self.context.error_count / self.metrics.total_transitions
            
            success_rate = self.metrics.successful_transitions / self.metrics.total_transitions
            self.metrics.efficiency_score = success_rate * (1 - self.metrics.error_rate)
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """메트릭 요약 반환"""
        return {
            'total_transitions': self.metrics.total_transitions,
            'successful_transitions': self.metrics.successful_transitions,
            'failed_transitions': self.metrics.failed_transitions,
            'average_state_duration': self.metrics.average_state_duration,
            'error_rate': self.metrics.error_rate,
            'efficiency_score': self.metrics.efficiency_score
        }
    
    # 상태 핸들러들
    async def _handle_initializing_state(self) -> Dict[str, Any]:
        """초기화 상태 처리"""
        self.logger.info("Flow 초기화 중...")
        
        try:
            # 도구 레지스트리 초기화
            await self.tool_registry.initialize()
            
            # 의사결정 트리 준비
            await self.decision_tree.prepare()
            
            # 초기 컨텍스트 검증
            if not self._validate_initial_context():
                return {'success': False, 'error': 'Invalid initial context'}
            
            return {'success': True, 'message': 'Initialization completed'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _handle_planning_state(self) -> Dict[str, Any]:
        """계획 상태 처리"""
        self.logger.info("분석 계획 수립 중...")
        
        try:
            # 의사결정 트리를 통한 계획 수립
            planning_context = {
                'data': self.context.state_data.get('data'),
                'user_requirements': self.context.state_data.get('user_requirements'),
                'constraints': self.context.state_data.get('constraints', [])
            }
            
            plan = await self.decision_tree.create_analysis_plan(planning_context)
            self.context.state_data['analysis_plan'] = plan
            
            return {'success': True, 'plan': plan}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _handle_executing_state(self) -> Dict[str, Any]:
        """실행 상태 처리"""
        self.logger.info("분석 실행 중...")
        
        try:
            plan = self.context.state_data.get('analysis_plan')
            if not plan:
                return {'success': False, 'error': 'No analysis plan available'}
            
            # 계획에 따른 실행
            execution_results = []
            for step in plan.get('steps', []):
                step_result = await self._execute_analysis_step(step)
                execution_results.append(step_result)
                
                if not step_result.get('success'):
                    return {'success': False, 'error': f'Step failed: {step}', 'results': execution_results}
            
            self.context.state_data['execution_results'] = execution_results
            return {'success': True, 'results': execution_results}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _handle_monitoring_state(self) -> Dict[str, Any]:
        """모니터링 상태 처리"""
        self.logger.info("실행 모니터링 중...")
        
        try:
            # 실행 결과 모니터링
            results = self.context.state_data.get('execution_results', [])
            
            # 품질 메트릭 계산
            quality_score = self._calculate_quality_score(results)
            
            # 완료 조건 확인
            completion_check = self._check_completion_criteria(results)
            
            monitoring_result = {
                'quality_score': quality_score,
                'completion_status': completion_check,
                'results_count': len(results)
            }
            
            return {'success': True, 'monitoring': monitoring_result}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _handle_adapting_state(self) -> Dict[str, Any]:
        """적응 상태 처리"""
        self.logger.info("전략 적응 중...")
        
        try:
            # 현재 결과 분석
            current_results = self.context.state_data.get('execution_results', [])
            
            # 적응 필요성 평가
            adaptation_needed = self._assess_adaptation_need(current_results)
            
            if adaptation_needed:
                # 새로운 전략 생성
                new_strategy = await self._generate_adaptive_strategy(current_results)
                self.context.state_data['adaptive_strategy'] = new_strategy
                
                return {'success': True, 'adapted': True, 'new_strategy': new_strategy}
            else:
                return {'success': True, 'adapted': False, 'message': 'No adaptation needed'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _handle_completing_state(self) -> Dict[str, Any]:
        """완료 상태 처리"""
        self.logger.info("Flow 완료 처리 중...")
        
        try:
            # 최종 결과 정리
            final_results = self._compile_final_results()
            
            # 성공 여부 판단
            success = self._determine_overall_success(final_results)
            
            return {
                'success': success,
                'final_results': final_results,
                'message': 'Flow completed successfully' if success else 'Flow completed with issues'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _handle_error_handling_state(self) -> Dict[str, Any]:
        """오류 처리 상태 처리"""
        self.logger.info("오류 처리 중...")
        
        try:
            last_error = self.context.state_data.get('last_error')
            
            # 복구 시도
            recovery_result = await self._attempt_error_recovery(last_error)
            
            if recovery_result.get('success'):
                # 복구 성공 시 이전 상태로 복귀
                if len(self.context.previous_states) >= 2:
                    previous_state = self.context.previous_states[-2]
                    await self._transition_to_state(previous_state)
                
                return {'success': True, 'recovered': True}
            else:
                return {'success': False, 'recovered': False, 'error': 'Recovery failed'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _handle_paused_state(self) -> Dict[str, Any]:
        """일시정지 상태 처리"""
        self.logger.info("Flow 일시정지 중...")
        
        # 사용자 입력 대기
        user_input = self.context.state_data.get('user_input')
        if user_input:
            self.context.state_data['user_input_received'] = True
            return {'success': True, 'resumed': True}
        
        return {'success': True, 'waiting': True}
    
    # 헬퍼 메서드들
    def _define_default_transitions(self) -> List[FlowTransition]:
        """기본 전환 규칙 정의"""
        return [
            # 초기화 -> 계획
            FlowTransition(
                from_state=FlowState.INITIALIZING,
                to_state=FlowState.PLANNING,
                condition=TransitionCondition.SUCCESS,
                priority=1
            ),
            
            # 계획 -> 실행
            FlowTransition(
                from_state=FlowState.PLANNING,
                to_state=FlowState.EXECUTING,
                condition=TransitionCondition.SUCCESS,
                priority=1
            ),
            
            # 실행 -> 모니터링
            FlowTransition(
                from_state=FlowState.EXECUTING,
                to_state=FlowState.MONITORING,
                condition=TransitionCondition.SUCCESS,
                priority=1
            ),
            
            # 모니터링 -> 완료
            FlowTransition(
                from_state=FlowState.MONITORING,
                to_state=FlowState.COMPLETING,
                condition=TransitionCondition.THRESHOLD_MET,
                condition_params={'key': 'quality_score', 'value': 0.8},
                priority=1
            ),
            
            # 모니터링 -> 적응
            FlowTransition(
                from_state=FlowState.MONITORING,
                to_state=FlowState.ADAPTING,
                condition=TransitionCondition.THRESHOLD_MET,
                condition_params={'key': 'quality_score', 'value': 0.5},
                priority=2
            ),
            
            # 적응 -> 실행
            FlowTransition(
                from_state=FlowState.ADAPTING,
                to_state=FlowState.EXECUTING,
                condition=TransitionCondition.SUCCESS,
                priority=1
            ),
            
            # 오류 처리 전환들
            FlowTransition(
                from_state=FlowState.PLANNING,
                to_state=FlowState.ERROR_HANDLING,
                condition=TransitionCondition.FAILURE,
                priority=3
            ),
            
            FlowTransition(
                from_state=FlowState.EXECUTING,
                to_state=FlowState.ERROR_HANDLING,
                condition=TransitionCondition.FAILURE,
                priority=3
            ),
            
            # 타임아웃 전환들
            FlowTransition(
                from_state=FlowState.EXECUTING,
                to_state=FlowState.ADAPTING,
                condition=TransitionCondition.TIMEOUT,
                condition_params={'timeout': 300},
                priority=2
            )
        ]
    
    def _setup_state_handlers(self) -> Dict[FlowState, Callable]:
        """상태 핸들러 설정"""
        return {
            FlowState.INITIALIZING: self._handle_initializing_state,
            FlowState.PLANNING: self._handle_planning_state,
            FlowState.EXECUTING: self._handle_executing_state,
            FlowState.MONITORING: self._handle_monitoring_state,
            FlowState.ADAPTING: self._handle_adapting_state,
            FlowState.COMPLETING: self._handle_completing_state,
            FlowState.ERROR_HANDLING: self._handle_error_handling_state,
            FlowState.PAUSED: self._handle_paused_state
        }
    
    def _validate_initial_context(self) -> bool:
        """초기 컨텍스트 검증"""
        required_keys = ['data', 'user_requirements']
        return all(key in self.context.state_data for key in required_keys)
    
    async def _execute_analysis_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """분석 단계 실행"""
        step_type = step.get('type')
        
        # 도구 레지스트리에서 적절한 도구 찾기
        tool = self.tool_registry.get_tool_by_name(step_type)
        
        if tool:
            return await tool.execute(step.get('parameters', {}))
        else:
            return {'success': False, 'error': f'Unknown step type: {step_type}'}
    
    def _calculate_quality_score(self, results: List[Dict[str, Any]]) -> float:
        """품질 점수 계산"""
        if not results:
            return 0.0
        
        success_count = sum(1 for r in results if r.get('success', False))
        return success_count / len(results)
    
    def _check_completion_criteria(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """완료 기준 확인"""
        total_steps = len(results)
        completed_steps = sum(1 for r in results if r.get('success', False))
        
        return {
            'total_steps': total_steps,
            'completed_steps': completed_steps,
            'completion_rate': completed_steps / total_steps if total_steps > 0 else 0,
            'is_complete': completed_steps == total_steps
        }
    
    def _assess_adaptation_need(self, results: List[Dict[str, Any]]) -> bool:
        """적응 필요성 평가"""
        if not results:
            return False
        
        failure_rate = sum(1 for r in results if not r.get('success', True)) / len(results)
        return failure_rate > 0.3  # 30% 이상 실패 시 적응 필요
    
    async def _generate_adaptive_strategy(self, current_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """적응적 전략 생성"""
        # 실패 패턴 분석
        failed_steps = [r for r in current_results if not r.get('success', True)]
        
        # 새로운 전략 생성
        new_strategy = {
            'approach': 'adaptive',
            'failed_steps': len(failed_steps),
            'adjustments': ['retry_with_different_parameters', 'use_alternative_method']
        }
        
        return new_strategy
    
    def _compile_final_results(self) -> Dict[str, Any]:
        """최종 결과 정리"""
        return {
            'execution_results': self.context.state_data.get('execution_results', []),
            'analysis_plan': self.context.state_data.get('analysis_plan', {}),
            'adaptive_strategy': self.context.state_data.get('adaptive_strategy'),
            'error_count': self.context.error_count,
            'total_duration': (datetime.now() - self.context.start_time).total_seconds()
        }
    
    def _determine_overall_success(self, final_results: Dict[str, Any]) -> bool:
        """전체 성공 여부 판단"""
        execution_results = final_results.get('execution_results', [])
        if not execution_results:
            return False
        
        success_rate = sum(1 for r in execution_results if r.get('success', False)) / len(execution_results)
        return success_rate >= 0.8 and self.context.error_count < self.max_error_count
    
    async def _attempt_error_recovery(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """오류 복구 시도"""
        error_type = error_info.get('error_type')
        
        # 오류 유형별 복구 전략
        if error_type in ['TimeoutError', 'ConnectionError']:
            # 재시도
            return {'success': True, 'recovery_method': 'retry'}
        elif error_type in ['ValueError', 'TypeError']:
            # 파라미터 조정
            return {'success': True, 'recovery_method': 'parameter_adjustment'}
        else:
            # 기본 복구 실패
            return {'success': False, 'recovery_method': 'none'}
    
    # 공개 인터페이스 메서드들
    def add_state_change_callback(self, callback: Callable):
        """상태 변경 콜백 추가"""
        self.state_change_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """오류 콜백 추가"""
        self.error_callbacks.append(callback)
    
    def get_current_state(self) -> FlowState:
        """현재 상태 반환"""
        return self.context.current_state
    
    def get_flow_context(self) -> FlowContext:
        """Flow 컨텍스트 반환"""
        return self.context
    
    def get_flow_metrics(self) -> FlowMetrics:
        """Flow 메트릭 반환"""
        return self.metrics
    
    async def pause_flow(self):
        """Flow 일시정지"""
        await self._transition_to_state(FlowState.PAUSED)
    
    async def resume_flow(self, user_input: Dict[str, Any] = None):
        """Flow 재개"""
        if user_input:
            self.context.state_data.update(user_input)
        
        self.context.state_data['user_input_received'] = True
    
    async def force_transition(self, target_state: FlowState):
        """강제 상태 전환"""
        self.context.state_data['manual_trigger'] = True
        await self._transition_to_state(target_state) 