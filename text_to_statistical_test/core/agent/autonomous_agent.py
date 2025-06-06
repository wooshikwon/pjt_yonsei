"""
Autonomous Agent

자율적 의사결정 및 행동을 수행하는 LLM Agent
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from datetime import datetime

from services.llm.llm_client import LLMClient
from services.llm.prompt_engine import PromptEngine
from services.llm.llm_response_parser import LLMResponseParser
from core.rag.knowledge_store import KnowledgeStore
from core.rag.query_engine import QueryEngine
from core.rag.context_builder import ContextBuilder
from services.code_executor.safe_code_runner import SafeCodeRunner
from utils.error_handler import handle_error, StatisticalException


class AgentState(Enum):
    """Agent 상태"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    INTERPRETING = "interpreting"
    ERROR = "error"
    COMPLETED = "completed"


class ActionType(Enum):
    """Agent 행동 유형"""
    ANALYZE_DATA = "analyze_data"
    SEARCH_KNOWLEDGE = "search_knowledge"
    GENERATE_CODE = "generate_code"
    EXECUTE_CODE = "execute_code"
    VALIDATE_ASSUMPTIONS = "validate_assumptions"
    INTERPRET_RESULTS = "interpret_results"
    ASK_USER = "ask_user"
    ADJUST_STRATEGY = "adjust_strategy"


@dataclass
class AgentAction:
    """Agent 행동"""
    action_type: ActionType
    parameters: Dict[str, Any]
    reasoning: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    result: Optional[Dict[str, Any]] = None
    success: bool = False


@dataclass
class AgentMemory:
    """Agent 기억"""
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    analysis_history: List[Dict[str, Any]] = field(default_factory=list)
    learned_patterns: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    error_patterns: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentGoal:
    """Agent 목표"""
    primary_objective: str
    success_criteria: List[str]
    constraints: List[str]
    priority: int = 1
    deadline: Optional[datetime] = None
    progress: float = 0.0


class AutonomousAgent:
    """자율적 의사결정 및 행동 주체 LLM Agent"""
    
    def __init__(self, agent_id: str = None):
        """
        AutonomousAgent 초기화
        
        Args:
            agent_id: Agent 식별자
        """
        self.agent_id = agent_id or f"agent_{int(time.time())}"
        self.logger = logging.getLogger(__name__)
        
        # 핵심 컴포넌트 초기화
        self.llm_client = LLMClient()
        self.prompt_engine = PromptEngine()
        self.response_parser = LLMResponseParser()
        self.knowledge_store = KnowledgeStore()
        self.query_engine = QueryEngine()
        self.context_builder = ContextBuilder()
        self.code_runner = SafeCodeRunner()
        
        # Agent 상태 관리
        self.state = AgentState.IDLE
        self.current_goal: Optional[AgentGoal] = None
        self.action_history: List[AgentAction] = []
        self.memory = AgentMemory()
        
        # 설정
        self.max_iterations = 50
        self.confidence_threshold = 0.7
        self.learning_enabled = True
        
        self.logger.info(f"AutonomousAgent {self.agent_id} 초기화 완료")
    
    async def pursue_goal(self, goal: AgentGoal, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        목표 추구 및 자율적 실행
        
        Args:
            goal: 추구할 목표
            context: 실행 컨텍스트
            
        Returns:
            Dict: 실행 결과
        """
        self.logger.info(f"목표 추구 시작: {goal.primary_objective}")
        
        try:
            # 목표 설정
            self.current_goal = goal
            self.state = AgentState.ANALYZING
            
            # 초기 상황 분석
            situation_analysis = await self._analyze_situation(context)
            
            # 전략 수립
            strategy = await self._develop_strategy(situation_analysis, goal)
            
            # 자율적 실행 루프
            execution_result = await self._autonomous_execution_loop(strategy, context)
            
            # 결과 해석 및 학습
            final_result = await self._interpret_and_learn(execution_result, goal)
            
            self.state = AgentState.COMPLETED
            self.logger.info("목표 추구 완료")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"목표 추구 중 오류: {e}")
            self.state = AgentState.ERROR
            return {
                'success': False,
                'error': str(e),
                'agent_id': self.agent_id,
                'goal': goal.primary_objective
            }
    
    async def _analyze_situation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """상황 분석"""
        self.logger.info("상황 분석 시작")
        
        try:
            # 데이터 특성 분석
            data_analysis = await self._analyze_data_characteristics(context.get('data'))
            
            # 사용자 요구사항 분석
            user_requirements = await self._analyze_user_requirements(context.get('user_input'))
            
            # 제약사항 분석
            constraints = await self._analyze_constraints(context)
            
            # 가용 도구 및 리소스 분석
            resources = await self._analyze_available_resources()
            
            situation_analysis = {
                'data_characteristics': data_analysis,
                'user_requirements': user_requirements,
                'constraints': constraints,
                'available_resources': resources,
                'complexity_level': self._assess_complexity(data_analysis, user_requirements),
                'recommended_approach': self._recommend_initial_approach(data_analysis, user_requirements)
            }
            
            # 메모리에 저장
            self.memory.analysis_history.append({
                'timestamp': datetime.now(),
                'type': 'situation_analysis',
                'result': situation_analysis
            })
            
            return situation_analysis
            
        except Exception as e:
            self.logger.error(f"상황 분석 오류: {e}")
            raise
    
    async def _develop_strategy(self, situation_analysis: Dict[str, Any], 
                              goal: AgentGoal) -> Dict[str, Any]:
        """전략 수립"""
        self.logger.info("전략 수립 시작")
        
        try:
            # RAG를 통한 관련 지식 검색
            knowledge_query = self._build_knowledge_query(situation_analysis, goal)
            relevant_knowledge = await self.query_engine.search(knowledge_query)
            
            # 전략 생성 프롬프트 구성
            strategy_prompt = self.prompt_engine.create_strategy_prompt(
                goal=goal,
                situation=situation_analysis,
                knowledge=relevant_knowledge,
                agent_memory=self.memory
            )
            
            # LLM을 통한 전략 생성
            strategy_response = await self.llm_client.generate_response(strategy_prompt)
            strategy = self.response_parser.parse_strategy_response(strategy_response)
            
            # 전략 검증 및 개선
            validated_strategy = await self._validate_and_improve_strategy(strategy, situation_analysis)
            
            self.logger.info(f"전략 수립 완료: {len(validated_strategy.get('steps', []))}단계")
            return validated_strategy
            
        except Exception as e:
            self.logger.error(f"전략 수립 오류: {e}")
            raise
    
    async def _autonomous_execution_loop(self, strategy: Dict[str, Any], 
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """자율적 실행 루프"""
        self.logger.info("자율적 실행 루프 시작")
        
        execution_results = []
        current_context = context.copy()
        
        try:
            for iteration in range(self.max_iterations):
                self.logger.info(f"실행 반복 {iteration + 1}/{self.max_iterations}")
                
                # 다음 행동 결정
                next_action = await self._decide_next_action(strategy, current_context, execution_results)
                
                if not next_action:
                    self.logger.info("더 이상 수행할 행동이 없습니다.")
                    break
                
                # 행동 실행
                action_result = await self._execute_action(next_action, current_context)
                
                # 결과 평가 및 컨텍스트 업데이트
                evaluation = await self._evaluate_action_result(action_result, next_action)
                current_context = self._update_context(current_context, action_result)
                
                execution_results.append({
                    'iteration': iteration + 1,
                    'action': next_action,
                    'result': action_result,
                    'evaluation': evaluation
                })
                
                # 목표 달성 여부 확인
                if await self._is_goal_achieved(execution_results):
                    self.logger.info("목표 달성 완료")
                    break
                
                # 전략 조정 필요성 확인
                if evaluation.get('requires_strategy_adjustment'):
                    strategy = await self._adjust_strategy(strategy, execution_results, current_context)
                
                # 오류 발생 시 복구 시도
                if not action_result.get('success') and evaluation.get('severity') == 'high':
                    recovery_result = await self._attempt_recovery(action_result, current_context)
                    if recovery_result.get('success'):
                        current_context = self._update_context(current_context, recovery_result)
            
            return {
                'success': True,
                'execution_results': execution_results,
                'final_context': current_context,
                'iterations_used': len(execution_results)
            }
            
        except Exception as e:
            self.logger.error(f"실행 루프 오류: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_results': execution_results,
                'final_context': current_context
            }
    
    async def _decide_next_action(self, strategy: Dict[str, Any], context: Dict[str, Any],
                                execution_results: List[Dict[str, Any]]) -> Optional[AgentAction]:
        """다음 행동 결정"""
        try:
            # 현재 상황 평가
            current_situation = await self._assess_current_situation(context, execution_results)
            
            # 가능한 행동들 생성
            possible_actions = await self._generate_possible_actions(strategy, current_situation)
            
            if not possible_actions:
                return None
            
            # 최적 행동 선택
            best_action = await self._select_best_action(possible_actions, current_situation)
            
            # 행동 기록
            self.action_history.append(best_action)
            
            return best_action
            
        except Exception as e:
            self.logger.error(f"행동 결정 오류: {e}")
            return None
    
    async def _execute_action(self, action: AgentAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """행동 실행"""
        self.logger.info(f"행동 실행: {action.action_type.value}")
        
        try:
            if action.action_type == ActionType.ANALYZE_DATA:
                result = await self._execute_data_analysis(action, context)
            elif action.action_type == ActionType.SEARCH_KNOWLEDGE:
                result = await self._execute_knowledge_search(action, context)
            elif action.action_type == ActionType.GENERATE_CODE:
                result = await self._execute_code_generation(action, context)
            elif action.action_type == ActionType.EXECUTE_CODE:
                result = await self._execute_code_execution(action, context)
            elif action.action_type == ActionType.VALIDATE_ASSUMPTIONS:
                result = await self._execute_assumption_validation(action, context)
            elif action.action_type == ActionType.INTERPRET_RESULTS:
                result = await self._execute_result_interpretation(action, context)
            elif action.action_type == ActionType.ASK_USER:
                result = await self._execute_user_interaction(action, context)
            elif action.action_type == ActionType.ADJUST_STRATEGY:
                result = await self._execute_strategy_adjustment(action, context)
            else:
                result = {'success': False, 'error': f'Unknown action type: {action.action_type}'}
            
            # 행동 결과 기록
            action.result = result
            action.success = result.get('success', False)
            
            return result
            
        except Exception as e:
            self.logger.error(f"행동 실행 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _interpret_and_learn(self, execution_result: Dict[str, Any], 
                                 goal: AgentGoal) -> Dict[str, Any]:
        """결과 해석 및 학습"""
        self.logger.info("결과 해석 및 학습 시작")
        
        try:
            # 실행 결과 해석
            interpretation = await self._interpret_execution_results(execution_result, goal)
            
            # 성공/실패 패턴 학습
            if self.learning_enabled:
                await self._learn_from_execution(execution_result, goal)
            
            # 최종 보고서 생성
            final_report = await self._generate_final_report(interpretation, execution_result, goal)
            
            return {
                'success': execution_result.get('success', False),
                'interpretation': interpretation,
                'final_report': final_report,
                'execution_summary': self._create_execution_summary(execution_result),
                'agent_id': self.agent_id,
                'goal_achieved': await self._is_goal_achieved(execution_result.get('execution_results', []))
            }
            
        except Exception as e:
            self.logger.error(f"결과 해석 및 학습 오류: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_id': self.agent_id
            }
    
    # 헬퍼 메서드들
    async def _analyze_data_characteristics(self, data) -> Dict[str, Any]:
        """데이터 특성 분석"""
        if data is None:
            return {'type': 'none', 'characteristics': {}}
        
        # 데이터 기본 정보 분석
        characteristics = {
            'shape': getattr(data, 'shape', None),
            'columns': list(getattr(data, 'columns', [])),
            'dtypes': dict(getattr(data, 'dtypes', {})),
            'missing_values': dict(getattr(data, 'isnull', lambda: {})().sum()) if hasattr(data, 'isnull') else {},
            'summary_stats': data.describe().to_dict() if hasattr(data, 'describe') else {}
        }
        
        return {
            'type': 'dataframe' if hasattr(data, 'shape') else 'unknown',
            'characteristics': characteristics
        }
    
    async def _analyze_user_requirements(self, user_input) -> Dict[str, Any]:
        """사용자 요구사항 분석"""
        if not user_input:
            return {'requirements': [], 'analysis_type': 'unknown'}
        
        # LLM을 통한 요구사항 분석
        analysis_prompt = self.prompt_engine.create_requirement_analysis_prompt(user_input)
        response = await self.llm_client.generate_response(analysis_prompt)
        
        return self.response_parser.parse_requirement_analysis(response)
    
    async def _analyze_constraints(self, context: Dict[str, Any]) -> List[str]:
        """제약사항 분석"""
        constraints = []
        
        # 데이터 제약사항
        if context.get('data') is not None:
            data = context['data']
            if hasattr(data, 'shape') and data.shape[0] < 30:
                constraints.append("소표본 크기")
            if hasattr(data, 'isnull') and data.isnull().sum().sum() > 0:
                constraints.append("결측값 존재")
        
        # 시간 제약사항
        if self.current_goal and self.current_goal.deadline:
            time_left = (self.current_goal.deadline - datetime.now()).total_seconds()
            if time_left < 3600:  # 1시간 미만
                constraints.append("시간 제약")
        
        return constraints
    
    async def _analyze_available_resources(self) -> Dict[str, Any]:
        """가용 리소스 분석"""
        return {
            'llm_client': self.llm_client is not None,
            'knowledge_store': self.knowledge_store is not None,
            'code_runner': self.code_runner is not None,
            'statistical_tools': True,  # 통계 도구 가용성
            'visualization_tools': True  # 시각화 도구 가용성
        }
    
    def _assess_complexity(self, data_analysis: Dict[str, Any], 
                          user_requirements: Dict[str, Any]) -> str:
        """복잡도 평가"""
        complexity_score = 0
        
        # 데이터 복잡도
        if data_analysis.get('characteristics', {}).get('shape'):
            rows, cols = data_analysis['characteristics']['shape']
            if rows > 1000 or cols > 20:
                complexity_score += 2
            elif rows > 100 or cols > 10:
                complexity_score += 1
        
        # 요구사항 복잡도
        requirements = user_requirements.get('requirements', [])
        if len(requirements) > 3:
            complexity_score += 2
        elif len(requirements) > 1:
            complexity_score += 1
        
        if complexity_score >= 4:
            return "high"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _recommend_initial_approach(self, data_analysis: Dict[str, Any],
                                  user_requirements: Dict[str, Any]) -> str:
        """초기 접근법 추천"""
        analysis_type = user_requirements.get('analysis_type', 'unknown')
        
        if analysis_type in ['comparison', 'group_comparison']:
            return "statistical_testing"
        elif analysis_type in ['relationship', 'correlation']:
            return "correlation_analysis"
        elif analysis_type in ['prediction', 'modeling']:
            return "regression_analysis"
        else:
            return "exploratory_analysis"
    
    def _build_knowledge_query(self, situation_analysis: Dict[str, Any], 
                              goal: AgentGoal) -> str:
        """지식 검색 쿼리 구성"""
        query_parts = [goal.primary_objective]
        
        # 데이터 특성 추가
        data_chars = situation_analysis.get('data_characteristics', {})
        if data_chars.get('type'):
            query_parts.append(f"data type: {data_chars['type']}")
        
        # 추천 접근법 추가
        if situation_analysis.get('recommended_approach'):
            query_parts.append(f"approach: {situation_analysis['recommended_approach']}")
        
        return " ".join(query_parts)
    
    async def _validate_and_improve_strategy(self, strategy: Dict[str, Any],
                                           situation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """전략 검증 및 개선"""
        # 기본 검증
        if not strategy.get('steps'):
            strategy['steps'] = ['analyze_data', 'execute_analysis', 'interpret_results']
        
        # 상황에 맞는 조정
        complexity = situation_analysis.get('complexity_level', 'medium')
        if complexity == 'high':
            # 고복잡도의 경우 더 세분화된 단계 추가
            if 'validate_assumptions' not in strategy['steps']:
                strategy['steps'].insert(-1, 'validate_assumptions')
        
        return strategy
    
    # 추가 실행 메서드들 (간소화된 구현)
    async def _execute_data_analysis(self, action: AgentAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 분석 실행"""
        return {'success': True, 'analysis_type': 'basic_statistics'}
    
    async def _execute_knowledge_search(self, action: AgentAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """지식 검색 실행"""
        query = action.parameters.get('query', '')
        results = await self.query_engine.search(query)
        return {'success': True, 'results': results}
    
    async def _execute_code_generation(self, action: AgentAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """코드 생성 실행"""
        return {'success': True, 'code': 'generated_code_placeholder'}
    
    async def _execute_code_execution(self, action: AgentAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """코드 실행"""
        code = action.parameters.get('code', '')
        result = self.code_runner.execute_code(code, context)
        return {'success': result.success, 'execution_result': result}
    
    async def _execute_assumption_validation(self, action: AgentAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """가정 검증 실행"""
        return {'success': True, 'assumptions_met': True}
    
    async def _execute_result_interpretation(self, action: AgentAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """결과 해석 실행"""
        return {'success': True, 'interpretation': 'results_interpreted'}
    
    async def _execute_user_interaction(self, action: AgentAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 상호작용 실행"""
        return {'success': True, 'user_response': 'simulated_response'}
    
    async def _execute_strategy_adjustment(self, action: AgentAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """전략 조정 실행"""
        return {'success': True, 'adjusted_strategy': 'new_strategy'}
    
    # 기타 헬퍼 메서드들
    async def _assess_current_situation(self, context: Dict[str, Any], 
                                      execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """현재 상황 평가"""
        return {
            'progress': len(execution_results) / self.max_iterations,
            'last_action_success': execution_results[-1].get('result', {}).get('success', False) if execution_results else True,
            'context_size': len(context)
        }
    
    async def _generate_possible_actions(self, strategy: Dict[str, Any], 
                                       situation: Dict[str, Any]) -> List[AgentAction]:
        """가능한 행동들 생성"""
        actions = []
        
        # 전략의 다음 단계에 따른 행동 생성
        for step in strategy.get('steps', []):
            if step == 'analyze_data':
                actions.append(AgentAction(
                    action_type=ActionType.ANALYZE_DATA,
                    parameters={'analysis_type': 'descriptive'},
                    reasoning="데이터 기본 분석 필요",
                    confidence=0.8
                ))
            elif step == 'search_knowledge':
                actions.append(AgentAction(
                    action_type=ActionType.SEARCH_KNOWLEDGE,
                    parameters={'query': 'statistical analysis methods'},
                    reasoning="관련 지식 검색 필요",
                    confidence=0.7
                ))
        
        return actions
    
    async def _select_best_action(self, possible_actions: List[AgentAction],
                                situation: Dict[str, Any]) -> AgentAction:
        """최적 행동 선택"""
        if not possible_actions:
            return None
        
        # 신뢰도 기준으로 선택
        best_action = max(possible_actions, key=lambda a: a.confidence)
        return best_action
    
    async def _evaluate_action_result(self, result: Dict[str, Any], 
                                    action: AgentAction) -> Dict[str, Any]:
        """행동 결과 평가"""
        return {
            'success': result.get('success', False),
            'quality': 'good' if result.get('success') else 'poor',
            'requires_strategy_adjustment': not result.get('success'),
            'severity': 'low' if result.get('success') else 'medium'
        }
    
    def _update_context(self, context: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """컨텍스트 업데이트"""
        updated_context = context.copy()
        if result.get('success'):
            updated_context['last_successful_result'] = result
        return updated_context
    
    async def _is_goal_achieved(self, execution_results: List[Dict[str, Any]]) -> bool:
        """목표 달성 여부 확인"""
        if not execution_results:
            return False
        
        # 최근 결과들이 성공적인지 확인
        recent_results = execution_results[-3:] if len(execution_results) >= 3 else execution_results
        success_rate = sum(1 for r in recent_results if r.get('result', {}).get('success', False)) / len(recent_results)
        
        return success_rate >= 0.8
    
    async def _adjust_strategy(self, strategy: Dict[str, Any], 
                             execution_results: List[Dict[str, Any]],
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """전략 조정"""
        # 실패한 단계들 분석
        failed_steps = [r for r in execution_results if not r.get('result', {}).get('success', False)]
        
        if failed_steps:
            # 대안 전략 생성
            strategy['steps'] = ['search_knowledge', 'analyze_data', 'execute_analysis']
            strategy['adjusted'] = True
        
        return strategy
    
    async def _attempt_recovery(self, action_result: Dict[str, Any], 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """복구 시도"""
        return {'success': True, 'recovery_method': 'retry_with_different_approach'}
    
    async def _interpret_execution_results(self, execution_result: Dict[str, Any],
                                         goal: AgentGoal) -> Dict[str, Any]:
        """실행 결과 해석"""
        return {
            'goal_achievement': execution_result.get('success', False),
            'key_findings': ['finding1', 'finding2'],
            'recommendations': ['recommendation1', 'recommendation2']
        }
    
    async def _learn_from_execution(self, execution_result: Dict[str, Any], goal: AgentGoal):
        """실행으로부터 학습"""
        # 성공/실패 패턴을 메모리에 저장
        pattern = {
            'goal_type': goal.primary_objective,
            'success': execution_result.get('success', False),
            'strategy_used': 'default',
            'timestamp': datetime.now()
        }
        self.memory.learned_patterns[f"pattern_{len(self.memory.learned_patterns)}"] = pattern
    
    async def _generate_final_report(self, interpretation: Dict[str, Any],
                                   execution_result: Dict[str, Any],
                                   goal: AgentGoal) -> Dict[str, Any]:
        """최종 보고서 생성"""
        return {
            'goal': goal.primary_objective,
            'success': execution_result.get('success', False),
            'interpretation': interpretation,
            'execution_summary': self._create_execution_summary(execution_result),
            'agent_id': self.agent_id,
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_execution_summary(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """실행 요약 생성"""
        return {
            'total_iterations': execution_result.get('iterations_used', 0),
            'success_rate': 1.0 if execution_result.get('success') else 0.0,
            'key_actions': len(self.action_history),
            'final_state': self.state.value
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Agent 상태 반환"""
        return {
            'agent_id': self.agent_id,
            'state': self.state.value,
            'current_goal': self.current_goal.primary_objective if self.current_goal else None,
            'action_count': len(self.action_history),
            'memory_size': len(self.memory.conversation_history),
            'learning_enabled': self.learning_enabled
        } 