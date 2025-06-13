# autonomous_agent.py

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from dataclasses import dataclass

from services.llm.llm_service import LLMService
from services.statistics.stats_service import StatisticsService
from .tools import ToolRegistry


# 실행 컨텍스트를 위한 데이터 클래스 정의
@dataclass
class AgentRunState:
    """분석 계획 실행 중 단계 간 상태를 명시적으로 관리하는 내부 상태 객체."""
    dataframe: pd.DataFrame
    last_test_result: Optional[Dict[str, Any]] = None
    last_test_params: Optional[Dict[str, Any]] = None


class AutonomousAgent:
    """
    자율적으로 통계 분석을 수행하는 주체.
    """
    
    def __init__(self, llm_service: LLMService, stats_service: StatisticsService):
        self.logger = logging.getLogger(__name__)
        self.llm_service = llm_service
        self.tool_registry = ToolRegistry(stats_service=stats_service)
        self.state: Optional[AgentRunState] = None
        self.logger.info(f"AutonomousAgent 초기화 완료. 사용 가능한 도구: {len(self.tool_registry.get_tool_definitions())}개")

    async def run_analysis(
        self,
        dataframe: pd.DataFrame,
        structured_request: Dict[str, Any],
        knowledge_context: str = ""
    ) -> Dict[str, Any]:
        """
        주어진 데이터와 구조화된 요청에 따라 자율적으로 통계 분석을 수행합니다.
        '계획 수립 -> 도구 실행 -> 결과 종합'의 3단계 워크플로우를 따릅니다.
        """
        print("\n" + "="*15 + " 자율 분석 시작 " + "="*15)

        # [수정] 에이전트 실행 시 상태 객체를 생성하여 인스턴스 변수에 할당합니다.
        self.state = AgentRunState(dataframe=dataframe)

        tool_definitions = self.tool_registry.get_tool_definitions()
        analysis_plan = await self._create_analysis_plan(
            structured_request, dataframe, tool_definitions, knowledge_context
        )
        if not analysis_plan or not analysis_plan.get('steps'):
            raise RuntimeError("LLM을 통해 분석 계획을 수립하는 데 실패했습니다.")

        # [수정] _execute_plan 호출 시 더 이상 dataframe을 전달하지 않습니다.
        execution_results = await self._execute_plan(analysis_plan)

        final_summary = await self._interpret_results(
            structured_request=structured_request,
            analysis_plan=analysis_plan,
            execution_results=execution_results
        )
        
        print("="*15 + " 자율 분석 종료 " + "="*15)
        
        return {
            "analysis_plan": analysis_plan,
            "execution_results": execution_results,
            "final_summary": final_summary
        }

    async def _create_analysis_plan(
        self,
        structured_request: Dict[str, Any],
        dataframe: pd.DataFrame,
        tool_definitions: List[Dict[str, Any]],
        knowledge_context: str
    ) -> Dict[str, Any]:
        """LLM을 호출하여 상세한 단계별 분석 계획을 수립합니다."""
        print("-> (1/3) 분석 계획 수립 중...")
        try:
            plan = await self.llm_service.create_analysis_plan(
                structured_request=structured_request,
                dataframe=dataframe,
                tool_definitions=tool_definitions,
                knowledge_context=knowledge_context
            )
            self.logger.info(f"분석 계획 수립 완료: {len(plan.get('steps', []))} 단계")
            return plan
        except Exception as e:
            self.logger.error(f"분석 계획 수립 중 오류: {e}", exc_info=True)
            return {}

    # [수정] 메서드 시그니처에서 불필요한 dataframe 파라미터를 제거합니다.
    async def _execute_plan(
        self,
        analysis_plan: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        수립된 계획에 따라 각 단계를 순차적으로 실행하고, 도구를 사용합니다.
        self.state를 통해 단계 간 상태를 명시적으로 관리합니다.
        """
        print("-> (2/3) 분석 계획 실행 중...")
        results = []

        for i, step in enumerate(analysis_plan.get("steps", [])):
            tool_name = step.get("tool_name")
            params = step.get("params", {})
            step_name = step.get("step_name", f"Unnamed Step {i+1}")
            print(f"--> 실행 ({i+1}/{len(analysis_plan['steps'])}): {step_name}")
            
            if not tool_name:
                self.logger.warning(f"'{step_name}' 단계에 실행할 도구가 지정되지 않았습니다. 건너뜁니다.")
                continue
            
            final_params = params.copy()
            try:
                # [수정] 존재하지 않는 지역 변수 'context' 대신 'self.state'를 사용합니다.
                if tool_name == "calculate_effect_size" and self.state.last_test_result:
                    self.logger.info("이전 테스트 결과를 'calculate_effect_size'의 컨텍스트로 전달합니다.")
                    final_params["test_results"] = self.state.last_test_result.get("test_results", {})

                # 도구 실행
                tool_function = self.tool_registry.get_tool(tool_name)
                self.logger.info(f"'{tool_name}' 실행. 파라미터: {final_params}")
                result = tool_function(data=self.state.dataframe, **final_params)
                
                # 컨텍스트 업데이트 시 'self.state'를 사용
                if tool_name == "run_statistical_test":
                    self.logger.info(f"'{tool_name}'의 결과와 파라미터를 컨텍스트에 저장합니다.")
                    self.state.last_test_result = result
                    self.state.last_test_params = final_params.copy()

                step_result = {
                    "step_name": step_name,
                    "tool_name": tool_name,
                    "params": final_params,
                    "output": result,
                    "status": "SUCCESS"
                }

            except Exception as e:
                self.logger.error(f"'{tool_name}' 도구 실행 중 오류: {e}", exc_info=True)
                step_result = {
                    "step_name": step_name,
                    "tool_name": tool_name,
                    "params": final_params,
                    "error": str(e),
                    "status": "FAILED"
                }
            
            results.append(step_result)
        
        self.logger.info("분석 계획 실행 완료.")
        return results

    async def _interpret_results(
        self,
        structured_request: Dict[str, Any],
        analysis_plan: Dict[str, Any],
        execution_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """실행된 모든 결과를 종합하여 최종 해석을 생성합니다."""
        print("-> (3/3) 최종 결과 해석 및 종합 중...")
        try:
            summary = await self.llm_service.summarize_analysis_results(
                structured_request=structured_request,
                analysis_plan=analysis_plan,
                execution_results=execution_results
            )
            self.logger.info("최종 결과 해석 및 종합 완료.")
            return summary
        except Exception as e:
            self.logger.error(f"최종 결과 해석 중 오류: {e}", exc_info=True)
            return {
                "error": "Failed to generate final summary.",
                "details": str(e)
            }