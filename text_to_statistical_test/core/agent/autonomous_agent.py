"""
Autonomous Agent for Statistical Analysis

LLM을 사용하여 통계 분석의 계획, 실행, 해석을 자율적으로 수행하는 에이전트.
'Orchestrator-Engine' 모델에 따라, 복잡한 로직은 서비스 계층에 위임하고
자신은 'Plan-Execute-Interpret'의 핵심 흐름을 담당한다.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from services.llm.llm_service import LLMService
# services/__init__.py 에서 완성된 인스턴스를 가져옵니다.
from services import statistics_service
from .tools import ToolRegistry

class AutonomousAgent:
    """
    자율적으로 통계 분석을 수행하는 주체.
    """
    
    def __init__(self, llm_service: LLMService):
        self.logger = logging.getLogger(__name__)
        self.llm_service = llm_service
        
        # ToolRegistry 초기화 시 모든 서비스 인스턴스 주입
        self.tool_registry = ToolRegistry(
            stats_service=statistics_service
        )
        
        self.logger.info(f"AutonomousAgent 초기화 완료. 사용 가능한 도구: {len(self.tool_registry.get_tool_definitions())}개")

    async def run_analysis(
        self,
        dataframe: pd.DataFrame,
        structured_request: Dict[str, Any],
        knowledge_context: str = ""  # RAG를 통해 보강된 컨텍스트
    ) -> Dict[str, Any]:
        """
        주어진 데이터와 구조화된 요청에 따라 자율적으로 통계 분석을 수행합니다.
        '계획 수립 -> 도구 실행 -> 결과 종합'의 3단계 워크플로우를 따릅니다.

        Args:
            dataframe (pd.DataFrame): 분석 대상 데이터.
            structured_request (Dict[str, Any]): UserRequestStep에서 생성된 구조화된 분석 목표.
            knowledge_context (str): RAG를 통해 검색된 관련 지식 컨텍스트.

        Returns:
            Dict[str, Any]: 분석의 모든 과정을 포함하는 최종 결과.
                           (계획, 가정 검토, 본 분석, 효과 크기, 사후 검정, 최종 해석 등)
        """
        print("\n" + "="*15 + " 자율 분석 시작 " + "="*15)

        # 사용 가능한 도구 목록을 LLM에 전달하여 계획 수립
        tool_definitions = self.tool_registry.get_tool_definitions()

        # 1. 계획 수립 (Plan)
        analysis_plan = await self._create_analysis_plan(
            structured_request, dataframe, tool_definitions, knowledge_context
        )
        if not analysis_plan or not analysis_plan.get('steps'):
            raise RuntimeError("LLM을 통해 분석 계획을 수립하는 데 실패했습니다.")

        # 2. 계획 실행 (Execute)
        execution_results = await self._execute_plan(analysis_plan, dataframe)

        # 3. 결과 해석 및 종합 (Interpret)
        final_summary = await self._interpret_results(
            structured_request=structured_request,
            analysis_plan=analysis_plan,
            execution_results=execution_results
        )
        
        print("="*15 + " 자율 분석 종료 " + "="*15)
        
        # 분석의 모든 과정을 포함하는 포괄적인 결과 객체를 반환합니다.
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
        knowledge_context: str  # RAG 컨텍스트 추가
    ) -> Dict[str, Any]:
        """LLM을 호출하여 상세한 단계별 분석 계획을 수립합니다."""
        print("-> (1/3) 분석 계획 수립 중...")
        try:
            plan = await self.llm_service.create_analysis_plan(
                structured_request=structured_request,
                dataframe=dataframe,
                tool_definitions=tool_definitions,
                knowledge_context=knowledge_context  # LLM 서비스에 전달
            )
            self.logger.info(f"분석 계획 수립 완료: {len(plan.get('steps', []))} 단계")
            self.logger.debug(f"수립된 계획: {plan}")
            return plan
        except Exception as e:
            self.logger.error(f"분석 계획 수립 중 오류: {e}", exc_info=True)
            return {}

    async def _execute_plan(
        self,
        analysis_plan: Dict[str, Any],
        dataframe: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        수립된 계획에 따라 각 단계를 순차적으로 실행하고, 도구를 사용합니다.
        이전 단계의 결과를 다음 단계의 입력으로 지능적으로 연결(wiring)합니다.
        """
        print("-> (2/3) 분석 계획 실행 중...")
        results = []
        
        # 이전 통계 테스트의 결과와 파라미터를 저장하기 위한 변수
        last_test_result: Optional[Dict[str, Any]] = None
        last_test_params: Optional[Dict[str, Any]] = None

        for i, step in enumerate(analysis_plan.get("steps", [])):
            tool_name = step.get("tool_name")
            params = step.get("params", {})
            step_name = step.get("step_name", f"Unnamed Step {i+1}")
            print(f"--> 실행 ({i+1}/{len(analysis_plan['steps'])}): {step_name}")
            
            if not tool_name:
                self.logger.warning(f"'{step_name}' 단계에 실행할 도구가 지정되지 않았습니다. 건너뜁니다.")
                continue

            try:
                # [!!!] 최종 데이터 와이어링 로직 (v2)
                # 각 도구의 요구사항에 맞춰, 필요한 컨텍스트만 선택적으로 주입합니다.
                if last_test_params:
                    # 이전 파라미터가 필요한 도구들 (`calculate_effect_size`, `run_posthoc_test`)
                    if tool_name in ["calculate_effect_size", "run_posthoc_test"]:
                        final_params = last_test_params.copy()
                        final_params.update(params)
                        params = final_params
                        self.logger.info(f"'{tool_name}'에 이전 테스트 파라미터를 주입합니다. 최종 파라미터: {params}")

                if last_test_result:
                    # 이전 결과가 필요한 도구 (`calculate_effect_size`)
                    if tool_name == "calculate_effect_size":
                        params["test_results"] = last_test_result
                        self.logger.info("이전 테스트 결과를 'test_results'에 주입합니다.")
                
                # 도구 레지스트리를 통해 실제 실행할 함수를 가져옴
                tool_function = self.tool_registry.get_tool(tool_name)
                
                self.logger.info(f"'{tool_name}' 실행. 파라미터: {params}")
                result = tool_function(data=dataframe, **params)
                
                # 다음 단계를 위해 주 분석(run_statistical_test)의 결과와 파라미터를 저장
                if tool_name == "run_statistical_test":
                    self.logger.info(f"'{tool_name}'의 결과와 파라미터를 후속 단계를 위해 저장합니다.")
                    last_test_result = result
                    last_test_params = params.copy()

                step_result = {
                    "step_name": step_name,
                    "tool_name": tool_name,
                    "params": params,
                    "output": result,
                    "status": "SUCCESS"
                }

            except Exception as e:
                self.logger.error(f"'{tool_name}' 도구 실행 중 오류: {e}", exc_info=True)
                step_result = {
                    "step_name": step_name,
                    "tool_name": tool_name,
                    "params": params,
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
            # 해석에 실패하더라도 수집된 데이터는 반환
            return {
                "error": "Failed to generate final summary.",
                "details": str(e),
                "structured_request": structured_request,
                "analysis_plan": analysis_plan,
                "execution_results": execution_results
            } 