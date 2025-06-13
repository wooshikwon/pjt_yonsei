# 파일명: core/pipeline/step2_autonomous_analysis.py
import logging
from typing import Dict, Any
import pandas as pd

from core.agent.autonomous_agent import AutonomousAgent
from core.pipeline.app_context import AppContext
from core.pipeline.step0_pipeline import PipelineStep
# [수정] 서비스 타입 힌트를 위해 클래스만 임포트
from services.llm.llm_service import LLMService
from services.rag.rag_service import RAGService
from services.statistics.stats_service import StatisticsService

class AutonomousAnalysisStep(PipelineStep):
    """2단계: AI Agent의 자율적 통계 분석 실행 (RAG 연동)"""
    
    # [수정] 생성자에서 모든 필요 서비스를 주입받음
    def __init__(self, llm_service: LLMService, rag_service: RAGService, stats_service: StatisticsService):
        super().__init__("자율 통계 분석")
        self.logger = logging.getLogger(__name__)
        # [수정] 주입받은 서비스 인스턴스를 사용
        self.llm_service = llm_service
        self.rag_service = rag_service
        self.stats_service = stats_service

    async def run(self, context: AppContext) -> AppContext:
        """
        사용자 요청을 구조화하고, RAG로 지식을 보강한 뒤, AI Agent를 통해 분석을 수행합니다.
        결과는 context 객체에 직접 추가됩니다.

        Args:
            context: 'dataframe', 'user_request' 키가 포함된 AppContext 객체.
        """
        df = context.dataframe
        user_request = context.user_request

        if not isinstance(df, pd.DataFrame) or not user_request:
            raise ValueError("AutonomousAnalysisStep: 'dataframe' 또는 'user_request'가 컨텍스트에 없습니다.")

        self.logger.info("사용자 요청을 구조화된 분석 목표로 변환합니다...")
        structured_request = await self.llm_service.structure_user_request(
            user_request=user_request,
            dataframe=df
        )
        context.structured_request = structured_request
        self.logger.info(f"구조화된 요청 생성 완료: {structured_request}")

        self.logger.info("RAG를 통해 관련 지식을 검색합니다...")
        retrieved_knowledge_text, retrieved_knowledge_raw = self._retrieve_knowledge(structured_request)

        self.logger.info("자율 분석 Agent를 초기화하고 실행합니다.")
        agent = AutonomousAgent(
            llm_service=self.llm_service,
            stats_service=self.stats_service
        )

        try:
            analysis_results = await agent.run_analysis(
                dataframe=df,
                structured_request=structured_request,
                knowledge_context=retrieved_knowledge_text
            )
        except Exception as e:
            self.logger.error("Autonomous Agent 실행 중 오류 발생.", exc_info=True)
            raise RuntimeError("자율 분석 단계에서 심각한 오류가 발생했습니다.") from e

        if not analysis_results:
            self.logger.error("Agent가 분석 결과를 반환하지 않았습니다.")
            raise ValueError("자율 분석이 실패했거나 결과를 생성하지 못했습니다.")

        self.logger.info("Agent 기반 자율 분석을 성공적으로 완료했습니다.")

        # Agent가 반환하는 결과 객체를 컨텍스트에 분해하여 저장합니다.
        context.analysis_plan = analysis_results.get("analysis_plan")
        context.execution_results = analysis_results.get("execution_results")
        context.final_summary = analysis_results.get("final_summary")
        context.retrieved_knowledge_text = retrieved_knowledge_text
        context.retrieved_knowledge_raw = retrieved_knowledge_raw

        final_summary_title = (context.final_summary or {}).get('title', 'N/A')
        self.logger.info(f"자율 분석 완료. 최종 요약 제목: {final_summary_title}")

        return context

    def _retrieve_knowledge(self, structured_request: Dict[str, Any]) -> tuple[str, list[dict]]:
        """구조화된 요청을 바탕으로 RAG 서비스에서 관련 지식을 검색합니다."""
        # 검색 쿼리를 생성 (요청의 핵심 내용을 조합)
        query = f"{structured_request.get('question', '')} " \
                f"분석 유형: {structured_request.get('analysis_type', '')} " \
                f"변수: {', '.join(structured_request.get('variables', []))}"
        
        self.logger.debug(f"RAG 검색 쿼리: {query}")
        
        try:
            # 여기서는 모든 컬렉션을 대상으로 검색
            search_results = self.rag_service.search(query, top_k=5, collection=None)
            
            if not search_results:
                self.logger.warning("RAG 검색 결과가 없습니다.")
                return "", []

            context_text = self.rag_service.build_context_from_results(search_results)
            return context_text, search_results

        except Exception as e:
            self.logger.error(f"RAG 검색 중 오류 발생: {e}", exc_info=True)
            # RAG가 실패해도 분석은 계속될 수 있도록 빈 컨텍스트 반환
            return "", []