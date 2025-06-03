# src/solution_advisor.py

import logging
import os
from typing import Dict, Any, List
import re

from .llm_service import LLMService
from .rag_retriever import RAGRetriever
# from .utils import format_risk_for_llm # 위험 평가 결과 포맷팅 유틸리티

logger = logging.getLogger(__name__)

class SolutionAdvisor:
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.prompts_base_path = settings.get('paths', {}).get('prompts_base', 'prompts/')
        self.llm_service = LLMService(settings)
        self.rag_retriever = RAGRetriever(settings)
        logger.info("SolutionAdvisor initialized.")

    def _load_prompt_template(self, prompt_file_path_relative: str) -> str:
        """Helper to load prompt template from the prompts directory."""
        full_path = os.path.join(self.prompts_base_path, prompt_file_path_relative)
        if not os.path.exists(full_path):
            logger.error(f"Prompt template file not found: {full_path}")
            return "분석된 위험에 대한 구체적인 주의사항, 계약 시 추가할 특약 조건(초안), 그리고 필요하다면 임대인/중개인에게 요청해야 할 추가 자료 목록을 제안해주세요. 최종적으로 계약 진행 여부에 대한 권고를 포함해주세요."
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading prompt template {full_path}: {e}", exc_info=True)
            return f"오류: 프롬프트 '{prompt_file_path_relative}' 로드 실패."

    def _prepare_advice_context(self, processed_input_data: Dict[str, Any], external_data: Dict[str, Any], risk_assessment_result: Dict[str, Any]) -> str:
        """LLM 조언 생성 프롬프트에 사용될 컨텍스트 문자열을 생성합니다."""
        # TODO: 필요한 정보를 요약하여 문자열로 구성
        context_parts = []
        context_parts.append("[계약 기본 정보 요약]")
        context_parts.append(f"  - 주소: {processed_input_data.get('property_details',{}).get('address_full', 'N/A')}")
        context_parts.append(f"  - 전세보증금: {processed_input_data.get('contract_info',{}).get('jeonse_deposit_amount', 'N/A')}")
        
        context_parts.append("\n[위험 평가 결과 요약]")
        context_parts.append(f"  - 종합 위험도: {risk_assessment_result.get('risk_level', 'N/A')}")
        context_parts.append(f"  - 위험 요약: {risk_assessment_result.get('summary', 'N/A')[:200]}...") # 요약 일부
        context_parts.append("  - 주요 식별 위험:")
        for risk in risk_assessment_result.get('identified_risks', []):
            context_parts.append(f"    - 항목: {risk.get('item', 'N/A')}, 내용: {risk.get('details', 'N/A')[:100]}...")
        
        return "\n".join(context_parts)

    def _parse_llm_advice_response(self, llm_response: str) -> Dict[str, Any]:
        logger.debug(f"Parsing LLM solution advice response: {llm_response[:300]}...")
        # 간단한 권고/특약/주의사항 추출
        recommendation = None
        rec_match = re.search(r"권고[\s:：]+(.+)", llm_response)
        if rec_match:
            recommendation = rec_match.group(1).strip()
        precautions = re.findall(r"주의사항[\s:：]+(.+)", llm_response)
        special_clauses = re.findall(r"특약[\s:：]+(.+)", llm_response)
        additional_docs = re.findall(r"추가 요청 서류[\s:：]+(.+)", llm_response)
        return {
            "recommendation": recommendation or "분석 필요 (LLM 응답 파싱 필요)",
            "precautions": precautions or ["LLM 응답에서 주의사항 목록 파싱 필요"],
            "special_clauses_draft": special_clauses or ["LLM 응답에서 특약 조건 초안 목록 파싱 필요"],
            "additional_documents_to_request": additional_docs or ["LLM 응답에서 추가 요청 서류 목록 파싱 필요"],
            "full_llm_advice_response": llm_response
        }

    def advise(self, processed_input_data: Dict[str, Any], external_data: Dict[str, Any], risk_assessment_result: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Generating solutions and advice using LLM and RAG...")

        # 1. 조언 생성 컨텍스트 준비
        advice_context_str = self._prepare_advice_context(processed_input_data, external_data, risk_assessment_result)

        # 2. RAG를 통해 관련 솔루션 정보 검색 (예: 위험 수준별 대응 가이드, 표준 특약 등)
        rag_query = f"위험 평가 결과({risk_assessment_result.get('risk_level', '정보 없음')}) 기반 전세 계약 주의사항 및 안전장치"
        retrieved_rag_docs = self.rag_retriever.retrieve_documents(rag_query, custom_top_k=3) # 조언에는 좀 더 타겟된 정보

        rag_context_for_llm = "\n\n[참고 RAG 자료 - 대응 가이드 및 특약 예시]\n"
        if retrieved_rag_docs:
            for i, doc in enumerate(retrieved_rag_docs):
                rag_context_for_llm += f"문서 {i+1} (출처: {doc['metadata'].get('source', 'N/A')} | 유형: {doc['metadata'].get('type', '정보')} | 유사도: {doc.get('score', 'N/A'):.2f}):\n"
                rag_context_for_llm += f"{doc['content'][:250]}...\n\n"
        else:
            rag_context_for_llm += "연관된 RAG 자료를 찾지 못했습니다.\n"
            
        # 3. LLM 프롬프트 구성
        prompt_template = self._load_prompt_template("advisory_generation_prompt.md")
        
        final_prompt = prompt_template.format(
            analysis_summary=advice_context_str,
            rag_references=rag_context_for_llm
        )
        
        system_message_template = self._load_prompt_template("system_role_prompt.md")
        # 시스템 메시지를 구체화하여 LLM의 역할을 명확히 할 수 있습니다.
        system_message = system_message_template.format(task_description="제공된 위험 분석 결과를 바탕으로, 사용자에게 명확하고 실행 가능한 계약 관련 조언, 주의사항, 추천 특약(초안) 및 필요한 추가 자료 요청 목록을 생성하는 역할. 최종적으로 계약 진행에 대한 권고 의견을 제시해야 함.")

        # 4. LLM 호출
        llm_response = self.llm_service.generate_response(final_prompt, system_message=system_message)

        if not llm_response:
            logger.error("Solution advice generation failed due to LLM error or empty response.")
            return {
                "recommendation": "판단 유보", 
                "precautions": ["LLM 분석 실패로 주의사항 생성 불가"],
                "special_clauses_draft": ["LLM 분석 실패로 특약 생성 불가"],
                "additional_documents_to_request": [],
                "full_llm_advice_response": None
            }

        # 5. LLM 응답 파싱 및 결과 반환
        advice_result = self._parse_llm_advice_response(llm_response)
        
        logger.info(f"Solution advice generated. Recommendation: {advice_result.get('recommendation')}")
        return advice_result