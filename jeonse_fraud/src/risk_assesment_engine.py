# src/risk_assessment_engine.py

import logging
import os # 프롬프트 파일 로드용
from typing import Dict, Any, List
import re

from .llm_service import LLMService
from .rag_retriever import RAGRetriever
# from .utils import format_data_for_llm # 데이터 포맷팅 유틸리티 (필요시 src/utils.py 에 생성)

logger = logging.getLogger(__name__)

class RiskAssessmentEngine:
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.prompts_base_path = settings.get('paths', {}).get('prompts_base', 'prompts/')
        self.llm_service = LLMService(settings)
        self.rag_retriever = RAGRetriever(settings)
        logger.info("RiskAssessmentEngine initialized.")

    def _load_prompt_template(self, prompt_file_path_relative: str) -> str:
        """Helper to load prompt template from the prompts directory."""
        full_path = os.path.join(self.prompts_base_path, prompt_file_path_relative)
        if not os.path.exists(full_path):
            logger.error(f"Prompt template file not found: {full_path}")
            # 실제 상황에 맞는 기본 프롬프트 또는 예외 처리
            return "제공된 정보를 바탕으로 전세 계약의 위험 요소를 상세히 분석하고, 각 위험 항목에 대한 평가와 근거를 제시해주세요. 최종적으로 종합 위험도를 [매우 높음, 높음, 주의, 낮음, 정보 부족으로 판단 유보] 중 하나로 평가해주세요."
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading prompt template {full_path}: {e}", exc_info=True)
            return f"오류: 프롬프트 '{prompt_file_path_relative}' 로드 실패."

    def _prepare_assessment_context(self, processed_input_data: Dict[str, Any], external_data: Dict[str, Any]) -> str:
        """LLM 프롬프트에 포함될 분석 대상 데이터의 요약 문자열을 생성합니다."""
        # TODO: 입력 데이터들을 LLM이 이해하기 쉬운 텍스트 형태로 가공하는 로직 필요
        # 예시:
        context_parts = []
        context_parts.append("[입력된 등기부등본 주요 내용]")
        context_parts.append(str(processed_input_data.get('property_register', '등기부 정보 없음'))) # 실제로는 json.dumps 등으로 보기 좋게
        
        context_parts.append("\n[입력된 계약 조건 주요 내용]")
        context_parts.append(str(processed_input_data.get('contract_info', '계약 조건 정보 없음')))
        
        context_parts.append("\n[외부 API 조회된 건축물대장 정보]")
        context_parts.append(str(external_data.get('building_ledger', '건축물대장 정보 없음')))
        
        context_parts.append("\n[외부 API 조회된 실거래가 정보]")
        context_parts.append(str(external_data.get('transaction_price', '실거래가 정보 없음')))
        
        # 임대인 관련 정보 (선택 입력 사항) 처리
        lessor_info = processed_input_data.get('contract_info', {}).get('lessor_optional_info')
        if lessor_info:
            context_parts.append("\n[입력된 임대인 관련 추가 정보]")
            context_parts.append(str(lessor_info))
        else:
            context_parts.append("\n[입력된 임대인 관련 추가 정보: 제공되지 않음]")


        return "\n".join(context_parts)

    def _parse_llm_risk_assessment_response(self, llm_response: str) -> Dict[str, Any]:
        logger.debug(f"Parsing LLM risk assessment response: {llm_response[:300]}...")
        # 간단한 위험도 추출
        risk_level = "정보 부족으로 판단 유보"
        match = re.search(r"위험도\s*[:：]\s*([가-힣]+)", llm_response)
        if match:
            risk_level = match.group(1)
        # 위험 항목 추출 (예: '항목: ...', '상세 분석: ...' 등)
        identified_risks = []
        for m in re.finditer(r"항목[:：]\s*(.+?)\n- \*\*상세 분석[:：]\*\* (.+?)\n- \*\*판단 근거.*?\*\* (.+?)(?:\n|$)", llm_response, re.DOTALL):
            identified_risks.append({
                "item": m.group(1).strip(),
                "details": m.group(2).strip(),
                "evidence_source": m.group(3).strip(),
                "rag_reference": None
            })
        return {
            "risk_level": risk_level,
            "summary": llm_response[:200],
            "identified_risks": identified_risks if identified_risks else [
                {"item": "LLM 응답에서 위험 항목 1 파싱 필요", "details": "...", "evidence_source": "...", "rag_reference": "..."}
            ],
            "raw_llm_response": llm_response
        }

    def assess(self, processed_input_data: Dict[str, Any], external_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Performing risk assessment using LLM and RAG...")

        # 1. 분석 컨텍스트 준비
        analysis_context_str = self._prepare_assessment_context(processed_input_data, external_data)

        # 2. RAG를 통해 관련 정보 검색
        #    분석 컨텍스트 또는 핵심 키워드를 질의로 사용
        rag_query = f"다음 부동산 계약 관련 위험 분석: {analysis_context_str[:500]}" # 너무 길지 않게
        retrieved_rag_docs = self.rag_retriever.retrieve_documents(rag_query)
        
        rag_context_for_llm = "\n\n[참고 RAG 자료]\n"
        if retrieved_rag_docs:
            for i, doc in enumerate(retrieved_rag_docs):
                rag_context_for_llm += f"문서 {i+1} (출처: {doc['metadata'].get('source', 'N/A')} | 유형: {doc['metadata'].get('type', '정보')} | 유사도: {doc.get('score', 'N/A'):.2f}):\n"
                rag_context_for_llm += f"{doc['content'][:300]}...\n\n" # 내용 요약 또는 일부
        else:
            rag_context_for_llm += "연관된 RAG 자료를 찾지 못했습니다.\n"

        # 3. LLM 프롬프트 구성 (연쇄 프롬프트 중 첫 단계 또는 통합 프롬프트 사용)
        #    예시: "risk_assessment_chain/03_overall_risk_synthesis_prompt.md" 사용
        prompt_template = self._load_prompt_template("risk_assessment_chain/03_overall_risk_synthesis_prompt.md")
        
        final_prompt = prompt_template.format(
            case_data_summary=analysis_context_str,
            rag_information=rag_context_for_llm
        )
        
        system_message_template = self._load_prompt_template("system_role_prompt.md")
        system_message = system_message_template.format(task_description="제공된 모든 정보를 종합하여 전세 계약의 최종 위험도를 평가하고, 각 위험 항목과 그 근거를 상세히 설명하는 역할")

        # 4. LLM 호출
        llm_response = self.llm_service.generate_response(final_prompt, system_message=system_message)

        if not llm_response:
            logger.error("Risk assessment failed due to LLM error or empty response.")
            return {"risk_level": "정보 부족으로 판단 유보", "summary": "LLM 분석 실패", "identified_risks": [], "raw_llm_response": None}

        # 5. LLM 응답 파싱 및 결과 반환
        assessment_result = self._parse_llm_risk_assessment_response(llm_response)
        
        logger.info(f"Risk assessment completed. Determined risk level: {assessment_result.get('risk_level')}")
        return assessment_result