# src/report_generator.py

import logging
import os
import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.reports_base_path = settings.get('paths', {}).get('reports_base', 'reports/')
        self.docs_base_path = settings.get('paths', {}).get('docs_base', 'docs/')
        self.output_format = settings.get('report', {}).get('output_format', 'markdown')
        self.include_appendix = settings.get('report', {}).get('include_evaluation_appendix', True)
        
        if not os.path.exists(self.reports_base_path):
            try:
                os.makedirs(self.reports_base_path, exist_ok=True)
            except OSError as e:
                logger.error(f"Could not create reports directory at {self.reports_base_path}. Error: {e}")
                # 보고서 생성 실패를 유발할 수 있으므로, 필요시 예외 발생
        logger.info("ReportGenerator initialized.")

    def _load_appendix_content(self) -> str:
        """평가 기준표 부록 내용을 로드합니다."""
        if not self.include_appendix:
            return ""
            
        appendix_file_path = os.path.join(self.docs_base_path, "evaluation_framework.md")
        if os.path.exists(appendix_file_path):
            try:
                with open(appendix_file_path, 'r', encoding='utf-8') as f:
                    return f"\n\n---\n## 부록: 상세 평가 기준 프레임워크\n\n{f.read()}"
            except Exception as e:
                logger.error(f"Error loading evaluation framework appendix from {appendix_file_path}: {e}")
                return "\n\n부록: 상세 평가 기준 프레임워크 로드 실패.\n"
        else:
            logger.warning(f"Evaluation framework appendix file not found at: {appendix_file_path}")
            return "\n\n부록: 상세 평가 기준 프레임워크 파일 없음.\n"

    def _format_report_content(self, test_case_id: str, processed_input_data: Dict[str, Any], external_data: Dict[str, Any], risk_assessment_result: Dict[str, Any], solution_advice_result: Dict[str, Any]) -> str:
        """보고서 내용을 Markdown 형식으로 구성합니다."""
        # TODO: 입력 데이터, 외부 데이터 등을 좀 더 보기 좋게 요약하는 로직 추가
        input_summary = f"- 등기부 정보 요약: {str(processed_input_data.get('property_register',{}).get('소재지번','N/A'))[:100]}...\n"
        input_summary += f"- 계약 정보 요약: 전세가 {processed_input_data.get('contract_info',{}).get('jeonse_deposit_amount','N/A')}, 주소 {processed_input_data.get('contract_info',{}).get('property_address','N/A')}\n"
        
        external_summary = "- 건축물대장 정보: " + ("정상 조회" if external_data.get('building_ledger') and not external_data['building_ledger'].get('error') else external_data.get('building_ledger',{}).get('error', '조회 실패')) + "\n"
        external_summary += "- 실거래가 정보: " + ("정상 조회" if external_data.get('transaction_price') and not external_data['transaction_price'].get('error') else external_data.get('transaction_price',{}).get('error', '조회 실패')) + "\n"


        content = f"# 전세 계약 위험 분석 보고서 (Test Case: {test_case_id})\n\n"
        content += f"**분석 일시:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        content += "## 1. 분석 대상 요약\n"
        content += "### 1.1. 사용자 입력 정보\n"
        content += f"{input_summary}\n"
        content += "### 1.2. 외부 API 조회 정보\n"
        content += f"{external_summary}\n"

        content += "## 2. LLM 종합 위험도 평가\n"
        content += f"**위험도:** {risk_assessment_result.get('risk_level', '정보 부족으로 판단 유보')}\n"
        content += f"**위험 요약 (LLM):**\n```\n{risk_assessment_result.get('summary', 'N/A')}\n```\n\n"

        content += "## 3. 세부 위험 항목 분석 결과 (LLM 기반)\n"
        identified_risks = risk_assessment_result.get('identified_risks', [])
        if identified_risks and isinstance(identified_risks, list) and len(identified_risks) > 0 and identified_risks[0].get('item') != "LLM 응답에서 위험 항목 1 파싱 필요": # 파싱 성공 시
            for i, risk in enumerate(identified_risks):
                content += f"### 3.{i+1} {risk.get('item', '항목명 없음')}\n"
                content += f"- **상세 분석:** {risk.get('details', '세부 내용 없음')}\n"
                content += f"- **판단 근거 (데이터 출처):** {risk.get('evidence_source', 'N/A')}\n"
                if risk.get('rag_reference'):
                    content += f"- **관련 참고자료 (RAG):** {risk.get('rag_reference')}\n"
                content += "\n"
        else:
            content += "LLM으로부터 구조화된 위험 항목을 추출하지 못했거나, 특이 위험 항목이 명시적으로 보고되지 않았습니다.\n"
            content += "LLM 원본 응답을 확인하십시오: \n"
            content += f"```\n{risk_assessment_result.get('raw_llm_response', '원본 LLM 응답 없음')}\n```\n\n"


        content += "## 4. LLM 최종 솔루션 제안\n"
        content += f"**최종 권고 (LLM):** {solution_advice_result.get('recommendation', '판단 유보')}\n\n"
        content += "**상세 제안 내용 (LLM 생성 원본 또는 요약):**\n"
        content += "```text\n"
        content += solution_advice_result.get('full_llm_advice_response', 'LLM 조언 없음') + "\n"
        content += "```\n"
        # TODO: solution_advice_result에서 파싱된 주의사항, 특약, 추가 요청 서류 등을 구조화하여 추가
        
        content += "\n## 5. 면책 조항 및 중요 안내\n"
        content += "- 본 보고서는 제공된 정보와 현재 시점의 LLM 및 RAG 분석을 기반으로 한 참고 자료이며, 법적 효력을 갖는 최종 판단이나 전문적인 법률/재정 자문을 대체할 수 없습니다.\n"
        content += "- 실제 계약 진행 시에는 반드시 공인중개사, 법무사 등 관련 전문가와 충분히 상담하시고, 모든 서류를 직접 꼼꼼히 확인하시기 바랍니다.\n"
        content += "- 임대인의 세금 체납 여부 등 본 분석에서 확인되지 않은 중요 정보는 반드시 별도로 확인해야 합니다.\n"
        content += "- LLM의 분석에는 의도치 않은 오류나 정보 누락이 포함될 수 있습니다.\n"
        
        content += self._load_appendix_content()
        return content

    def create(self, test_case_id: str, processed_input_data: Dict[str, Any], external_data: Dict[str, Any], risk_assessment_result: Dict[str, Any], solution_advice_result: Dict[str, Any]) -> str:
        report_content_str = self._format_report_content(test_case_id, processed_input_data, external_data, risk_assessment_result, solution_advice_result)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 파일명에 output_format 확장자 사용 (md가 기본)
        file_extension = self.output_format if self.output_format == "md" or self.output_format == "markdown" else self.output_format
        report_filename = f"{test_case_id}_report_{timestamp}.{file_extension}"
        report_filepath = os.path.join(self.reports_base_path, report_filename)

        try:
            with open(report_filepath, 'w', encoding='utf-8') as f:
                f.write(report_content_str)
            logger.info(f"Report successfully saved to: {report_filepath}")
            return report_filepath
        except Exception as e:
            logger.error(f"Error saving report to {report_filepath}: {e}", exc_info=True)
            return ""