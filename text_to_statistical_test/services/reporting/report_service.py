import os
import uuid
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

from config.settings import get_settings
from utils.json_utils import CustomJSONEncoder

settings = get_settings()


class ReportService:
    """
    분석 결과를 바탕으로 Markdown 형식의 보고서를 생성합니다.
    """

    def create_markdown_report(
        self,
        final_summary: Dict[str, Any],
        execution_results: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """
        Markdown 형식의 보고서를 생성하고, 파일로 저장합니다.

        Args:
            final_summary: LLM이 생성한 최종 분석 요약. 'title', 'key_findings', 'conclusion' 키를 포함.
            execution_results: 분석의 각 단계별 실행 결과 리스트.

        Returns:
            Tuple[str, str]: (저장된 보고서의 파일 경로, 생성된 Markdown 콘텐츠)
        """
        report_id = uuid.uuid4()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        title = final_summary.get('title', '분석 보고서')
        key_findings = final_summary.get('key_findings', '주요 발견 사항이 생성되지 않았습니다.')
        conclusion = final_summary.get('conclusion', '결론이 생성되지 않았습니다.')

        # Markdown 콘텐츠 생성
        md_content = []
        md_content.append(f"# {title}")
        md_content.append(f"_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}에 생성됨_")
        md_content.append("\n---\n")

        md_content.append("## 주요 발견 사항 (Key Findings)")
        md_content.append(key_findings)
        md_content.append("\n")

        md_content.append("## 결론 및 권장 사항 (Conclusion & Recommendations)")
        md_content.append(conclusion)
        md_content.append("\n---\n")

        md_content.append("## 상세 분석 과정 (Detailed Steps)")
        for i, step in enumerate(execution_results):
            step_name = step.get('step_name', f'단계 {i+1}')
            tool_name = step.get('tool_name', 'N/A')
            status = step.get('status', 'N/A')
            
            md_content.append(f"\n### {i+1}. {step_name} (`{tool_name}`)")
            md_content.append(f"**상태:** {status}")

            if status == 'SUCCESS':
                output = step.get('output', {})
                # JSON output을 예쁘게 포맷팅
                pretty_output = json.dumps(output, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
                md_content.append(f"\n```json\n{pretty_output}\n```")
            else:
                error = step.get('error', 'Unknown error')
                md_content.append(f"**오류:** `{error}`")
        
        final_md_content = "\n".join(md_content)

        # 파일로 저장
        file_name = f"report_{timestamp}_{report_id}.md"
        report_path = os.path.join(settings.paths.reports_dir, file_name)

        os.makedirs(settings.paths.reports_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_md_content)

        return report_path, final_md_content 