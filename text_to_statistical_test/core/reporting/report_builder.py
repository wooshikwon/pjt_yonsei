"""
Report Builder

LLM이 생성한 리포트 콘텐츠를 최종 파일(HTML)로 변환하고 저장합니다.
"""

import logging
from pathlib import Path
from datetime import datetime
from markdown_it import MarkdownIt

class ReportBuilder:
    """
    LLM이 생성한 마크다운 형식의 리포트 내용을 받아,
    독립적인 HTML 파일로 변환하고 저장하는 역할을 담당합니다.
    """
    
    def __init__(self, output_dir: str = "output_data/reports"):
        self.logger = logging.getLogger(__name__)
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.md = MarkdownIt()

    def _get_html_template(self, content: str) -> str:
        """리포트 내용을 감싸는 기본 HTML 템플릿"""
        css_style = """
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1, h2, h3 { color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }
            h1 { font-size: 2.5em; }
            h2 { font-size: 2em; }
            pre { background-color: #ecf0f1; padding: 15px; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; }
            code { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; background-color: #f8f8f8; padding: 2px 4px; border-radius: 3px;}
            pre > code { background-color: transparent; padding: 0; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { border: 1px solid #bdc3c7; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            img { max-width: 100%; height: auto; border-radius: 5px; margin-top: 15px; }
        </style>
        """
        return f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>자율 통계 분석 보고서</title>
            {css_style}
        </head>
        <body>
            {content}
        </body>
        </html>
        """

    def build_and_save(self, report_content: str, original_file_name: str) -> str:
        """
        마크다운 콘텐츠를 HTML로 변환하고, 고유한 파일명으로 저장합니다.

        Args:
            report_content (str): LLM이 생성한 마크다운 형식의 리포트 전체 내용.
            original_file_name (str): 리포트 파일명을 생성하는 데 사용할 원본 데이터 파일명.

        Returns:
            str: 저장된 HTML 리포트 파일의 전체 경로.
        """
        try:
            self.logger.info("Markdown 콘텐츠를 HTML로 변환합니다.")
            html_content = self.md.render(report_content)
            full_html = self._get_html_template(html_content)

            # 고유한 파일명 생성
            base_name = Path(original_file_name).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"report_{base_name}_{timestamp}.html"
            report_filepath = self.output_path / report_filename

            self.logger.info(f"HTML 리포트를 다음 경로에 저장합니다: {report_filepath}")
            with open(report_filepath, 'w', encoding='utf-8') as f:
                f.write(full_html)
            
            return str(report_filepath)

        except Exception as e:
            self.logger.error(f"리포트 파일 생성 및 저장 중 오류 발생: {e}", exc_info=True)
            # 예외를 다시 발생시켜 상위 호출자가 처리하도록 함
            raise 