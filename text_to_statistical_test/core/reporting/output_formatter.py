"""
Output Formatter

다양한 형식으로 결과 출력 담당
"""

import logging
from typing import Dict, Any, List, Optional
import json
import pandas as pd
from pathlib import Path
import markdown
from datetime import datetime


class OutputFormatter:
    """출력 포맷터"""
    
    def __init__(self):
        """OutputFormatter 초기화"""
        self.logger = logging.getLogger(__name__)
    
    def format_output(self, 
                     data: Dict[str, Any], 
                     format_type: str,
                     output_path: Optional[str] = None) -> str:
        """
        데이터를 지정된 형식으로 포맷팅
        
        Args:
            data: 포맷팅할 데이터
            format_type: 출력 형식 ('json', 'markdown', 'html', 'csv')
            output_path: 출력 파일 경로
            
        Returns:
            포맷팅된 문자열 또는 파일 경로
        """
        try:
            if format_type == 'json':
                return self._format_json(data, output_path)
            elif format_type == 'markdown':
                return self._format_markdown(data, output_path)
            elif format_type == 'html':
                return self._format_html(data, output_path)
            elif format_type == 'csv':
                return self._format_csv(data, output_path)
            else:
                raise ValueError(f"지원하지 않는 형식: {format_type}")
                
        except Exception as e:
            self.logger.error(f"출력 포맷팅 실패: {e}")
            return ""
    
    def _format_json(self, data: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """JSON 형식으로 포맷팅"""
        json_str = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            return output_path
        
        return json_str
    
    def _format_markdown(self, data: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Markdown 형식으로 포맷팅"""
        md_content = []
        
        # 제목
        md_content.append("# 통계 분석 보고서")
        md_content.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_content.append("")
        
        # 요약
        if 'executive_summary' in data:
            md_content.append("## 요약")
            summary = data['executive_summary']
            
            if 'key_findings' in summary:
                md_content.append("### 주요 발견사항")
                for finding in summary['key_findings']:
                    md_content.append(f"- {finding}")
                md_content.append("")
            
            if 'business_impact' in summary:
                md_content.append("### 비즈니스 영향")
                md_content.append(summary['business_impact'])
                md_content.append("")
            
            if 'recommendations' in summary:
                md_content.append("### 권장사항")
                md_content.append(summary['recommendations'])
                md_content.append("")
        
        # 분석 세부사항
        if 'analysis_details' in data:
            md_content.append("## 분석 세부사항")
            md_content.append("```json")
            md_content.append(json.dumps(data['analysis_details'], indent=2, ensure_ascii=False, default=str))
            md_content.append("```")
            md_content.append("")
        
        # 해석
        if 'interpretation' in data:
            md_content.append("## 통계적 해석")
            md_content.append(data['interpretation'])
            md_content.append("")
        
        # 부록
        if 'appendix' in data:
            md_content.append("## 부록")
            appendix = data['appendix']
            
            if 'methodology' in appendix:
                md_content.append("### 분석 방법론")
                md_content.append(appendix['methodology'])
                md_content.append("")
            
            if 'assumptions' in appendix:
                md_content.append("### 가정사항")
                for assumption in appendix['assumptions']:
                    md_content.append(f"- {assumption}")
                md_content.append("")
            
            if 'limitations' in appendix:
                md_content.append("### 한계점")
                for limitation in appendix['limitations']:
                    md_content.append(f"- {limitation}")
                md_content.append("")
        
        md_str = "\n".join(md_content)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_str)
            return output_path
        
        return md_str
    
    def _format_html(self, data: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """HTML 형식으로 포맷팅"""
        # Markdown을 먼저 생성한 후 HTML로 변환
        md_content = self._format_markdown(data)
        html_content = markdown.markdown(md_content, extensions=['tables', 'codehilite'])
        
        # HTML 템플릿 적용
        full_html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>통계 분석 보고서</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
        """
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_html)
            return output_path
        
        return full_html
    
    def _format_csv(self, data: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """CSV 형식으로 포맷팅 (분석 결과 테이블만)"""
        if 'analysis_details' not in data:
            raise ValueError("CSV 변환을 위한 분석 데이터가 없습니다")
        
        # 분석 결과를 DataFrame으로 변환
        analysis_data = data['analysis_details']
        
        # 테스트 결과가 있는 경우
        if 'test_results' in analysis_data:
            results_list = []
            for test_name, result in analysis_data['test_results'].items():
                if isinstance(result, dict):
                    row = {'test_name': test_name}
                    row.update(result)
                    results_list.append(row)
            
            if results_list:
                df = pd.DataFrame(results_list)
                
                if output_path:
                    df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    return output_path
                
                return df.to_csv(index=False)
        
        # 일반적인 딕셔너리 데이터를 CSV로 변환
        df = pd.DataFrame([analysis_data])
        
        if output_path:
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            return output_path
        
        return df.to_csv(index=False) 