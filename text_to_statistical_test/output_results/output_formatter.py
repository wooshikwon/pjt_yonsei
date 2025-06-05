"""
OutputFormatter: 분석 결과의 다양한 출력 형식 지원

통계 분석 결과를 사용자의 요구에 맞는 다양한 형식으로 
포맷팅하고 변환하는 기능을 제공합니다.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import logging
from io import StringIO
import base64


class OutputFormatter:
    """
    분석 결과의 다양한 출력 형식 지원
    
    통계 분석 결과를 JSON, CSV, Excel, HTML, Markdown 등
    다양한 형식으로 변환하여 제공합니다.
    """
    
    def __init__(self):
        """OutputFormatter 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 지원하는 출력 형식
        self.supported_formats = [
            'json', 'csv', 'excel', 'html', 'markdown', 'txt', 'xml'
        ]
    
    def format_statistical_result(self, result_data: Dict[str, Any], 
                                format_type: str = 'json',
                                include_metadata: bool = True) -> str:
        """
        통계 분석 결과를 지정된 형식으로 포맷팅
        
        Args:
            result_data: 통계 분석 결과 데이터
            format_type: 출력 형식 (json, csv, excel, html, markdown, txt)
            include_metadata: 메타데이터 포함 여부
            
        Returns:
            str: 포맷팅된 결과 문자열
        """
        if format_type not in self.supported_formats:
            raise ValueError(f"지원되지 않는 형식: {format_type}")
        
        # 메타데이터 추가
        if include_metadata:
            formatted_data = {
                'metadata': {
                    'formatted_at': datetime.now().isoformat(),
                    'format_type': format_type,
                    'version': '1.0'
                },
                'results': result_data
            }
        else:
            formatted_data = result_data
        
        # 형식별 포맷팅
        if format_type == 'json':
            return self._format_as_json(formatted_data)
        elif format_type == 'csv':
            return self._format_as_csv(formatted_data)
        elif format_type == 'html':
            return self._format_as_html(formatted_data)
        elif format_type == 'markdown':
            return self._format_as_markdown(formatted_data)
        elif format_type == 'txt':
            return self._format_as_text(formatted_data)
        elif format_type == 'xml':
            return self._format_as_xml(formatted_data)
        else:
            return self._format_as_json(formatted_data)
    
    def format_session_summary(self, session_data: Dict[str, Any],
                             format_type: str = 'markdown') -> str:
        """
        세션 요약 정보를 지정된 형식으로 포맷팅
        
        Args:
            session_data: 세션 데이터
            format_type: 출력 형식
            
        Returns:
            str: 포맷팅된 세션 요약
        """
        if format_type == 'markdown':
            return self._format_session_as_markdown(session_data)
        elif format_type == 'html':
            return self._format_session_as_html(session_data)
        elif format_type == 'json':
            return self._format_as_json(session_data)
        else:
            return self._format_session_as_text(session_data)
    
    def create_analysis_dashboard(self, analysis_results: List[Dict],
                                session_metadata: Dict) -> str:
        """
        분석 결과들을 종합한 대시보드 HTML 생성
        
        Args:
            analysis_results: 분석 결과 목록
            session_metadata: 세션 메타데이터
            
        Returns:
            str: HTML 대시보드 문자열
        """
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>통계 분석 대시보드</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .session-info {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .analysis-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            background-color: #fafafa;
        }}
        .analysis-title {{
            color: #2c3e50;
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .result-item {{
            margin: 8px 0;
            padding: 5px 10px;
            background-color: white;
            border-left: 4px solid #3498db;
        }}
        .significant {{
            border-left-color: #e74c3c;
            background-color: #ffeaa7;
        }}
        .not-significant {{
            border-left-color: #27ae60;
            background-color: #d5f5d5;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .stat-box {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 통계 분석 대시보드</h1>
            <p>세션 ID: {session_id}</p>
            <p>생성일시: {created_at}</p>
        </div>
        
        <div class="session-info">
            <h2>📋 세션 정보</h2>
            <p><strong>총 분석 수:</strong> {total_analyses}</p>
            <p><strong>세션 상태:</strong> {session_status}</p>
            <p><strong>마지막 업데이트:</strong> {last_updated}</p>
        </div>
        
        <div class="summary-stats">
            <div class="stat-box">
                <div class="stat-number">{total_analyses}</div>
                <div class="stat-label">총 분석 수</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{significant_count}</div>
                <div class="stat-label">유의한 결과</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{unique_tests}</div>
                <div class="stat-label">사용된 검정 방법</div>
            </div>
        </div>
        
        <h2>🔬 분석 결과 상세</h2>
        {analysis_cards}
    </div>
</body>
</html>
        """
        
        # 분석 결과 카드 생성
        analysis_cards = []
        significant_count = 0
        test_types = set()
        
        for i, result in enumerate(analysis_results, 1):
            # 결과 메타데이터 추출
            metadata = result.get('metadata', {})
            stats = result.get('statistical_results', {})
            summary = result.get('summary', {})
            
            # 유의성 판단
            is_significant = summary.get('significant', False)
            if is_significant:
                significant_count += 1
            
            # 검정 유형 수집
            test_types.add(metadata.get('analysis_type', 'unknown'))
            
            # 분석 카드 HTML 생성
            significance_class = 'significant' if is_significant else 'not-significant'
            significance_text = '유의함 (p < 0.05)' if is_significant else '유의하지 않음 (p ≥ 0.05)'
            
            card_html = f"""
            <div class="analysis-card">
                <div class="analysis-title">분석 #{i}: {metadata.get('analysis_type', 'Unknown')}</div>
                <div class="result-item">
                    <strong>분석 시간:</strong> {metadata.get('timestamp', 'Unknown')}
                </div>
                <div class="result-item {significance_class}">
                    <strong>유의성:</strong> {significance_text}
                </div>
                {self._format_statistical_details(stats)}
            </div>
            """
            analysis_cards.append(card_html)
        
        # 템플릿 변수 치환
        return html_template.format(
            session_id=session_metadata.get('session_id', 'Unknown'),
            created_at=session_metadata.get('created_at', 'Unknown'),
            total_analyses=len(analysis_results),
            session_status=session_metadata.get('status', 'Unknown'),
            last_updated=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            significant_count=significant_count,
            unique_tests=len(test_types),
            analysis_cards=''.join(analysis_cards)
        )
    
    def _format_as_json(self, data: Dict) -> str:
        """JSON 형식으로 포맷팅"""
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)
    
    def _format_as_csv(self, data: Dict) -> str:
        """CSV 형식으로 포맷팅"""
        output = StringIO()
        
        # 결과 데이터가 리스트인 경우
        if 'results' in data and isinstance(data['results'], list):
            if data['results']:
                writer = csv.DictWriter(output, fieldnames=data['results'][0].keys())
                writer.writeheader()
                writer.writerows(data['results'])
        else:
            # 단일 결과인 경우 키-값 쌍으로 변환
            results = data.get('results', data)
            flattened = self._flatten_dict(results)
            
            writer = csv.writer(output)
            writer.writerow(['Variable', 'Value'])
            for key, value in flattened.items():
                writer.writerow([key, value])
        
        return output.getvalue()
    
    def _format_as_html(self, data: Dict) -> str:
        """HTML 형식으로 포맷팅"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>통계 분석 결과</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .metadata { background-color: #e8f4fd; padding: 15px; border-radius: 5px; }
    </style>
</head>
<body>
"""
        
        # 메타데이터 섹션
        if 'metadata' in data:
            html += "<div class='metadata'><h3>메타데이터</h3>"
            for key, value in data['metadata'].items():
                html += f"<p><strong>{key}:</strong> {value}</p>"
            html += "</div>"
        
        # 결과 테이블
        html += "<h3>분석 결과</h3><table>"
        results = data.get('results', data)
        flattened = self._flatten_dict(results)
        
        html += "<tr><th>항목</th><th>값</th></tr>"
        for key, value in flattened.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        
        html += "</table></body></html>"
        return html
    
    def _format_as_markdown(self, data: Dict) -> str:
        """Markdown 형식으로 포맷팅"""
        md_content = []
        
        # 제목
        md_content.append("# 통계 분석 결과")
        md_content.append("")
        
        # 메타데이터
        if 'metadata' in data:
            md_content.append("## 메타데이터")
            for key, value in data['metadata'].items():
                md_content.append(f"- **{key}:** {value}")
            md_content.append("")
        
        # 결과
        md_content.append("## 분석 결과")
        results = data.get('results', data)
        flattened = self._flatten_dict(results)
        
        md_content.append("| 항목 | 값 |")
        md_content.append("| --- | --- |")
        for key, value in flattened.items():
            md_content.append(f"| {key} | {value} |")
        
        return "\n".join(md_content)
    
    def _format_as_text(self, data: Dict) -> str:
        """일반 텍스트 형식으로 포맷팅"""
        text_content = []
        
        text_content.append("=" * 50)
        text_content.append("통계 분석 결과")
        text_content.append("=" * 50)
        text_content.append("")
        
        # 메타데이터
        if 'metadata' in data:
            text_content.append("메타데이터:")
            text_content.append("-" * 20)
            for key, value in data['metadata'].items():
                text_content.append(f"{key}: {value}")
            text_content.append("")
        
        # 결과
        text_content.append("분석 결과:")
        text_content.append("-" * 20)
        results = data.get('results', data)
        flattened = self._flatten_dict(results)
        
        for key, value in flattened.items():
            text_content.append(f"{key}: {value}")
        
        return "\n".join(text_content)
    
    def _format_as_xml(self, data: Dict) -> str:
        """XML 형식으로 포맷팅"""
        def dict_to_xml(d, root_name="root"):
            xml_str = f"<{root_name}>"
            for key, value in d.items():
                if isinstance(value, dict):
                    xml_str += dict_to_xml(value, key)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            xml_str += dict_to_xml(item, key)
                        else:
                            xml_str += f"<{key}>{item}</{key}>"
                else:
                    xml_str += f"<{key}>{value}</{key}>"
            xml_str += f"</{root_name}>"
            return xml_str
        
        xml_header = '<?xml version="1.0" encoding="UTF-8"?>'
        return xml_header + dict_to_xml(data, "statistical_analysis")
    
    def _format_session_as_markdown(self, session_data: Dict) -> str:
        """세션 데이터를 Markdown으로 포맷팅"""
        md_content = []
        
        md_content.append("# 세션 요약 보고서")
        md_content.append("")
        md_content.append(f"**세션 ID:** {session_data.get('session_id', 'Unknown')}")
        md_content.append(f"**생성일시:** {session_data.get('created_at', 'Unknown')}")
        md_content.append(f"**상태:** {session_data.get('status', 'Unknown')}")
        md_content.append("")
        
        # 파일 수 정보
        if 'file_counts' in session_data:
            md_content.append("## 📂 생성된 파일 수")
            for category, count in session_data['file_counts'].items():
                md_content.append(f"- **{category}:** {count}개")
            md_content.append("")
        
        return "\n".join(md_content)
    
    def _format_session_as_html(self, session_data: Dict) -> str:
        """세션 데이터를 HTML로 포맷팅"""
        return f"""
        <div class="session-summary">
            <h2>세션 요약</h2>
            <p><strong>세션 ID:</strong> {session_data.get('session_id', 'Unknown')}</p>
            <p><strong>생성일시:</strong> {session_data.get('created_at', 'Unknown')}</p>
            <p><strong>상태:</strong> {session_data.get('status', 'Unknown')}</p>
        </div>
        """
    
    def _format_session_as_text(self, session_data: Dict) -> str:
        """세션 데이터를 텍스트로 포맷팅"""
        return f"""
세션 요약
========
세션 ID: {session_data.get('session_id', 'Unknown')}
생성일시: {session_data.get('created_at', 'Unknown')}
상태: {session_data.get('status', 'Unknown')}
        """.strip()
    
    def _format_statistical_details(self, stats: Dict) -> str:
        """통계 상세 정보를 HTML로 포맷팅"""
        if not stats:
            return '<div class="result-item">통계 정보 없음</div>'
        
        details_html = []
        
        # 주요 통계량들
        important_stats = ['test_statistic', 'p_value', 'confidence_interval', 
                          'effect_size', 'degrees_of_freedom']
        
        for stat_name in important_stats:
            if stat_name in stats:
                value = stats[stat_name]
                formatted_name = stat_name.replace('_', ' ').title()
                details_html.append(
                    f'<div class="result-item"><strong>{formatted_name}:</strong> {value}</div>'
                )
        
        return ''.join(details_html)
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """중첩된 딕셔너리를 평면화"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, ', '.join(map(str, v))))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def export_to_excel(self, data_dict: Dict[str, pd.DataFrame], 
                       output_path: str) -> str:
        """
        여러 데이터프레임을 Excel 파일로 내보내기
        
        Args:
            data_dict: 시트명: 데이터프레임 딕셔너리
            output_path: 출력 파일 경로
            
        Returns:
            str: 저장된 파일 경로
        """
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for sheet_name, df in data_dict.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            self.logger.info(f"Excel 파일 생성 완료: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Excel 파일 생성 실패: {e}")
            raise 