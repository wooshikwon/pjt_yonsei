"""
TemplateManager: 보고서 템플릿 관리

다양한 보고서 템플릿을 관리하고 사용자 정의 템플릿을 지원하여
통계 분석 결과를 다양한 형태의 보고서로 생성할 수 있도록 합니다.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from string import Template
import re


class TemplateManager:
    """
    보고서 템플릿 관리 및 렌더링
    
    다양한 보고서 템플릿을 관리하고, 변수 치환을 통해
    사용자 정의 보고서를 생성합니다.
    """
    
    def __init__(self, template_dir: str = "reporting/templates"):
        """
        TemplateManager 초기화
        
        Args:
            template_dir: 템플릿 디렉토리 경로
        """
        self.template_dir = Path(template_dir)
        self.logger = logging.getLogger(__name__)
        
        # 템플릿 디렉토리 생성
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # 내장 템플릿 생성
        self._create_default_templates()
        
        # 템플릿 캐시
        self._template_cache = {}
    
    def _create_default_templates(self):
        """기본 템플릿들을 생성합니다"""
        
        # 기본 Markdown 보고서 템플릿
        basic_markdown_template = """# 통계 분석 보고서

**분석 일시:** $analysis_date  
**세션 ID:** $session_id  
**분석자:** $analyst_name

## 📊 분석 개요

본 보고서는 Enhanced RAG 기반 Multi-turn 통계 분석 시스템을 통해 수행된 분석 결과를 정리한 문서입니다.

### 🎯 분석 목적
$analysis_purpose

### 📋 데이터 정보
- **데이터 소스:** $data_source
- **데이터 크기:** $data_shape
- **분석 대상 변수:** $target_variables
- **분석 기간:** $analysis_period

## 🔍 수행된 분석

$analysis_summary

## 📈 주요 결과

$key_findings

### 통계적 검정 결과

$statistical_results

## 💡 결론 및 권장사항

### 주요 발견사항
$main_conclusions

### 실무적 권장사항
$recommendations

### 추가 분석 제안
$future_analysis

## 📝 분석 과정 상세

$detailed_process

## 🔗 참고 자료

- 사용된 통계 방법론: $methodology_references
- 데이터 전처리 과정: $preprocessing_details
- 분석 코드 및 결과: $code_references

---
*본 보고서는 $generation_timestamp에 자동 생성되었습니다.*
"""
        
        # 상세 HTML 보고서 템플릿
        detailed_html_template = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>통계 분석 보고서 - $session_id</title>
    <style>
        body {
            font-family: 'Noto Sans KR', Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
            margin-top: 10px;
        }
        .content {
            padding: 40px;
        }
        .section {
            margin-bottom: 40px;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .section-overview {
            background-color: #e3f2fd;
            border-left: 5px solid #2196f3;
        }
        .section-results {
            background-color: #e8f5e9;
            border-left: 5px solid #4caf50;
        }
        .section-conclusions {
            background-color: #fff3e0;
            border-left: 5px solid #ff9800;
        }
        .section h2 {
            color: #333;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #ddd;
            text-align: center;
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #2196f3;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        .highlight {
            background-color: #ffeb3b;
            padding: 2px 6px;
            border-radius: 3px;
        }
        .footer {
            background-color: #f5f5f5;
            padding: 20px 40px;
            text-align: center;
            color: #666;
            border-top: 1px solid #ddd;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .code-block {
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 통계 분석 보고서</h1>
            <div class="subtitle">Enhanced RAG 기반 Multi-turn 분석 시스템</div>
            <div class="subtitle">세션 ID: $session_id | 생성일: $analysis_date</div>
        </div>
        
        <div class="content">
            <div class="section section-overview">
                <h2>🎯 분석 개요</h2>
                <p><strong>분석 목적:</strong> $analysis_purpose</p>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">$total_analyses</div>
                        <div class="stat-label">수행된 분석 수</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">$data_rows</div>
                        <div class="stat-label">분석 데이터 행 수</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">$significant_results</div>
                        <div class="stat-label">유의한 결과</div>
                    </div>
                </div>
            </div>
            
            <div class="section section-results">
                <h2>📈 주요 분석 결과</h2>
                $detailed_results_html
            </div>
            
            <div class="section section-conclusions">
                <h2>💡 결론 및 권장사항</h2>
                $conclusions_html
            </div>
            
            <div class="section">
                <h2>🔍 분석 과정 상세</h2>
                $process_details_html
            </div>
        </div>
        
        <div class="footer">
            <p>본 보고서는 $generation_timestamp에 자동 생성되었습니다.</p>
            <p>Enhanced RAG 기반 Multi-turn 통계 분석 시스템 v1.0</p>
        </div>
    </div>
</body>
</html>"""
        
        # 요약 보고서 템플릿
        summary_template = """# 분석 요약 보고서

**세션:** $session_id  
**날짜:** $analysis_date

## 🔢 주요 지표

- **총 분석 수:** $total_analyses
- **유의한 결과:** $significant_results
- **사용된 통계 방법:** $methods_used
- **분석 소요 시간:** $total_duration

## 📊 핵심 발견사항

$key_insights

## 🎯 다음 단계

$next_steps

---
*자동 생성된 요약 보고서*
"""
        
        # 템플릿 파일 저장
        templates = {
            'basic_markdown.md': basic_markdown_template,
            'detailed_html.html': detailed_html_template,
            'summary.md': summary_template
        }
        
        for filename, content in templates.items():
            template_path = self.template_dir / filename
            if not template_path.exists():
                with open(template_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.logger.info(f"기본 템플릿 생성: {template_path}")
    
    def get_template(self, template_name: str) -> str:
        """
        템플릿 로드
        
        Args:
            template_name: 템플릿 이름
            
        Returns:
            str: 템플릿 내용
        """
        # 캐시에서 확인
        if template_name in self._template_cache:
            return self._template_cache[template_name]
        
        # 파일에서 로드
        template_path = self.template_dir / template_name
        
        if not template_path.exists():
            raise FileNotFoundError(f"템플릿을 찾을 수 없습니다: {template_name}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # 캐시에 저장
        self._template_cache[template_name] = template_content
        
        return template_content
    
    def render_template(self, template_name: str, variables: Dict[str, Any]) -> str:
        """
        템플릿 렌더링 (변수 치환)
        
        Args:
            template_name: 템플릿 이름
            variables: 치환할 변수들
            
        Returns:
            str: 렌더링된 내용
        """
        template_content = self.get_template(template_name)
        
        # 기본 변수 추가
        default_variables = {
            'generation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_date': datetime.now().strftime('%Y년 %m월 %d일'),
            'analyst_name': '시스템 분석',
            'data_source': '사용자 제공 데이터',
            'methodology_references': '통계 분석 표준 절차'
        }
        
        # 변수 병합 (사용자 변수가 우선)
        all_variables = {**default_variables, **variables}
        
        # None 값 처리
        processed_variables = {}
        for key, value in all_variables.items():
            if value is None:
                processed_variables[key] = '정보 없음'
            elif isinstance(value, (list, dict)):
                processed_variables[key] = self._format_complex_data(value)
            else:
                processed_variables[key] = str(value)
        
        # Template 클래스를 사용한 안전한 치환
        try:
            template = Template(template_content)
            rendered_content = template.safe_substitute(**processed_variables)
            
            # 치환되지 않은 변수 확인
            missing_vars = re.findall(r'\$\{?(\w+)\}?', rendered_content)
            if missing_vars:
                self.logger.warning(f"치환되지 않은 변수들: {missing_vars}")
            
            return rendered_content
            
        except Exception as e:
            self.logger.error(f"템플릿 렌더링 실패: {e}")
            raise
    
    def create_custom_template(self, template_name: str, content: str) -> str:
        """
        사용자 정의 템플릿 생성
        
        Args:
            template_name: 템플릿 이름
            content: 템플릿 내용
            
        Returns:
            str: 생성된 템플릿 파일 경로
        """
        template_path = self.template_dir / template_name
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 캐시 무효화
        if template_name in self._template_cache:
            del self._template_cache[template_name]
        
        self.logger.info(f"사용자 정의 템플릿 생성: {template_path}")
        return str(template_path)
    
    def list_templates(self) -> List[str]:
        """
        사용 가능한 템플릿 목록 반환
        
        Returns:
            List[str]: 템플릿 파일명 목록
        """
        template_files = []
        for file_path in self.template_dir.glob('*'):
            if file_path.is_file():
                template_files.append(file_path.name)
        
        return sorted(template_files)
    
    def validate_template(self, template_name: str) -> Dict[str, Any]:
        """
        템플릿 유효성 검사
        
        Args:
            template_name: 템플릿 이름
            
        Returns:
            Dict: 검사 결과
        """
        try:
            template_content = self.get_template(template_name)
            
            # 변수 추출
            variables = re.findall(r'\$\{?(\w+)\}?', template_content)
            unique_variables = list(set(variables))
            
            # 기본 구조 검사
            has_title = bool(re.search(r'#.*title|<title|제목', template_content, re.IGNORECASE))
            has_content_section = len(template_content) > 100
            
            result = {
                'valid': True,
                'template_name': template_name,
                'variables_found': unique_variables,
                'variable_count': len(unique_variables),
                'has_title': has_title,
                'has_content_section': has_content_section,
                'template_length': len(template_content),
                'issues': []
            }
            
            # 잠재적 이슈 확인
            if not has_title:
                result['issues'].append('제목 섹션이 없을 수 있습니다.')
            
            if len(unique_variables) == 0:
                result['issues'].append('동적 변수가 없습니다.')
            
            if len(template_content) < 50:
                result['issues'].append('템플릿이 너무 짧습니다.')
            
            return result
            
        except Exception as e:
            return {
                'valid': False,
                'template_name': template_name,
                'error': str(e),
                'issues': [f'템플릿 로드 실패: {str(e)}']
            }
    
    def get_template_variables(self, template_name: str) -> List[str]:
        """
        템플릿에서 사용되는 변수 목록 추출
        
        Args:
            template_name: 템플릿 이름
            
        Returns:
            List[str]: 변수 목록
        """
        template_content = self.get_template(template_name)
        variables = re.findall(r'\$\{?(\w+)\}?', template_content)
        return list(set(variables))
    
    def _format_complex_data(self, data: Any) -> str:
        """복잡한 데이터 타입을 문자열로 포맷팅"""
        if isinstance(data, dict):
            if len(data) <= 5:
                # 작은 딕셔너리는 키-값 나열
                items = [f"{k}: {v}" for k, v in data.items()]
                return ', '.join(items)
            else:
                # 큰 딕셔너리는 요약
                return f"딕셔너리 ({len(data)}개 항목)"
        
        elif isinstance(data, list):
            if len(data) <= 3:
                # 작은 리스트는 전체 표시
                return ', '.join(map(str, data))
            else:
                # 큰 리스트는 앞 3개만 표시
                preview = ', '.join(map(str, data[:3]))
                return f"{preview} ... (총 {len(data)}개)"
        
        else:
            return str(data)
    
    def export_template_package(self, template_names: List[str], 
                              output_path: str) -> str:
        """
        선택된 템플릿들을 패키지로 내보내기
        
        Args:
            template_names: 내보낼 템플릿 이름 목록
            output_path: 출력 ZIP 파일 경로
            
        Returns:
            str: 생성된 패키지 파일 경로
        """
        import zipfile
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for template_name in template_names:
                template_path = self.template_dir / template_name
                if template_path.exists():
                    zipf.write(template_path, template_name)
        
        self.logger.info(f"템플릿 패키지 생성: {output_path}")
        return output_path
    
    def import_template_package(self, package_path: str) -> List[str]:
        """
        템플릿 패키지 가져오기
        
        Args:
            package_path: 템플릿 패키지 파일 경로
            
        Returns:
            List[str]: 가져온 템플릿 목록
        """
        import zipfile
        
        imported_templates = []
        
        with zipfile.ZipFile(package_path, 'r') as zipf:
            for filename in zipf.namelist():
                if filename.endswith(('.md', '.html', '.txt')):
                    with zipf.open(filename) as template_file:
                        content = template_file.read().decode('utf-8')
                        self.create_custom_template(filename, content)
                        imported_templates.append(filename)
        
        self.logger.info(f"템플릿 패키지 가져오기 완료: {len(imported_templates)}개")
        return imported_templates 