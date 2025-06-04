"""
PromptCrafter: 프롬프트 엔지니어링 및 조립 전문가

템플릿 파일을 기반으로 현재 분석 컨텍스트와 노드 정보를 조합하여
LLM에 전달할 최종 프롬프트를 동적으로 생성합니다.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import re
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound


class PromptCrafter:
    """
    프롬프트 엔지니어링 및 조립 전문가
    
    Jinja2 템플릿 시스템을 사용하여 동적으로 프롬프트를 생성하고,
    워크플로우 노드별 맞춤형 프롬프트를 제공합니다.
    """
    
    def __init__(self, prompt_template_dir: str, workflow_data: Dict = None):
        """
        PromptCrafter 초기화
        
        Args:
            prompt_template_dir: 프롬프트 템플릿 디렉토리 경로
            workflow_data: 워크플로우 데이터 (선택적)
        """
        self.template_dir = Path(prompt_template_dir)
        self.workflow_data = workflow_data or {}
        self.logger = logging.getLogger(__name__)
        
        # Jinja2 환경 설정
        if self.template_dir.exists():
            self._jinja_env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                trim_blocks=True,
                lstrip_blocks=True
            )
        else:
            self.logger.warning(f"템플릿 디렉토리가 존재하지 않습니다: {self.template_dir}")
            self._jinja_env = Environment()
        
        # 기본 템플릿 캐시
        self._template_cache: Dict[str, Template] = {}
    
    def get_prompt_for_node(self, node_id: str, dynamic_data: Dict = None, 
                           agent_context_summary: str = None) -> str:
        """
        특정 노드에 대한 프롬프트를 생성합니다.
        
        Args:
            node_id: 워크플로우 노드 ID
            dynamic_data: 현재 노드 처리를 위한 특정 데이터
            agent_context_summary: 이전 대화/작업 이력 요약
            
        Returns:
            str: 생성된 프롬프트
        """
        # 템플릿 파일명 결정
        template_name = self._get_template_name_for_node(node_id)
        
        # 컨텍스트 데이터 구성
        context_data = self._build_context_data(
            node_id, dynamic_data, agent_context_summary
        )
        
        # 프롬프트 렌더링
        try:
            return self.render_prompt(template_name, context_data)
        except TemplateNotFound:
            # 템플릿이 없으면 기본 프롬프트 생성
            return self._generate_default_prompt(node_id, context_data)
    
    def render_prompt(self, template_name: str, context_data: Dict) -> str:
        """
        템플릿과 컨텍스트 데이터를 결합하여 최종 프롬프트를 생성합니다.
        
        Args:
            template_name: 템플릿 파일명
            context_data: 템플릿에 전달할 데이터
            
        Returns:
            str: 렌더링된 프롬프트
        """
        try:
            template = self._load_template(template_name)
            return template.render(**context_data)
        except Exception as e:
            self.logger.error(f"프롬프트 렌더링 실패 ({template_name}): {e}")
            # 폴백: 기본 프롬프트
            return self._create_fallback_prompt(context_data)
    
    def _load_template(self, template_name: str) -> Template:
        """지정된 이름의 템플릿 파일을 로드합니다."""
        if template_name in self._template_cache:
            return self._template_cache[template_name]
        
        try:
            template = self._jinja_env.get_template(template_name)
            self._template_cache[template_name] = template
            return template
        except TemplateNotFound:
            self.logger.warning(f"템플릿 파일을 찾을 수 없습니다: {template_name}")
            raise
    
    def _get_template_name_for_node(self, node_id: str) -> str:
        """노드 ID에 대응하는 템플릿 파일명을 결정합니다."""
        # 노드 ID 패턴 분석
        if node_id == "start":
            return "common/start_analysis.md"
        elif node_id.startswith("1"):
            # 1단계: 사용자 이해
            return f"stage_1_user_understanding/{node_id.replace('-', '_')}.md"
        elif node_id.startswith("2"):
            # 2단계: 데이터 분석
            return f"stage_2_data_analysis/{node_id.replace('-', '_')}.md"
        elif node_id.startswith("3"):
            # 3단계: 데이터 전처리
            return f"stage_3_preprocessing/{node_id.replace('-', '_')}.md"
        elif node_id.startswith("4"):
            # 4단계: 가설 설정
            return f"stage_4_hypothesis/{node_id.replace('-', '_')}.md"
        elif node_id.startswith("5"):
            # 5단계: 통계 검정 선택
            return f"stage_5_test_selection/{node_id.replace('-', '_')}.md"
        elif node_id.startswith("6"):
            # 6단계: 검정 수행
            return f"stage_6_test_execution/{node_id.replace('-', '_')}.md"
        elif node_id.startswith("7"):
            # 7단계: 결과 해석
            return f"stage_7_interpretation/{node_id.replace('-', '_')}.md"
        elif node_id in ["8", "9"]:
            # 종료 단계
            return f"common/termination_{node_id}.md"
        else:
            return f"common/default_node.md"
    
    def _build_context_data(self, node_id: str, dynamic_data: Dict = None, 
                           agent_context_summary: str = None) -> Dict:
        """템플릿에 전달할 컨텍스트 데이터를 구성합니다."""
        context = {
            'node_id': node_id,
            'timestamp': self._get_current_timestamp(),
            'workflow_data': self.workflow_data,
            'agent_context_summary': agent_context_summary or ""
        }
        
        # 동적 데이터 추가
        if dynamic_data:
            context.update(dynamic_data)
        
        # 노드별 특화 데이터 추가
        context.update(self._get_node_specific_context(node_id))
        
        return context
    
    def _get_node_specific_context(self, node_id: str) -> Dict:
        """노드별 특화 컨텍스트를 생성합니다."""
        if node_id.startswith("1"):
            return {
                'stage_name': '사용자 요청 이해',
                'stage_description': '사용자의 분석 요청을 파악하고 분석 목표를 설정합니다.',
                'key_tasks': ['요청 분석', '목표 설정', '사용자 확인']
            }
        elif node_id.startswith("2"):
            return {
                'stage_name': '데이터 탐색 및 분석',
                'stage_description': '데이터의 구조와 특성을 파악합니다.',
                'key_tasks': ['데이터 로딩', '기본 통계', '변수 타입 확인']
            }
        elif node_id.startswith("3"):
            return {
                'stage_name': '데이터 전처리',
                'stage_description': '분석에 적합하도록 데이터를 준비합니다.',
                'key_tasks': ['결측치 처리', '이상치 처리', '데이터 변환']
            }
        elif node_id.startswith("4"):
            return {
                'stage_name': '가설 설정',
                'stage_description': '통계적 가설을 설정합니다.',
                'key_tasks': ['귀무가설', '대립가설', '유의수준']
            }
        elif node_id.startswith("5"):
            return {
                'stage_name': '통계 검정 선택',
                'stage_description': '적절한 통계 검정 방법을 선택합니다.',
                'key_tasks': ['전제조건 확인', '검정 방법 선택', '파라미터 설정']
            }
        elif node_id.startswith("6"):
            return {
                'stage_name': '검정 수행',
                'stage_description': '선택된 통계 검정을 실행합니다.',
                'key_tasks': ['코드 실행', '결과 계산', '수치 확인']
            }
        elif node_id.startswith("7"):
            return {
                'stage_name': '결과 해석',
                'stage_description': '통계 검정 결과를 해석합니다.',
                'key_tasks': ['유의성 판단', '효과크기', '실무적 의미']
            }
        else:
            return {
                'stage_name': '일반 처리',
                'stage_description': '워크플로우 처리를 진행합니다.',
                'key_tasks': []
            }
    
    def _generate_default_prompt(self, node_id: str, context_data: Dict) -> str:
        """기본 프롬프트를 생성합니다."""
        stage_info = context_data.get('stage_name', '알 수 없는 단계')
        description = context_data.get('stage_description', '')
        
        prompt = f"""
# 통계 분석 워크플로우: {stage_info}

## 현재 단계
**노드 ID**: {node_id}
**설명**: {description}

## 컨텍스트
{context_data.get('agent_context_summary', '이전 컨텍스트 없음')}

## 요청사항
다음 분석 단계를 수행해주세요:
"""
        
        # 주요 작업 목록 추가
        key_tasks = context_data.get('key_tasks', [])
        if key_tasks:
            prompt += f"\n### 주요 작업\n"
            for task in key_tasks:
                prompt += f"- {task}\n"
        
        prompt += f"""
## 출력 형식
분석 결과를 명확하고 구조화된 형태로 제공해주세요.
필요한 경우 추가 질문이나 확인사항을 포함해주세요.
"""
        
        return prompt.strip()
    
    def _create_fallback_prompt(self, context_data: Dict) -> str:
        """폴백 프롬프트를 생성합니다."""
        return f"""
# 통계 분석 지원

## 현재 상황
노드: {context_data.get('node_id', 'unknown')}

## 컨텍스트
{context_data.get('agent_context_summary', '컨텍스트 정보 없음')}

## 요청
현재 단계에서 필요한 분석을 수행해주세요.
명확하고 단계별로 설명해주세요.
"""
    
    def _get_current_timestamp(self) -> str:
        """현재 시간을 문자열로 반환합니다."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def add_custom_template(self, template_name: str, template_content: str):
        """커스텀 템플릿을 추가합니다."""
        try:
            template = self._jinja_env.from_string(template_content)
            self._template_cache[template_name] = template
            self.logger.info(f"커스텀 템플릿 추가: {template_name}")
        except Exception as e:
            self.logger.error(f"커스텀 템플릿 추가 실패: {e}")
    
    def list_available_templates(self) -> list:
        """사용 가능한 템플릿 목록을 반환합니다."""
        if not self.template_dir.exists():
            return []
        
        templates = []
        for template_file in self.template_dir.rglob("*.md"):
            relative_path = template_file.relative_to(self.template_dir)
            templates.append(str(relative_path))
        
        return sorted(templates)
    
    def validate_template(self, template_name: str) -> bool:
        """템플릿 유효성을 검사합니다."""
        try:
            self._load_template(template_name)
            return True
        except Exception as e:
            self.logger.warning(f"템플릿 유효성 검사 실패 ({template_name}): {e}")
            return False 