"""
Prompt Engine

동적 프롬프트 생성 및 관리
- 템플릿 기반 프롬프트 생성
- 컨텍스트 주입
- 프롬프트 최적화
- 다국어 지원
"""

import logging
import json
import re
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import yaml

from utils.error_handler import ErrorHandler, PromptException
from utils.global_cache import get_global_cache

logger = logging.getLogger(__name__)

@dataclass
class PromptTemplate:
    """프롬프트 템플릿 데이터 클래스"""
    name: str
    template: str
    variables: List[str]
    description: str
    category: str
    language: str = "ko"
    metadata: Dict[str, Any] = None

@dataclass
class PromptContext:
    """프롬프트 컨텍스트 데이터 클래스"""
    user_request: Optional[str] = None
    data_info: Optional[Dict[str, Any]] = None
    analysis_type: Optional[str] = None
    previous_results: Optional[Dict[str, Any]] = None
    domain_knowledge: Optional[List[str]] = None
    constraints: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None

class PromptEngine:
    """프롬프트 엔진 메인 클래스"""
    
    def __init__(self, 
                 templates_dir: Optional[str] = None,
                 default_language: str = "ko",
                 cache_enabled: bool = True):
        """
        프롬프트 엔진 초기화
        
        Args:
            templates_dir: 템플릿 디렉토리 경로
            default_language: 기본 언어
            cache_enabled: 캐시 사용 여부
        """
        self.templates_dir = Path(templates_dir) if templates_dir else Path("prompts")
        self.default_language = default_language
        self.cache_enabled = cache_enabled
        
        if cache_enabled:
            self.cache = get_global_cache()
        
        self.error_handler = ErrorHandler()
        
        # 템플릿 저장소
        self.templates: Dict[str, PromptTemplate] = {}
        
        # 기본 템플릿 로드
        self._load_default_templates()
        
        # 외부 템플릿 로드 (있는 경우)
        if self.templates_dir.exists():
            self._load_external_templates()
        
        logger.info(f"프롬프트 엔진 초기화 완료 - 템플릿 수: {len(self.templates)}")
    
    def _load_default_templates(self):
        """기본 템플릿 로드"""
        default_templates = {
            "data_analysis_request": PromptTemplate(
                name="data_analysis_request",
                template="""
데이터 분석 요청을 분석해주세요.

사용자 요청: {user_request}

데이터 정보:
{data_info}

다음 항목들을 분석해주세요:
1. 분석 목적과 목표
2. 적절한 통계 분석 방법
3. 데이터 전처리 필요사항
4. 예상되는 결과 형태
5. 주의사항 및 제약사항

분석 결과를 JSON 형태로 제공해주세요.
""",
                variables=["user_request", "data_info"],
                description="사용자의 데이터 분석 요청을 분석하는 템플릿",
                category="analysis"
            ),
            
            "statistical_test_proposal": PromptTemplate(
                name="statistical_test_proposal",
                template="""
다음 데이터와 분석 목적에 적합한 통계 검정을 제안해주세요.

데이터 정보:
- 변수 유형: {variable_types}
- 샘플 크기: {sample_size}
- 분포 특성: {distribution_info}

분석 목적: {analysis_purpose}

기존 도메인 지식:
{domain_knowledge}

다음을 포함하여 제안해주세요:
1. 추천 통계 검정 방법 (우선순위별 3개)
2. 각 방법의 가정사항과 적용 조건
3. 예상되는 결과 해석 방법
4. 대안적 접근 방법
5. 시각화 제안

JSON 형태로 응답해주세요.
""",
                variables=["variable_types", "sample_size", "distribution_info", "analysis_purpose", "domain_knowledge"],
                description="통계 검정 방법을 제안하는 템플릿",
                category="statistics"
            ),
            
            "code_generation": PromptTemplate(
                name="code_generation",
                template="""
다음 분석을 수행하는 Python 코드를 생성해주세요.

분석 사양:
{analysis_spec}

데이터 정보:
{data_info}

요구사항:
- pandas, numpy, scipy, matplotlib, seaborn 사용
- 코드에 상세한 주석 포함
- 에러 처리 포함
- 결과 시각화 포함
- 결과 해석을 위한 통계량 계산

제약사항:
{constraints}

완전히 실행 가능한 코드를 제공해주세요.
""",
                variables=["analysis_spec", "data_info", "constraints"],
                description="통계 분석 코드를 생성하는 템플릿",
                category="code"
            ),
            
            "result_interpretation": PromptTemplate(
                name="result_interpretation",
                template="""
다음 통계 분석 결과를 해석해주세요.

분석 방법: {analysis_method}
결과 데이터:
{results}

데이터 컨텍스트:
{data_context}

도메인 지식:
{domain_knowledge}

다음을 포함하여 해석해주세요:
1. 주요 발견사항
2. 통계적 유의성 해석
3. 실무적 의미
4. 제한사항 및 주의사항
5. 추가 분석 제안

일반인도 이해할 수 있도록 쉽게 설명해주세요.
""",
                variables=["analysis_method", "results", "data_context", "domain_knowledge"],
                description="통계 분석 결과를 해석하는 템플릿",
                category="interpretation"
            ),
            
            "report_generation": PromptTemplate(
                name="report_generation",
                template="""
다음 분석 결과를 바탕으로 종합 보고서를 작성해주세요.

분석 개요:
{analysis_overview}

주요 결과:
{main_results}

시각화 정보:
{visualization_info}

사용자 선호도:
{user_preferences}

보고서 구성:
1. 요약 (Executive Summary)
2. 분석 목적 및 방법
3. 주요 발견사항
4. 상세 결과
5. 결론 및 권장사항
6. 부록 (기술적 세부사항)

전문적이면서도 이해하기 쉬운 보고서를 작성해주세요.
""",
                variables=["analysis_overview", "main_results", "visualization_info", "user_preferences"],
                description="종합 분석 보고서를 생성하는 템플릿",
                category="reporting"
            )
        }
        
        self.templates.update(default_templates)
    
    def _load_external_templates(self):
        """외부 템플릿 파일 로드"""
        try:
            for template_file in self.templates_dir.glob("*.yaml"):
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = yaml.safe_load(f)
                    
                template = PromptTemplate(**template_data)
                self.templates[template.name] = template
                
            logger.info(f"외부 템플릿 {len(list(self.templates_dir.glob('*.yaml')))}개 로드됨")
            
        except Exception as e:
            logger.warning(f"외부 템플릿 로드 실패: {str(e)}")
    
    def generate_prompt(self, 
                       template_name: str,
                       context: Union[PromptContext, Dict[str, Any]],
                       **kwargs) -> str:
        """
        프롬프트 생성
        
        Args:
            template_name: 템플릿 이름
            context: 프롬프트 컨텍스트
            **kwargs: 추가 변수
            
        Returns:
            str: 생성된 프롬프트
        """
        try:
            # 템플릿 조회
            if template_name not in self.templates:
                raise PromptException(f"템플릿을 찾을 수 없습니다: {template_name}")
            
            template = self.templates[template_name]
            
            # 컨텍스트 처리
            if isinstance(context, PromptContext):
                variables = self._extract_variables_from_context(context)
            else:
                variables = context
            
            # 추가 변수 병합
            variables.update(kwargs)
            
            # 프롬프트 생성
            prompt = self._render_template(template, variables)
            
            # 후처리
            prompt = self._post_process_prompt(prompt)
            
            logger.debug(f"프롬프트 생성 완료: {template_name}")
            return prompt
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'template': template_name})
            raise PromptException(f"프롬프트 생성 실패: {error_info['message']}")
    
    def _extract_variables_from_context(self, context: PromptContext) -> Dict[str, Any]:
        """컨텍스트에서 변수 추출"""
        variables = {}
        
        if context.user_request:
            variables['user_request'] = context.user_request
        
        if context.data_info:
            variables['data_info'] = self._format_data_info(context.data_info)
            variables.update(context.data_info)
        
        if context.analysis_type:
            variables['analysis_type'] = context.analysis_type
            variables['analysis_purpose'] = context.analysis_type
        
        if context.previous_results:
            variables['previous_results'] = json.dumps(context.previous_results, indent=2, ensure_ascii=False)
            variables['results'] = variables['previous_results']
        
        if context.domain_knowledge:
            variables['domain_knowledge'] = '\n'.join(f"- {item}" for item in context.domain_knowledge)
        
        if context.constraints:
            variables['constraints'] = self._format_constraints(context.constraints)
        
        if context.preferences:
            variables['user_preferences'] = json.dumps(context.preferences, indent=2, ensure_ascii=False)
        
        return variables
    
    def _format_data_info(self, data_info: Dict[str, Any]) -> str:
        """데이터 정보 포맷팅"""
        formatted = []
        
        if 'columns' in data_info:
            formatted.append(f"컬럼 수: {len(data_info['columns'])}")
            formatted.append(f"컬럼 목록: {', '.join(data_info['columns'])}")
        
        if 'shape' in data_info:
            formatted.append(f"데이터 크기: {data_info['shape'][0]}행 x {data_info['shape'][1]}열")
        
        if 'dtypes' in data_info:
            formatted.append("데이터 타입:")
            for col, dtype in data_info['dtypes'].items():
                formatted.append(f"  - {col}: {dtype}")
        
        if 'missing_values' in data_info:
            formatted.append("결측값:")
            for col, count in data_info['missing_values'].items():
                if count > 0:
                    formatted.append(f"  - {col}: {count}개")
        
        return '\n'.join(formatted)
    
    def _format_constraints(self, constraints: Dict[str, Any]) -> str:
        """제약사항 포맷팅"""
        formatted = []
        
        for key, value in constraints.items():
            if isinstance(value, list):
                formatted.append(f"- {key}: {', '.join(map(str, value))}")
            else:
                formatted.append(f"- {key}: {value}")
        
        return '\n'.join(formatted)
    
    def _render_template(self, template: PromptTemplate, variables: Dict[str, Any]) -> str:
        """템플릿 렌더링"""
        prompt = template.template
        
        # 변수 치환
        for var_name in template.variables:
            placeholder = f"{{{var_name}}}"
            if var_name in variables:
                value = variables[var_name]
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, indent=2, ensure_ascii=False)
                prompt = prompt.replace(placeholder, str(value))
            else:
                # 기본값 또는 빈 문자열로 치환
                prompt = prompt.replace(placeholder, f"[{var_name} 정보 없음]")
        
        return prompt
    
    def _post_process_prompt(self, prompt: str) -> str:
        """프롬프트 후처리"""
        # 여러 줄 공백 제거
        prompt = re.sub(r'\n\s*\n\s*\n', '\n\n', prompt)
        
        # 앞뒤 공백 제거
        prompt = prompt.strip()
        
        return prompt
    
    def add_template(self, template: PromptTemplate):
        """템플릿 추가"""
        self.templates[template.name] = template
        logger.info(f"템플릿 추가됨: {template.name}")
    
    def remove_template(self, template_name: str):
        """템플릿 제거"""
        if template_name in self.templates:
            del self.templates[template_name]
            logger.info(f"템플릿 제거됨: {template_name}")
    
    def list_templates(self, category: Optional[str] = None) -> List[str]:
        """템플릿 목록 조회"""
        if category:
            return [name for name, template in self.templates.items() 
                   if template.category == category]
        return list(self.templates.keys())
    
    def get_template_info(self, template_name: str) -> Optional[PromptTemplate]:
        """템플릿 정보 조회"""
        return self.templates.get(template_name)
    
    def optimize_prompt(self, prompt: str, target_length: Optional[int] = None) -> str:
        """프롬프트 최적화"""
        # 기본적인 최적화 로직
        optimized = prompt
        
        # 불필요한 공백 제거
        optimized = re.sub(r'\s+', ' ', optimized)
        
        # 길이 제한이 있는 경우
        if target_length and len(optimized) > target_length:
            # 중요하지 않은 부분 축약
            optimized = self._truncate_prompt(optimized, target_length)
        
        return optimized
    
    def _truncate_prompt(self, prompt: str, max_length: int) -> str:
        """프롬프트 길이 제한"""
        if len(prompt) <= max_length:
            return prompt
        
        # 중요한 부분을 보존하면서 축약
        lines = prompt.split('\n')
        truncated_lines = []
        current_length = 0
        
        for line in lines:
            if current_length + len(line) + 1 <= max_length:
                truncated_lines.append(line)
                current_length += len(line) + 1
            else:
                # 남은 공간에 맞춰 마지막 줄 축약
                remaining = max_length - current_length - 4  # "..." 고려
                if remaining > 0:
                    truncated_lines.append(line[:remaining] + "...")
                break
        
        return '\n'.join(truncated_lines)
    
    def validate_template(self, template: PromptTemplate) -> List[str]:
        """템플릿 유효성 검사"""
        issues = []
        
        # 필수 필드 확인
        if not template.name:
            issues.append("템플릿 이름이 없습니다")
        
        if not template.template:
            issues.append("템플릿 내용이 없습니다")
        
        # 변수 일치성 확인
        template_vars = set(re.findall(r'\{(\w+)\}', template.template))
        declared_vars = set(template.variables)
        
        missing_vars = template_vars - declared_vars
        if missing_vars:
            issues.append(f"선언되지 않은 변수: {missing_vars}")
        
        unused_vars = declared_vars - template_vars
        if unused_vars:
            issues.append(f"사용되지 않은 변수: {unused_vars}")
        
        return issues
    
    def get_statistics(self) -> Dict[str, Any]:
        """프롬프트 엔진 사용 통계"""
        return {
            'total_templates': len(self.templates),
            'categories': list(set(t.category for t in self.templates.values())),
            'cache_enabled': hasattr(self, 'cache') and self.cache is not None,
            'default_language': getattr(self, 'default_language', 'ko')
        }
    
    def create_analysis_proposal_prompt(self, input_data: Dict[str, Any], 
                                      rag_context: Dict[str, Any]) -> str:
        """분석 제안을 위한 프롬프트 생성"""
        try:
            # 프롬프트 컨텍스트 구성
            context = PromptContext(
                user_request=input_data.get('user_request', ''),
                data_info=input_data.get('data_overview', {}),
                analysis_type=input_data.get('analysis_recommendations', {}).get('suitable_analyses', []),
                domain_knowledge=rag_context.get('statistical_knowledge', []),
                constraints={
                    'data_limitations': input_data.get('data_quality_assessment', {}).get('limitations', []),
                    'statistical_assumptions': input_data.get('summary_insights', {}).get('statistical_considerations', [])
                }
            )
            
            # 기본 분석 제안 템플릿 사용
            prompt = self.generate_prompt("statistical_test_proposal", context)
            
            return prompt
            
        except Exception as e:
            logger.error(f"분석 제안 프롬프트 생성 오류: {e}")
            return self._get_fallback_analysis_prompt(input_data)
    
    def create_domain_insight_prompt(self, input_data: Dict[str, Any], 
                                   rag_context: Dict[str, Any]) -> str:
        """도메인 인사이트를 위한 프롬프트 생성"""
        try:
            context = PromptContext(
                user_request=input_data.get('user_request', ''),
                data_info=input_data.get('data_overview', {}),
                domain_knowledge=rag_context.get('domain_knowledge', []),
                previous_results=input_data.get('analysis_recommendations', {}),
                constraints={
                    'business_context': rag_context.get('business_context', {}),
                    'industry_standards': rag_context.get('industry_standards', [])
                }
            )
            
            # 도메인 인사이트 템플릿 생성
            template = """
다음 데이터 분석 상황에서 도메인별 인사이트를 제공해주세요.

사용자 요청: {user_request}

데이터 정보:
{data_info}

관련 도메인 지식:
{domain_knowledge}

비즈니스 컨텍스트:
{constraints}

다음을 포함하여 분석해주세요:

1. 비즈니스 컨텍스트:
   - 핵심 비즈니스 지표 식별
   - 업계 표준 및 벤치마크
   - 의사결정에 미치는 영향

2. 유사 사례:
   - 동일 업계 분석 사례
   - 성공/실패 요인
   - 적용 가능한 인사이트

3. 도메인별 고려사항:
   - 업계 특성 반영사항
   - 규제 및 윤리적 고려사항
   - 실무적 제약사항

JSON 형태로 응답해주세요.
"""
            
            # 임시 템플릿으로 프롬프트 생성
            temp_template = PromptTemplate(
                name="domain_insight_temp",
                template=template,
                variables=["user_request", "data_info", "domain_knowledge", "constraints"],
                description="도메인 인사이트 임시 템플릿",
                category="domain"
            )
            
            return self._render_template(temp_template, self._extract_variables_from_context(context))
            
        except Exception as e:
            logger.error(f"도메인 인사이트 프롬프트 생성 오류: {e}")
            return self._get_fallback_domain_prompt(input_data)
    
    def _get_fallback_analysis_prompt(self, input_data: Dict[str, Any]) -> str:
        """분석 제안 폴백 프롬프트"""
        user_request = input_data.get('user_request', '사용자 요청 없음')
        data_info = input_data.get('data_overview', {})
        
        return f"""
다음 데이터 분석 요청에 대해 적절한 통계 분석 방법을 제안해주세요.

사용자 요청: {user_request}

데이터 정보:
- 행 수: {data_info.get('row_count', '알 수 없음')}
- 열 수: {data_info.get('column_count', '알 수 없음')}
- 주요 변수: {', '.join(data_info.get('columns', [])[:5])}

다음 형태로 응답해주세요:

추천 방법:
- 방법 1: [방법명] - [설명]
- 방법 2: [방법명] - [설명]

대안 방법:
- 대안 1: [방법명] - [설명]

근거:
- [선택 이유]
"""
    
    def _get_fallback_domain_prompt(self, input_data: Dict[str, Any]) -> str:
        """도메인 인사이트 폴백 프롬프트"""
        user_request = input_data.get('user_request', '사용자 요청 없음')
        
        return f"""
다음 분석 요청에 대한 도메인 인사이트를 제공해주세요.

사용자 요청: {user_request}

다음을 포함하여 분석해주세요:

비즈니스 컨텍스트:
- 이 분석이 비즈니스에 미치는 영향
- 주요 성과 지표

유사 사례:
- 관련 업계 사례

고려사항:
- 분석 시 주의할 점
- 실무적 제약사항
""" 