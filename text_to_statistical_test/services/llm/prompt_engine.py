import logging
from typing import Dict, Any

# [UTIL-REQ] error_handler.py의 PromptException 클래스가 필요합니다.
from utils.error_handler import PromptException

logger = logging.getLogger(__name__)

class PromptEngine:
    """다양한 작업에 대한 프롬프트를 동적으로 생성하고 관리합니다."""
    def __init__(self):
        self._templates = self._load_templates()
        logger.info(f"{len(self._templates)}개의 프롬프트 템플릿이 로드되었습니다.")

    def create_prompt(self, template_name: str, context: Dict[str, Any]) -> str:
        """템플릿과 컨텍스트를 결합하여 최종 프롬프트를 생성합니다."""
        if template_name not in self._templates:
            raise PromptException(f"'{template_name}' 템플릿을 찾을 수 없습니다.")
        
        template = self._templates[template_name]
        try:
            return template.format(**context)
        except KeyError as e:
            raise PromptException(f"프롬프트 생성 실패: '{template_name}' 템플릿에 '{e}' 키가 필요합니다.")

    def _load_templates(self) -> Dict[str, str]:
        """프로젝트에서 사용할 프롬프트 템플릿들을 정의합니다."""
        templates = {}

        # 2단계: 사용자 요청 분석용
        templates['interpret_user_request'] = """
당신은 데이터 분석 전문가입니다. 사용자의 자연어 요청과 데이터 정보를 분석하여, 분석의 핵심 목표, 관련 변수, 적절한 분석 유형을 구조화된 JSON 형식으로 제안해주세요.
사용자가 한글로 언급한 개념(예: '성별', '만족도')을 데이터의 실제 영어 컬럼명과 지능적으로 매칭하세요.
만약 회귀분석 요청이라면 analysis_type을 'regression'으로, 로지스틱 회귀분석이면 'logistic_regression'으로 지정해주세요.

## 사용자 요청
"{user_request}"

## 데이터 정보
{data_context}

## 응답 형식 (JSON)
```json
{{
    "main_goal": "분석의 주요 목적 요약",
    "analysis_type": "group_comparison | correlation | regression | logistic_regression | descriptive | categorical",
    "variables": {{
        "dependent": ["종속변수/분석대상 컬럼명"],
        "independent": ["독립변수/그룹변수 컬럼명"]
    }},
    "hypotheses": ["분석을 통해 검증할 수 있는 가설 1~2개"]
}}
"""

    # 3단계: 상세 분석 계획 수립용
        templates['create_detailed_analysis_plan'] = """
당신은 모든 통계 분석 방법에 통달한 AI 통계학자입니다.
사용자의 분석 목표와 데이터 변수 타입을 바탕으로, 통계 분석을 위한 완벽한 실행 계획을 JSON 형식으로 수립해주세요.

분석 목표
{structured_request}

지시사항
분석 식별: 분석 목표에 가장 적합한 핵심 분석(primary_test)을 결정하세요. (예: "독립표본 t-검정", "다중선형회귀분석")
가정 정의: 해당 분석을 위해 반드시 검토해야 할 통계적 가정(assumptions)을 모두 나열하세요. (예: "정규성", "등분산성", "다중공선성")
대안 계획: 핵심 가정이 깨졌을 때 사용할 대안 분석(fallback_test)을 명시하세요.
사후 분석: 분석 결과에 따라 사후 분석이 필요한 경우(posthoc_needed) true로 설정하세요. (주로 ANOVA 계열)
효과 크기: 분석 결과의 실제적 중요성을 측정할 효과 크기 계산법(effect_size_method)을 지정하세요.
응답 형식 (JSON)
JSON

{{
    "primary_test": "핵심 분석 방법 이름",
    "assumptions": ["필요한 사전 가정 목록"],
    "fallback_test": "가정 실패 시 사용할 대안 분석 이름 또는 null",
    "posthoc_needed": true,
    "effect_size_method": "효과 크기 측정 방법 이름 (예: Cohen's d)"
}}
"""

    # 4단계: 시각화 추천용
        templates['recommend_visualizations'] = """
당신은 데이터 시각화 전문가입니다. 주어진 통계 분석 결과를 바탕으로, 결과를 가장 잘 설명할 수 있는 1~3개의 시각화 유형을 JSON 형식으로 추천해주세요.

분석 결과 요약
{analysis_results}

추천 형식
그룹 비교 (t-test, ANOVA): 'boxplot', 'violinplot'
관계 분석 (상관, 회귀): 'scatterplot', 'regplot'
회귀 진단: 'residual_plot', 'qq_plot'
분포 확인: 'histogram', 'kdeplot'
응답 형식 (JSON Array)
JSON

[
    {{
        "type": "시각화 종류 (예: boxplot)",
        "params": {{
            "x": "x축에 사용할 컬럼명",
            "y": "y축에 사용할 컬럼명"
        }},
        "title": "차트 제목"
    }}
]
"""

    # 5단계: 보고서 서술 생성용
        templates['generate_report_narrative'] = """
당신은 데이터 스토리텔러입니다. 주어진 분석 계획과 통계 실행 결과를 바탕으로, 비전문가도 쉽게 이해할 수 있는 분석 보고서의 서술부를 JSON 형식으로 작성해주세요.

분석 계획
{final_plan}

통계 분석 결과
{analysis_results}

시각화 자료 목록
{visual_artifacts}

보고서 작성 가이드
Executive Summary: 분석의 핵심 결론과 비즈니스 관점의 시사점을 1~2문장으로 요약.
Analysis Process: 어떤 데이터를 가지고 어떤 분석을 수행했는지 간략히 설명. 가정 검토 결과와 그에 따른 분석 방법 선택 과정을 반드시 포함.
Detailed Interpretation: 통계 결과를(p-value, 효과 크기 등) 쉬운 말로 풀어 설명. 시각화 자료를 함께 참조하여 설명.
Conclusion: 분석을 통해 최종적으로 무엇을 알 수 있었는지 명확히 결론.
응답 형식 (JSON)
JSON

{{
  "executive_summary": "...",
  "analysis_process": "...",
  "detailed_interpretation": "...",
  "conclusion": "..."
}}
"""
        return templates