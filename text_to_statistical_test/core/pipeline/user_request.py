"""
User Request Pipeline

2단계: LLM Agent 기반 사용자 요청 분석 및 목표 정의
LLM Agent가 사용자의 자연어 질문과 데이터를 함께 분석하여
유연하고 지능적으로 분석 목표를 설정합니다.
"""

import logging
from typing import Dict, Any, Optional, List
import json
import pandas as pd

from .base_pipeline_step import BasePipelineStep, PipelineStepRegistry
from services.llm.llm_client import LLMClient
from services.llm.prompt_engine import PromptEngine
from core.agent.autonomous_agent import AutonomousAgent
from utils.ui_helpers import get_user_input
from utils.data_loader import DataLoader


class UserRequestStep(BasePipelineStep):
    """2단계: LLM Agent 기반 사용자 요청 분석 및 목표 정의"""
    
    def __init__(self):
        """UserRequestStep 초기화"""
        super().__init__("LLM Agent 기반 사용자 요청 분석", 2)
        self.llm_client = LLMClient()
        self.prompt_engine = PromptEngine()
        self.agent = AutonomousAgent(agent_id="request_analyst")
        self.data_loader = DataLoader()
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """입력 데이터 유효성 검증"""
        required_fields = ['selected_file', 'file_info']
        return all(field in input_data for field in required_fields)
    
    def get_expected_output_schema(self) -> Dict[str, Any]:
        """예상 출력 스키마 반환"""
        return {
            'user_request': str,
            'analysis_objectives': dict,
            'agent_analysis': dict,
            'data_understanding': dict,
            'analysis_plan': dict,
            'confidence_level': str
        }
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """LLM Agent 기반 사용자 요청 분석 실행"""
        self.logger.info("LLM Agent 기반 사용자 요청 분석 시작")
        
        try:
            # 사용자 요청 수집
            user_request = self._get_user_request()
            if not user_request:
                return {
                    'error': True,
                    'error_message': '사용자 요청이 제공되지 않았습니다.'
                }
            
            # 데이터 로딩 및 기본 이해
            data_info = self._load_and_understand_data(input_data)
            if data_info.get('error'):
                return data_info
            
            # LLM Agent를 통한 통합 분석
            agent_analysis = self._analyze_with_llm_agent(user_request, data_info, input_data)
            
            # 분석 계획 생성
            analysis_plan = self._generate_analysis_plan(agent_analysis, data_info)
            
            self.logger.info("LLM Agent 기반 분석 완료")
            
            return {
                'success': True,
                'user_request': user_request,
                'analysis_objectives': agent_analysis.get('objectives', {}),
                'agent_analysis': agent_analysis,
                'data_understanding': data_info,
                'analysis_plan': analysis_plan,
                'confidence_level': agent_analysis.get('confidence', 'medium'),
                'step_info': self.get_step_info()
            }
            
        except Exception as e:
            self.logger.error(f"LLM Agent 분석 중 오류: {e}")
            return {
                'error': True,
                'error_message': f'분석 중 오류가 발생했습니다: {str(e)}',
                'error_type': 'agent_analysis_error'
            }
    
    def _get_user_request(self) -> Optional[str]:
        """사용자 요청 입력 받기"""
        print("\n" + "="*60)
        print("📝 분석하고 싶은 내용을 자연어로 설명해주세요")
        print("="*60)
        print("예시:")
        print("• 성별에 따른 만족도 평균 차이를 분석해줘")
        print("• 나이와 소득의 상관관계를 알고 싶어")
        print("• 교육수준별로 연봉 차이가 있는지 확인해줘")
        print("• 지역별 매출 분포를 비교 분석해줘")
        print("-"*60)
        
        user_request = get_user_input(
            "분석 요청을 입력해주세요: ",
            input_type="text"
        )
        
        if user_request and len(user_request.strip()) > 5:
            return user_request.strip()
        else:
            print("❌ 분석 요청이 너무 짧습니다. 최소 5글자 이상 입력해주세요.")
            return None
    
    def _load_and_understand_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 로딩 및 기본 이해"""
        try:
            file_path = input_data['selected_file']
            
            # 데이터 로딩
            data, metadata = self.data_loader.load_file(file_path)
            if data is None:
                return {
                    'error': True,
                    'error_message': f'데이터 로딩 실패: {metadata.get("error", "Unknown error")}'
                }
            
            # 데이터 기본 정보 수집
            data_info = {
                'file_path': file_path,
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
                'sample_data': data.head(3).to_dict('records'),
                'missing_info': {col: int(data[col].isnull().sum()) for col in data.columns},
                'numerical_columns': list(data.select_dtypes(include=['number']).columns),
                'categorical_columns': list(data.select_dtypes(include=['object', 'category']).columns),
                'data_object': data  # 다음 단계에서 사용할 데이터 객체
            }
            
            return data_info
            
        except Exception as e:
            self.logger.error(f"데이터 로딩 및 이해 오류: {e}")
            return {
                'error': True,
                'error_message': f'데이터 처리 중 오류: {str(e)}'
            }
    
    def _analyze_with_llm_agent(self, user_request: str, data_info: Dict[str, Any], 
                               input_data: Dict[str, Any]) -> Dict[str, Any]:
        """LLM Agent를 통한 통합 분석"""
        try:
            # 데이터 컨텍스트 구성
            data_context = self._build_data_context(data_info)
            
            # LLM Agent 분석 프롬프트 생성
            analysis_prompt = self._create_agent_analysis_prompt(user_request, data_context)
            
            # LLM Agent 실행
            response = self.llm_client.generate_response(
                analysis_prompt,
                max_tokens=1500,
                temperature=0.3
            )
            
            # 응답 파싱
            agent_analysis = self._parse_agent_response(response.content)
            
            # 응답 검증 및 보완
            validated_analysis = self._validate_and_enhance_analysis(
                agent_analysis, user_request, data_info
            )
            
            return validated_analysis
            
        except Exception as e:
            self.logger.error(f"LLM Agent 분석 오류: {e}")
            # 백업 분석 실행
            return self._fallback_analysis(user_request, data_info)
    
    def _build_data_context(self, data_info: Dict[str, Any]) -> str:
        """데이터 컨텍스트 구성"""
        context_parts = []
        
        # 기본 정보
        context_parts.append(f"데이터 크기: {data_info['shape'][0]}행 × {data_info['shape'][1]}열")
        
        # 컬럼 정보
        context_parts.append("컬럼 정보:")
        for col in data_info['columns']:
            dtype = data_info['dtypes'][col]
            missing = data_info['missing_info'][col]
            missing_pct = round((missing / data_info['shape'][0]) * 100, 1)
            
            context_parts.append(f"  - {col} ({dtype}): 결측치 {missing}개 ({missing_pct}%)")
        
        # 샘플 데이터
        context_parts.append("\n샘플 데이터 (처음 3행):")
        for i, row in enumerate(data_info['sample_data'], 1):
            row_str = ", ".join([f"{k}={v}" for k, v in row.items()][:5])  # 처음 5개 컬럼만
            context_parts.append(f"  {i}. {row_str}")
        
        # 변수 타입 요약
        num_cols = len(data_info['numerical_columns'])
        cat_cols = len(data_info['categorical_columns'])
        context_parts.append(f"\n변수 유형: 수치형 {num_cols}개, 범주형 {cat_cols}개")
        
        return "\n".join(context_parts)
    
    def _create_agent_analysis_prompt(self, user_request: str, data_context: str) -> str:
        """LLM Agent 분석용 프롬프트 생성"""
        prompt = f"""
당신은 데이터 분석 전문가입니다. 사용자의 자연어 요청과 데이터 정보를 분석하여 최적의 통계 분석 방법을 제안해주세요.

## 사용자 요청
"{user_request}"

## 데이터 정보
{data_context}

## 분석 과제
사용자의 요청을 분석하여 다음을 결정해주세요:

1. 사용자가 원하는 분석의 핵심 목적
2. 분석에 필요한 변수들 (영어 컬럼명과 한글 요청 간 매칭 포함)
3. 적절한 통계 분석 방법
4. 분석 과정에서 고려해야 할 사항들

## 응답 형식 (JSON)
다음 형식으로 정확히 응답해주세요:

```json
{{
    "objectives": {{
        "main_goal": "분석의 주요 목적",
        "specific_questions": ["구체적인 분석 질문들"],
        "analysis_type": "group_comparison|correlation|regression|descriptive|categorical"
    }},
    "variables": {{
        "target_variables": ["종속변수/분석대상 컬럼명들"],
        "predictor_variables": ["독립변수/그룹변수 컬럼명들"],
        "variable_matching": {{
            "사용자언급단어": "실제컬럼명"
        }}
    }},
    "analysis_methods": {{
        "primary_method": "주요 분석 방법",
        "alternative_methods": ["대안 분석 방법들"],
        "preprocessing_needed": ["필요한 전처리 단계들"]
    }},
    "considerations": {{
        "data_quality_issues": ["데이터 품질 이슈들"],
        "statistical_assumptions": ["확인해야 할 통계적 가정들"],
        "potential_challenges": ["예상되는 분석 어려움들"]
    }},
    "confidence": "high|medium|low",
    "reasoning": "분석 판단의 근거"
}}
```

중요: 사용자가 한글로 언급한 개념들을 데이터의 실제 영어 컬럼명과 지능적으로 매칭하세요.
예: "성별" → "gender", "만족도" → "satisfaction", "나이" → "age"
"""
        
        return prompt
    
    def _parse_agent_response(self, response_content: str) -> Dict[str, Any]:
        """LLM Agent 응답 파싱"""
        try:
            # JSON 블록 추출
            json_start = response_content.find('```json')
            json_end = response_content.find('```', json_start + 7)
            
            if json_start != -1 and json_end != -1:
                json_str = response_content[json_start + 7:json_end].strip()
            else:
                # JSON 블록이 없으면 전체 응답에서 JSON 찾기
                json_str = response_content.strip()
            
            # JSON 파싱
            parsed_response = json.loads(json_str)
            return parsed_response
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON 파싱 실패: {e}")
            # 기본 구조 반환
            return {
                "objectives": {
                    "main_goal": "데이터 분석",
                    "analysis_type": "descriptive"
                },
                "variables": {
                    "target_variables": [],
                    "predictor_variables": []
                },
                "confidence": "low",
                "reasoning": "JSON 파싱 실패로 기본 분석 적용"
            }
    
    def _validate_and_enhance_analysis(self, analysis: Dict[str, Any], 
                                     user_request: str, data_info: Dict[str, Any]) -> Dict[str, Any]:
        """분석 결과 검증 및 보완"""
        validated = analysis.copy()
        
        # 기본 구조 확인
        if 'objectives' not in validated:
            validated['objectives'] = {}
        if 'variables' not in validated:
            validated['variables'] = {}
        if 'analysis_methods' not in validated:
            validated['analysis_methods'] = {}
        
        # 변수명 검증 및 매칭
        available_columns = data_info['columns']
        
        # target_variables 검증
        target_vars = validated['variables'].get('target_variables', [])
        validated_targets = [var for var in target_vars if var in available_columns]
        
        # predictor_variables 검증
        predictor_vars = validated['variables'].get('predictor_variables', [])
        validated_predictors = [var for var in predictor_vars if var in available_columns]
        
        # 변수가 없으면 추론 시도
        if not validated_targets and not validated_predictors:
            inferred_vars = self._infer_variables_from_request(user_request, available_columns)
            validated_targets.extend(inferred_vars.get('targets', []))
            validated_predictors.extend(inferred_vars.get('predictors', []))
        
        validated['variables']['target_variables'] = validated_targets
        validated['variables']['predictor_variables'] = validated_predictors
        
        # 신뢰도 조정
        if not validated_targets and not validated_predictors:
            validated['confidence'] = 'low'
        elif validated.get('confidence') not in ['high', 'medium', 'low']:
            validated['confidence'] = 'medium'
        
        return validated
    
    def _infer_variables_from_request(self, user_request: str, 
                                    available_columns: List[str]) -> Dict[str, List[str]]:
        """요청에서 변수 추론"""
        request_lower = user_request.lower()
        
        # 한글-영어 매칭 사전
        common_mappings = {
            '성별': ['gender', 'sex'],
            '나이': ['age'],
            '만족도': ['satisfaction', 'rating', 'score'],
            '소득': ['income', 'salary', 'wage'],
            '연봉': ['salary', 'income', 'wage'],
            '교육': ['education', 'degree'],
            '지역': ['region', 'area', 'location'],
            '매출': ['sales', 'revenue'],
            '가격': ['price', 'cost'],
            '수량': ['quantity', 'amount']
        }
        
        targets = []
        predictors = []
        
        for korean_term, english_terms in common_mappings.items():
            if korean_term in request_lower:
                for eng_term in english_terms:
                    matching_cols = [col for col in available_columns 
                                   if eng_term.lower() in col.lower()]
                    if matching_cols:
                        # 비교/차이 분석의 경우
                        if any(word in request_lower for word in ['차이', '비교', '따른']):
                            if korean_term in ['성별', '지역', '교육']:
                                predictors.extend(matching_cols)
                            else:
                                targets.extend(matching_cols)
                        else:
                            targets.extend(matching_cols)
        
        return {'targets': list(set(targets)), 'predictors': list(set(predictors))}
    
    def _fallback_analysis(self, user_request: str, data_info: Dict[str, Any]) -> Dict[str, Any]:
        """백업 분석 (LLM 실패 시)"""
        self.logger.info("백업 분석 실행")
        
        # 기본 구조 생성
        analysis = {
            "objectives": {
                "main_goal": "데이터 탐색적 분석",
                "specific_questions": ["데이터의 기본 특성 파악"],
                "analysis_type": "descriptive"
            },
            "variables": {
                "target_variables": data_info['numerical_columns'][:2],  # 처음 2개 수치형 변수
                "predictor_variables": data_info['categorical_columns'][:2],  # 처음 2개 범주형 변수
            },
            "analysis_methods": {
                "primary_method": "기술통계분석",
                "alternative_methods": ["데이터 시각화"],
                "preprocessing_needed": ["결측치 확인"]
            },
            "confidence": "low",
            "reasoning": "LLM 분석 실패로 기본 탐색적 분석 적용"
        }
        
        return analysis
    
    def _generate_analysis_plan(self, agent_analysis: Dict[str, Any], 
                              data_info: Dict[str, Any]) -> Dict[str, Any]:
        """분석 계획 생성"""
        plan = {
            'analysis_steps': [],
            'expected_outputs': [],
            'estimated_duration': 'medium',
            'complexity_level': agent_analysis.get('confidence', 'medium')
        }
        
        # 분석 유형에 따른 계획 생성
        analysis_type = agent_analysis.get('objectives', {}).get('analysis_type', 'descriptive')
        
        if analysis_type == 'group_comparison':
            plan['analysis_steps'] = [
                '기술통계 계산',
                '정규성 검정',
                '그룹간 비교 검정 (t-test/ANOVA)',
                '결과 해석 및 시각화'
            ]
        elif analysis_type == 'correlation':
            plan['analysis_steps'] = [
                '기술통계 계산',
                '상관관계 분석',
                '산점도 시각화',
                '결과 해석'
            ]
        elif analysis_type == 'regression':
            plan['analysis_steps'] = [
                '기술통계 계산',
                '회귀분석 가정 검토',
                '회귀모델 수행',
                '모델 평가 및 해석'
            ]
        else:  # descriptive
            plan['analysis_steps'] = [
                '기술통계 계산',
                '데이터 분포 확인',
                '시각화 생성',
                '기본 패턴 분석'
            ]
        
        plan['expected_outputs'] = [
            '통계 검정 결과',
            '시각화 차트',
            '분석 해석 보고서'
        ]
        
        return plan
    
    def get_step_info(self) -> Dict[str, Any]:
        """단계 정보 반환"""
        base_info = super().get_step_info()
        base_info.update({
            'description': 'LLM Agent 기반 사용자 요청 분석',
            'input_requirements': ['selected_file', 'file_info'],
            'output_provides': [
                'analysis_objectives', 'agent_analysis', 'data_understanding', 'analysis_plan'
            ],
            'capabilities': [
                'LLM 기반 자연어 이해', '자동 변수 매칭', '분석 방법 추천', '분석 계획 수립'
            ]
        })
        return base_info


