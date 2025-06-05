"""
Enhanced RAG 기반 Analysis Recommender

비즈니스 도메인 지식과 DB 스키마 구조를 활용하여 
컨텍스트 인식 통계 분석 방법을 LLM을 통해 추천하는 모듈
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path


class AnalysisRecommendation:
    """Enhanced RAG 기반 분석 추천 결과를 담는 클래스"""
    
    def __init__(self, method_name: str, description: str, reasoning: str, 
                 confidence_score: float, business_interpretation: str = "",
                 schema_considerations: str = "", required_variables: Dict[str, str] = None):
        self.method_name = method_name
        self.description = description
        self.reasoning = reasoning
        self.confidence_score = confidence_score
        self.business_interpretation = business_interpretation
        self.schema_considerations = schema_considerations
        self.required_variables = required_variables or {}


class AnalysisRecommender:
    """Enhanced RAG 기반 LLM 분석 방법 추천 클래스"""
    
    def __init__(self, llm_client, prompt_crafter):
        self.llm_client = llm_client
        self.prompt_crafter = prompt_crafter
        self.logger = logging.getLogger(__name__)
    
    def generate_recommendations(self, natural_language_request: str, data_summary: Dict,
                               business_context: Dict, schema_context: Dict) -> List[AnalysisRecommendation]:
        """
        Enhanced RAG 정보를 활용하여 컨텍스트 인식 분석 방법을 추천합니다.
        
        Args:
            natural_language_request: 사용자의 자연어 분석 요청
            data_summary: 데이터 요약 정보
            business_context: RAG 시스템에서 검색된 비즈니스 컨텍스트
            schema_context: RAG 시스템에서 검색된 DB 스키마 컨텍스트
            
        Returns:
            List[AnalysisRecommendation]: 추천된 분석 방법들 (최대 3개)
        """
        try:
            self.logger.info("Enhanced RAG 기반 분석 방법 추천 생성 시작")
            
            # 컨텍스트 통합 프롬프트 생성
            prompt = self._create_enhanced_recommendation_prompt(
                natural_language_request, data_summary, business_context, schema_context
            )
            
            # LLM 호출
            response = self.llm_client.get_completion(prompt)
            
            # 응답 파싱
            recommendations = self._parse_enhanced_recommendations(response)
            
            self.logger.info(f"{len(recommendations)}개의 추천 방법 생성 완료")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Enhanced RAG 추천 생성 중 오류: {e}")
            return self._get_fallback_recommendations()
    
    def _create_enhanced_recommendation_prompt(self, natural_language_request: str, 
                                            data_summary: Dict, business_context: Dict, 
                                            schema_context: Dict) -> str:
        """Enhanced RAG 컨텍스트를 활용한 추천 프롬프트 생성 (JSON 프롬프트 활용)"""
        
        try:
            # PromptCrafter의 JSON 프롬프트 사용
            dynamic_data = {
                'natural_language_request': natural_language_request,
                'data_summary': data_summary,
                'business_context': business_context,
                'schema_context': schema_context,
                # 데이터 요약 정보 개별 필드
                'shape': data_summary.get('shape', 'N/A'),
                'columns': ', '.join(data_summary.get('columns', [])),
                'numeric_columns': ', '.join(data_summary.get('numeric_columns', [])),
                'categorical_columns': ', '.join(data_summary.get('categorical_columns', [])),
                # 비즈니스 컨텍스트 개별 필드  
                'domain_knowledge': business_context.get('domain_knowledge', ''),
                'terminology': business_context.get('terminology', ''),
                'analysis_guidelines': business_context.get('analysis_guidelines', ''),
                # 스키마 컨텍스트 개별 필드
                'table_definitions': schema_context.get('table_definitions', ''),
                'column_descriptions': schema_context.get('column_descriptions', ''),
                'relationships': schema_context.get('relationships', ''),
                'constraints': schema_context.get('constraints', '')
            }
            
            # ai_recommendation_generation 노드에 해당하는 JSON 프롬프트 사용
            prompt = self.prompt_crafter.get_prompt_for_node(
                'ai_recommendation_generation', 
                dynamic_data=dynamic_data
            )
            
            if prompt:
                self.logger.info("JSON 프롬프트 템플릿을 사용하여 추천 프롬프트 생성")
                return prompt
            else:
                self.logger.warning("JSON 프롬프트를 찾을 수 없어 폴백 프롬프트 사용")
                return self._create_fallback_prompt(
                    natural_language_request, data_summary, business_context, schema_context
                )
                
        except Exception as e:
            self.logger.error(f"JSON 프롬프트 생성 오류: {e}")
            return self._create_fallback_prompt(
                natural_language_request, data_summary, business_context, schema_context
            )
    
    def _create_fallback_prompt(self, natural_language_request: str, 
                              data_summary: Dict, business_context: Dict, 
                              schema_context: Dict) -> str:
        """JSON 프롬프트 사용 실패시 폴백 프롬프트"""
        
        prompt = f"""
# 📊 Enhanced RAG 기반 통계 분석 방법 추천

## 🗣️ 사용자 분석 요청
{natural_language_request}

## 📋 데이터 정보
- **크기**: {data_summary.get('shape', 'N/A')}
- **컬럼**: {', '.join(data_summary.get('columns', []))}
- **수치형 변수**: {', '.join(data_summary.get('numeric_columns', []))}
- **범주형 변수**: {', '.join(data_summary.get('categorical_columns', []))}

## 🏢 비즈니스 컨텍스트 (RAG 검색 결과)
"""
        
        # 비즈니스 컨텍스트 정보 추가
        if business_context:
            if 'domain_knowledge' in business_context:
                prompt += f"**도메인 지식**: {business_context['domain_knowledge']}\n"
            if 'terminology' in business_context:
                prompt += f"**업계 용어**: {business_context['terminology']}\n"
            if 'analysis_guidelines' in business_context:
                prompt += f"**분석 가이드라인**: {business_context['analysis_guidelines']}\n"
        
        prompt += "\n## 🗄️ DB 스키마 컨텍스트 (RAG 검색 결과)\n"
        
        # 스키마 컨텍스트 정보 추가
        if schema_context:
            if 'table_definitions' in schema_context:
                prompt += f"**테이블 정의**: {schema_context['table_definitions']}\n"
            if 'column_descriptions' in schema_context:
                prompt += f"**컬럼 설명**: {schema_context['column_descriptions']}\n"
            if 'relationships' in schema_context:
                prompt += f"**테이블 관계**: {schema_context['relationships']}\n"
            if 'constraints' in schema_context:
                prompt += f"**제약조건**: {schema_context['constraints']}\n"
        
        prompt += f"""

## 📝 추천 요청사항
위의 **사용자 요청**, **데이터 정보**, **비즈니스 컨텍스트**, **DB 스키마 정보**를 모두 고려하여 
가장 적합한 통계 분석 방법 **3가지**를 추천해주세요.

각 추천에 대해 다음 형식으로 답변해주세요:

## 🥇 추천 1: [분석방법명]
- **설명**: [방법의 간단한 설명]
- **적용 이유**: [이 데이터와 비즈니스 컨텍스트에 적합한 이유]
- **신뢰도**: [0.0-1.0 사이의 값]
- **비즈니스 해석**: [비즈니스 도메인 관점에서의 결과 해석 방향]
- **스키마 고려사항**: [DB 스키마 제약조건 및 관계 고려사항]
- **필요변수**: [종속변수: 컬럼명, 독립변수: 컬럼명 등]

## 🥈 추천 2: [분석방법명]
[동일한 형식으로...]

## 🥉 추천 3: [분석방법명]
[동일한 형식으로...]

**중요**: 반드시 비즈니스 컨텍스트와 DB 스키마 정보를 고려하여 실무적으로 의미 있는 분석만 추천해주세요.
"""
        return prompt
    
    def _parse_enhanced_recommendations(self, response: str) -> List[AnalysisRecommendation]:
        """Enhanced RAG 기반 LLM 응답을 파싱하여 추천 목록 생성"""
        recommendations = []
        
        try:
            # 추천 섹션 분할
            sections = response.split('## 🥇')[1:] + response.split('## 🥈')[1:] + response.split('## 🥉')[1:]
            sections = [s for s in sections if s.strip()]  # 빈 섹션 제거
            
            for i, section in enumerate(sections[:3]):  # 최대 3개
                lines = section.strip().split('\n')
                
                if not lines:
                    continue
                    
                # 방법명 추출
                method_name = lines[0].split(':')[1].strip() if ':' in lines[0] else f"분석방법 {i+1}"
                
                # 각 필드 추출 및 초기화
                description = ""
                reasoning = ""
                confidence_score = 0.8  # 기본값
                business_interpretation = ""
                schema_considerations = ""
                required_variables = {}
                
                for line in lines[1:]:
                    line = line.strip()
                    if line.startswith('- **설명**:'):
                        description = line.replace('- **설명**:', '').strip()
                    elif line.startswith('- **적용 이유**:'):
                        reasoning = line.replace('- **적용 이유**:', '').strip()
                    elif line.startswith('- **신뢰도**:'):
                        try:
                            confidence_str = line.replace('- **신뢰도**:', '').strip()
                            confidence_score = float(confidence_str)
                        except:
                            confidence_score = 0.8
                    elif line.startswith('- **비즈니스 해석**:'):
                        business_interpretation = line.replace('- **비즈니스 해석**:', '').strip()
                    elif line.startswith('- **스키마 고려사항**:'):
                        schema_considerations = line.replace('- **스키마 고려사항**:', '').strip()
                    elif line.startswith('- **필요변수**:'):
                        var_str = line.replace('- **필요변수**:', '').strip()
                        required_variables = {'raw': var_str}
                
                recommendation = AnalysisRecommendation(
                    method_name=method_name,
                    description=description,
                    reasoning=reasoning,
                    confidence_score=confidence_score,
                    business_interpretation=business_interpretation,
                    schema_considerations=schema_considerations,
                    required_variables=required_variables
                )
                recommendations.append(recommendation)
                
        except Exception as e:
            self.logger.error(f"Enhanced RAG 추천 응답 파싱 오류: {e}")
            return self._get_fallback_recommendations()
        
        return recommendations[:3]  # 최대 3개
    
    def _get_fallback_recommendations(self) -> List[AnalysisRecommendation]:
        """Enhanced RAG 추천 실패시 기본 추천"""
        return [
            AnalysisRecommendation(
                method_name="기술통계 분석",
                description="데이터의 기본 통계량 (평균, 표준편차 등)을 계산합니다",
                reasoning="모든 데이터에 적용 가능한 기본 분석입니다",
                confidence_score=0.9,
                business_interpretation="데이터의 전반적인 특성을 파악하여 의사결정 기초자료로 활용",
                schema_considerations="테이블 구조에 관계없이 적용 가능"
            ),
            AnalysisRecommendation(
                method_name="상관관계 분석",
                description="변수들 간의 선형 관계의 강도를 측정합니다",
                reasoning="수치형 변수가 2개 이상 있는 경우 유용한 분석입니다",
                confidence_score=0.7,
                business_interpretation="변수 간 관계를 통한 비즈니스 인사이트 도출",
                schema_considerations="정규화된 테이블 구조에서 JOIN을 통한 분석 고려"
            ),
            AnalysisRecommendation(
                method_name="그룹별 비교 분석",
                description="범주형 변수에 따른 그룹 간 차이를 분석합니다",
                reasoning="범주형 변수가 있는 경우 그룹별 특성 비교가 가능합니다",
                confidence_score=0.6,
                business_interpretation="세그먼트별 성과 차이 분석을 통한 전략 수립",
                schema_considerations="외래키 관계를 활용한 차원별 분석 가능"
            )
        ]


def display_analysis_recommendations(recommendations: List[AnalysisRecommendation]) -> Optional[int]:
    """
    Enhanced RAG 기반 분석 추천 결과를 표시하고 사용자 선택을 받습니다.
    
    Args:
        recommendations: 추천된 분석 방법들
        
    Returns:
        Optional[int]: 선택된 추천의 인덱스 (0-based) 또는 None
    """
    if not recommendations:
        print("❌ 추천할 수 있는 분석 방법이 없습니다.")
        return None
    
    print("\n🤖 Enhanced RAG 기반 AI 분석 방법 추천 결과:")
    print("=" * 70)
    
    for i, rec in enumerate(recommendations, 1):
        confidence_bar = "🟩" * int(rec.confidence_score * 10) + "⬜" * (10 - int(rec.confidence_score * 10))
        
        print(f"\n{i}. 🎯 **{rec.method_name}** (신뢰도: {rec.confidence_score:.1f})")
        print(f"   {confidence_bar}")
        print(f"   📝 {rec.description}")
        print(f"   🔍 적용 이유: {rec.reasoning}")
        
        if rec.business_interpretation:
            print(f"   🏢 비즈니스 해석: {rec.business_interpretation}")
        
        if rec.schema_considerations:
            print(f"   🗄️ 스키마 고려사항: {rec.schema_considerations}")
    
    print("\n" + "=" * 70)
    
    while True:
        try:
            choice = input(f"\n🎯 추천된 분석 방법을 선택하세요 (1-{len(recommendations)}, 또는 0=직접입력): ").strip()
            
            if not choice:
                continue
                
            choice_num = int(choice)
            if choice_num == 0:
                print("💬 자유롭게 분석을 요청하세요.")
                return None
            elif 1 <= choice_num <= len(recommendations):
                selected_rec = recommendations[choice_num - 1]
                print(f"✅ 선택된 분석: {selected_rec.method_name}")
                return choice_num - 1
            else:
                print(f"❌ 0부터 {len(recommendations)} 사이의 숫자를 입력하세요.")
                
        except ValueError:
            print("❌ 숫자를 입력하세요.")
        except KeyboardInterrupt:
            print("\n\n👋 분석 방법 선택을 취소합니다.")
            return None 