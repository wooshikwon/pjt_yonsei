"""
Data Summary Pipeline

3단계: LLM Agent 기반 데이터 심층 분석 및 요약
LLM Agent가 데이터의 특성을 이해하고 사용자 요청과 연관지어
지능적으로 데이터를 분석하고 인사이트를 도출합니다.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import json

from .base_pipeline_step import BasePipelineStep, PipelineStepRegistry
from services.llm.llm_client import LLMClient
from services.statistics.descriptive_stats import DescriptiveStats
from services.statistics.data_preprocessor import DataPreprocessor


class DataSummaryStep(BasePipelineStep):
    """3단계: LLM Agent 기반 데이터 심층 분석 및 요약"""
    
    def __init__(self):
        """DataSummaryStep 초기화"""
        super().__init__("LLM Agent 기반 데이터 심층 분석", 3)
        self.llm_client = LLMClient()
        self.stats_calculator = DescriptiveStats()
        self.preprocessor = DataPreprocessor()
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """입력 데이터 유효성 검증"""
        required_fields = ['user_request', 'analysis_objectives', 'data_understanding']
        
        # 유연한 필드명 검증
        for field in required_fields:
            if field not in input_data:
                # 대안 필드명 확인
                alternative_found = False
                if field == 'analysis_objectives':
                    alternative_found = 'refined_objectives' in input_data
                elif field == 'data_understanding':
                    alternative_found = ('data_object' in input_data or 'selected_file' in input_data)
                
                if not alternative_found:
                    self.logger.error(f"필수 필드 누락: {field}")
                    return False
        
        return True
    
    def get_expected_output_schema(self) -> Dict[str, Any]:
        """예상 출력 스키마 반환"""
        return {
            'agent_data_analysis': dict,
            'data_insights': dict,
            'quality_assessment': dict,
            'analysis_recommendations': dict,
            'data_object': object,
            'enhanced_understanding': dict
        }
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """LLM Agent 기반 데이터 심층 분석 실행"""
        self.logger.info("3단계: LLM Agent 기반 데이터 심층 분석 시작")
        
        try:
            # 데이터 객체 확보
            data = self._get_data_object(input_data)
            if data is None:
                return {
                    'error': True,
                    'error_message': '데이터 객체를 가져올 수 없습니다.'
                }
            
            # 분석 목표 통합
            objectives = self._get_analysis_objectives(input_data)
            user_request = input_data.get('user_request', '')
            
            # 기본 데이터 통계 계산
            basic_stats = self._calculate_basic_statistics(data)
            
            # LLM Agent를 통한 데이터 분석
            agent_analysis = self._analyze_data_with_llm_agent(
                data, user_request, objectives, basic_stats
            )
            
            # 데이터 품질 평가
            quality_assessment = self._assess_data_quality_with_llm(data, agent_analysis)
            
            # 분석 추천사항 생성
            recommendations = self._generate_recommendations_with_llm(
                data, user_request, objectives, agent_analysis, quality_assessment
            )
            
            # 향상된 데이터 이해 구성
            enhanced_understanding = self._build_enhanced_understanding(
                data, agent_analysis, quality_assessment, recommendations
            )
            
            self.logger.info("LLM Agent 기반 데이터 분석 완료")
            
            return {
                'success': True,
                'agent_data_analysis': agent_analysis,
                'data_insights': agent_analysis.get('insights', {}),
                'quality_assessment': quality_assessment,
                'analysis_recommendations': recommendations,
                'data_object': data,
                'enhanced_understanding': enhanced_understanding,
                'step_info': self.get_step_info()
            }
                
        except Exception as e:
            self.logger.error(f"LLM Agent 데이터 분석 오류: {e}")
            return {
                'error': True,
                'error_message': f'데이터 분석 중 오류가 발생했습니다: {str(e)}',
                'error_type': 'agent_analysis_error'
            }
    
    def _get_data_object(self, input_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """데이터 객체 확보"""
        # 이미 로딩된 데이터 객체가 있는 경우
        if 'data_object' in input_data:
            return input_data['data_object']
        
        # data_understanding에서 데이터 객체 확인
        data_understanding = input_data.get('data_understanding', {})
        if 'data_object' in data_understanding:
            return data_understanding['data_object']
        
        # 파일에서 새로 로딩
        if 'selected_file' in input_data:
            from utils.data_loader import DataLoader
            loader = DataLoader()
            data, metadata = loader.load_file(input_data['selected_file'])
            return data
        
        return None
    
    def _get_analysis_objectives(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """분석 목표 통합 처리"""
        # 우선순위: analysis_objectives > refined_objectives
        objectives = input_data.get('analysis_objectives')
        if not objectives:
            objectives = input_data.get('refined_objectives', {})
        
        return objectives
    
    def _calculate_basic_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """기본 통계 계산"""
        try:
            numerical_cols = list(data.select_dtypes(include=['number']).columns)
            categorical_cols = list(data.select_dtypes(include=['object', 'category']).columns)
            
            stats = {
                'shape': {'rows': len(data), 'columns': len(data.columns)},
                'columns': list(data.columns),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
                'numerical_columns': numerical_cols,
                'categorical_columns': categorical_cols,
                'missing_values': {col: int(data[col].isnull().sum()) for col in data.columns},
                'sample_data': data.head(5).to_dict('records')
            }
            
            # 수치형 변수 기술통계
            if numerical_cols:
                stats['numerical_summary'] = data[numerical_cols].describe().to_dict()
            
            # 범주형 변수 빈도
            if categorical_cols:
                stats['categorical_summary'] = {}
                for col in categorical_cols:
                    value_counts = data[col].value_counts().head(10)
                    stats['categorical_summary'][col] = value_counts.to_dict()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"기본 통계 계산 오류: {e}")
            return {
                'shape': {'rows': len(data), 'columns': len(data.columns)},
                'columns': list(data.columns),
                'error': str(e)
            }
    
    def _analyze_data_with_llm_agent(self, data: pd.DataFrame, user_request: str, 
                                   objectives: Dict[str, Any], basic_stats: Dict[str, Any]) -> Dict[str, Any]:
        """LLM Agent를 통한 데이터 분석"""
        try:
            # 데이터 컨텍스트 구성
            data_context = self._build_detailed_data_context(data, basic_stats)
            
            # LLM 분석 프롬프트 생성
            analysis_prompt = self._create_data_analysis_prompt(
                user_request, objectives, data_context, basic_stats
            )
            
            # LLM Agent 실행
            response = self.llm_client.generate_response(
                analysis_prompt,
                max_tokens=2000,
                temperature=0.3
            )
            
            # 응답 파싱
            agent_analysis = self._parse_data_analysis_response(response.content)
            
            # 응답 검증 및 보완
            validated_analysis = self._validate_data_analysis(
                agent_analysis, data, basic_stats
            )
            
            return validated_analysis
            
        except Exception as e:
            self.logger.error(f"LLM 데이터 분석 오류: {e}")
            return self._fallback_data_analysis(data, basic_stats)
    
    def _build_detailed_data_context(self, data: pd.DataFrame, basic_stats: Dict[str, Any]) -> str:
        """상세 데이터 컨텍스트 구성"""
        context_parts = []
        
        # 기본 정보
        shape = basic_stats['shape']
        context_parts.append(f"데이터 크기: {shape['rows']}행 × {shape['columns']}열")
        
        # 변수 유형별 요약
        num_cols = len(basic_stats.get('numerical_columns', []))
        cat_cols = len(basic_stats.get('categorical_columns', []))
        context_parts.append(f"변수 구성: 수치형 {num_cols}개, 범주형 {cat_cols}개")
        
        # 결측치 현황
        missing_info = basic_stats.get('missing_values', {})
        total_missing = sum(missing_info.values())
        if total_missing > 0:
            missing_pct = round((total_missing / (shape['rows'] * shape['columns'])) * 100, 2)
            context_parts.append(f"결측치: 전체 {total_missing}개 ({missing_pct}%)")
        
        # 수치형 변수 요약
        if 'numerical_summary' in basic_stats:
            context_parts.append("\n수치형 변수 요약:")
            for col, stats in basic_stats['numerical_summary'].items():
                mean_val = round(stats.get('mean', 0), 2)
                std_val = round(stats.get('std', 0), 2)
                context_parts.append(f"  - {col}: 평균 {mean_val}, 표준편차 {std_val}")
        
        # 범주형 변수 요약
        if 'categorical_summary' in basic_stats:
            context_parts.append("\n범주형 변수 요약:")
            for col, counts in basic_stats['categorical_summary'].items():
                unique_count = len(counts)
                most_common = max(counts, key=counts.get) if counts else 'N/A'
                context_parts.append(f"  - {col}: {unique_count}개 범주, 최빈값 '{most_common}'")
        
        # 샘플 데이터
        context_parts.append("\n샘플 데이터 (처음 3행):")
        for i, row in enumerate(basic_stats.get('sample_data', [])[:3], 1):
            row_str = ", ".join([f"{k}={v}" for k, v in list(row.items())[:5]])
            context_parts.append(f"  {i}. {row_str}...")
        
        return "\n".join(context_parts)
    
    def _create_data_analysis_prompt(self, user_request: str, objectives: Dict[str, Any], 
                                   data_context: str, basic_stats: Dict[str, Any]) -> str:
        """데이터 분석용 프롬프트 생성"""
        prompt = f"""
당신은 데이터 과학자입니다. 사용자의 요청과 데이터를 분석하여 깊은 인사이트를 제공해주세요.

## 사용자 요청
"{user_request}"

## 분석 목표
{json.dumps(objectives, ensure_ascii=False, indent=2)}

## 데이터 정보
{data_context}

## 분석 과제
다음을 수행해주세요:

1. 사용자 요청에 맞는 데이터의 핵심 특성 파악
2. 데이터 품질 및 분석 적합성 평가
3. 주요 패턴, 트렌드, 이상치 식별
4. 분석 목표 달성을 위한 데이터 준비 방안 제시
5. 예상되는 분석 결과 및 인사이트 예측

## 응답 형식 (JSON)
```json
{{
    "data_characteristics": {{
        "key_patterns": ["데이터에서 발견한 주요 패턴들"],
        "data_distribution": "데이터 분포의 특성",
        "variable_relationships": "주요 변수 간 관계",
        "data_quality": "high|medium|low"
    }},
    "insights": {{
        "primary_findings": ["주요 발견사항들"],
        "potential_issues": ["잠재적 문제점들"],
        "interesting_patterns": ["흥미로운 패턴이나 이상치"],
        "analysis_implications": "분석에 미치는 영향"
    }},
    "analysis_readiness": {{
        "suitability_for_request": "high|medium|low",
        "required_preprocessing": ["필요한 전처리 단계들"],
        "data_limitations": ["데이터의 한계점들"],
        "recommended_approach": "추천 분석 접근법"
    }},
    "specific_observations": {{
        "target_variables_analysis": "목표 변수들의 특성",
        "predictor_variables_analysis": "예측 변수들의 특성",
        "correlation_insights": "변수 간 상관관계 인사이트",
        "outlier_impact": "이상치가 분석에 미치는 영향"
    }},
    "confidence": "high|medium|low",
    "reasoning": "분석 판단의 근거"
}}
```

사용자의 구체적인 요청을 중심으로 데이터를 해석하고, 실제 분석에 도움이 될 구체적이고 실용적인 인사이트를 제공해주세요.
"""
        
        return prompt
    
    def _parse_data_analysis_response(self, response_content: str) -> Dict[str, Any]:
        """LLM 데이터 분석 응답 파싱"""
        try:
            # JSON 블록 추출
            json_start = response_content.find('```json')
            json_end = response_content.find('```', json_start + 7)
            
            if json_start != -1 and json_end != -1:
                json_str = response_content[json_start + 7:json_end].strip()
            else:
                json_str = response_content.strip()
            
            # JSON 파싱
            parsed_response = json.loads(json_str)
            return parsed_response
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"데이터 분석 응답 JSON 파싱 실패: {e}")
            return {
                "data_characteristics": {
                    "key_patterns": ["데이터 분석을 위한 기본 패턴 식별"],
                    "data_quality": "medium"
                },
                "insights": {
                    "primary_findings": ["기본 데이터 탐색 수행"],
                    "analysis_implications": "표준 데이터 분석 접근"
                },
                "confidence": "low",
                "reasoning": "JSON 파싱 실패로 기본 분석 적용"
            }
    
    def _validate_data_analysis(self, analysis: Dict[str, Any], data: pd.DataFrame, 
                              basic_stats: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 분석 결과 검증 및 보완"""
        validated = analysis.copy()
        
        # 기본 구조 확인
        if 'data_characteristics' not in validated:
            validated['data_characteristics'] = {}
        if 'insights' not in validated:
            validated['insights'] = {}
        if 'analysis_readiness' not in validated:
            validated['analysis_readiness'] = {}
        
        # 데이터 품질 검증
        missing_pct = sum(basic_stats.get('missing_values', {}).values()) / (data.shape[0] * data.shape[1]) * 100
        
        if missing_pct > 20:
            if 'data_quality' not in validated['data_characteristics']:
                validated['data_characteristics']['data_quality'] = 'low'
        elif missing_pct > 5:
            if 'data_quality' not in validated['data_characteristics']:
                validated['data_characteristics']['data_quality'] = 'medium'
        else:
            if 'data_quality' not in validated['data_characteristics']:
                validated['data_characteristics']['data_quality'] = 'high'
        
        # 신뢰도 조정
        if validated.get('confidence') not in ['high', 'medium', 'low']:
            validated['confidence'] = 'medium'
        
        return validated
    
    def _assess_data_quality_with_llm(self, data: pd.DataFrame, 
                                    agent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """LLM을 통한 데이터 품질 평가"""
        try:
            # 품질 지표 계산
            quality_metrics = self._calculate_quality_metrics(data)
            
            # LLM을 통한 품질 해석
            quality_prompt = self._create_quality_assessment_prompt(quality_metrics, agent_analysis)
            
            response = self.llm_client.generate_response(
                quality_prompt,
                max_tokens=1000,
                temperature=0.2
            )
            
            quality_assessment = self._parse_quality_response(response.content)
            quality_assessment['metrics'] = quality_metrics
            
            return quality_assessment
            
        except Exception as e:
            self.logger.error(f"LLM 품질 평가 오류: {e}")
            return self._fallback_quality_assessment(data)
    
    def _calculate_quality_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """데이터 품질 지표 계산"""
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        
        # 중복행 확인
        duplicate_rows = data.duplicated().sum()
        
        # 수치형 변수 이상치 간단 추정
        numerical_cols = data.select_dtypes(include=['number']).columns
        outlier_count = 0
        
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_count += outliers
        
        return {
            'missing_percentage': round((missing_cells / total_cells) * 100, 2),
            'duplicate_rows': int(duplicate_rows),
            'duplicate_percentage': round((duplicate_rows / data.shape[0]) * 100, 2),
            'outlier_count': int(outlier_count),
            'data_consistency': 'high' if missing_cells == 0 and duplicate_rows == 0 else 'medium',
            'completeness_score': round(((total_cells - missing_cells) / total_cells) * 100, 2)
        }
    
    def _create_quality_assessment_prompt(self, quality_metrics: Dict[str, Any], 
                                        agent_analysis: Dict[str, Any]) -> str:
        """품질 평가 프롬프트 생성"""
        return f"""
데이터 품질 지표를 해석하고 분석에 미치는 영향을 평가해주세요.

## 품질 지표
- 결측치 비율: {quality_metrics['missing_percentage']}%
- 중복행 비율: {quality_metrics['duplicate_percentage']}%
- 추정 이상치 개수: {quality_metrics['outlier_count']}개
- 완성도 점수: {quality_metrics['completeness_score']}%

## 이전 분석 결과
{json.dumps(agent_analysis, ensure_ascii=False, indent=2)}

다음 형식으로 응답해주세요:

```json
{{
    "overall_quality": "excellent|good|fair|poor",
    "quality_issues": ["주요 품질 이슈들"],
    "impact_on_analysis": "분석에 미치는 영향",
    "recommendations": ["품질 개선 권장사항들"],
    "proceed_with_analysis": "yes|caution|no"
}}
```
"""
    
    def _parse_quality_response(self, response_content: str) -> Dict[str, Any]:
        """품질 평가 응답 파싱"""
        try:
            json_start = response_content.find('```json')
            json_end = response_content.find('```', json_start + 7)
            
            if json_start != -1 and json_end != -1:
                json_str = response_content[json_start + 7:json_end].strip()
            else:
                json_str = response_content.strip()
            
            return json.loads(json_str)
            
        except json.JSONDecodeError:
            return {
                "overall_quality": "fair",
                "quality_issues": ["자동 품질 평가 수행"],
                "impact_on_analysis": "표준 분석 절차 적용",
                "proceed_with_analysis": "yes"
            }
    
    def _generate_recommendations_with_llm(self, data: pd.DataFrame, user_request: str,
                                         objectives: Dict[str, Any], agent_analysis: Dict[str, Any],
                                         quality_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """LLM을 통한 분석 추천사항 생성"""
        try:
            recommendations_prompt = f"""
사용자 요청과 데이터 분석 결과를 바탕으로 최적의 분석 전략을 추천해주세요.

## 사용자 요청
"{user_request}"

## 데이터 분석 결과
{json.dumps(agent_analysis, ensure_ascii=False, indent=2)}

## 품질 평가 결과
{json.dumps(quality_assessment, ensure_ascii=False, indent=2)}

다음 형식으로 응답해주세요:

```json
{{
    "recommended_analyses": ["추천 분석 방법들"],
    "preprocessing_steps": ["필요한 전처리 단계들"],
    "potential_challenges": ["예상되는 분석 어려움들"],
    "success_factors": ["분석 성공을 위한 요소들"],
    "alternative_approaches": ["대안 접근 방법들"],
    "expected_insights": ["기대할 수 있는 인사이트들"]
}}
```
"""
            
            response = self.llm_client.generate_response(
                recommendations_prompt,
                max_tokens=1200,
                temperature=0.3
            )
            
            return self._parse_recommendations_response(response.content)
            
        except Exception as e:
            self.logger.error(f"추천사항 생성 오류: {e}")
            return self._fallback_recommendations()
    
    def _parse_recommendations_response(self, response_content: str) -> Dict[str, Any]:
        """추천사항 응답 파싱"""
        try:
            json_start = response_content.find('```json')
            json_end = response_content.find('```', json_start + 7)
            
            if json_start != -1 and json_end != -1:
                json_str = response_content[json_start + 7:json_end].strip()
            else:
                json_str = response_content.strip()
            
            return json.loads(json_str)
            
        except json.JSONDecodeError:
            return self._fallback_recommendations()
    
    def _build_enhanced_understanding(self, data: pd.DataFrame, agent_analysis: Dict[str, Any],
                                    quality_assessment: Dict[str, Any], 
                                    recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """향상된 데이터 이해 구성"""
        return {
            'data_summary': {
                'shape': data.shape,
                'columns': list(data.columns),
                'quality_score': quality_assessment.get('overall_quality', 'fair')
            },
            'analysis_insights': agent_analysis.get('insights', {}),
            'readiness_assessment': agent_analysis.get('analysis_readiness', {}),
            'quality_overview': quality_assessment,
            'next_steps': recommendations.get('recommended_analyses', []),
            'preprocessing_needed': recommendations.get('preprocessing_steps', []),
            'confidence_level': agent_analysis.get('confidence', 'medium')
        }
    
    def _fallback_data_analysis(self, data: pd.DataFrame, basic_stats: Dict[str, Any]) -> Dict[str, Any]:
        """백업 데이터 분석 (LLM 실패 시)"""
        return {
            "data_characteristics": {
                "key_patterns": ["기본 데이터 탐색"],
                "data_quality": "medium"
            },
            "insights": {
                "primary_findings": [f"{data.shape[0]}행 {data.shape[1]}열의 데이터"],
                "analysis_implications": "표준 통계 분석 수행 가능"
            },
            "analysis_readiness": {
                "suitability_for_request": "medium",
                "recommended_approach": "기술통계 및 시각화"
            },
            "confidence": "low",
            "reasoning": "LLM 분석 실패로 기본 접근법 적용"
        }
    
    def _fallback_quality_assessment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """백업 품질 평가"""
        missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
        
        return {
            "overall_quality": "good" if missing_pct < 5 else "fair",
            "quality_issues": [f"결측치 {missing_pct:.1f}%"],
            "impact_on_analysis": "표준 전처리 후 분석 가능",
            "proceed_with_analysis": "yes",
            "metrics": {
                "missing_percentage": round(missing_pct, 2),
                "completeness_score": round(100 - missing_pct, 2)
            }
        }
    
    def _fallback_recommendations(self) -> Dict[str, Any]:
        """백업 추천사항"""
        return {
            "recommended_analyses": ["기술통계분석", "데이터 시각화"],
            "preprocessing_steps": ["결측치 처리", "이상치 확인"],
            "potential_challenges": ["데이터 품질 이슈"],
            "success_factors": ["적절한 전처리"],
            "expected_insights": ["데이터 기본 특성 파악"]
        }
    
    def get_step_info(self) -> Dict[str, Any]:
        """단계 정보 반환"""
        base_info = super().get_step_info()
        base_info.update({
            'description': 'LLM Agent 기반 데이터 심층 분석',
            'input_requirements': ['user_request', 'analysis_objectives', 'data_understanding'],
            'output_provides': [
                'agent_data_analysis', 'data_insights', 'quality_assessment', 
                'analysis_recommendations', 'enhanced_understanding'
            ],
            'capabilities': [
                'LLM 기반 데이터 해석', '지능적 품질 평가', '맞춤형 분석 추천', 
                '자동 인사이트 도출'
            ]
        })
        return base_info


