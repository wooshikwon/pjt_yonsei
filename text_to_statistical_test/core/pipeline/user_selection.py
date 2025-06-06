"""
User Selection Pipeline

5단계: 사용자 피드백 기반 분석 방식 구체화
사용자는 LLM의 제안을 검토하고, 필요한 경우 추가적인 요구사항이나 선호하는 분석 방향을 제시합니다.
시스템은 이를 반영하여 최종 분석 계획을 확정합니다.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .base_pipeline_step import BasePipelineStep, PipelineStepRegistry
from services.llm.llm_client import LLMClient
from services.llm.prompt_engine import PromptEngine


class UserSelectionStep(BasePipelineStep):
    """5단계: 사용자 피드백 기반 분석 방식 구체화"""
    
    def __init__(self, conversation_history=None):
        """
        UserSelectionStep 초기화
        
        Args:
            conversation_history: 대화 이력 관리자 (외부에서 주입)
        """
        super().__init__("사용자 피드백 기반 분석 방식 구체화", 5)
        self.conversation_history = conversation_history  # 외부에서 주입받음
        self.llm_client = LLMClient()
        self.prompt_engine = PromptEngine()
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        입력 데이터 유효성 검증
        
        Args:
            input_data: 4단계에서 전달받은 데이터
            
        Returns:
            bool: 유효성 검증 결과
        """
        required_fields = [
            'analysis_proposals', 'statistical_context', 'domain_insights',
            'execution_plan', 'visualization_suggestions'
        ]
        return all(field in input_data for field in required_fields)
    
    def get_expected_output_schema(self) -> Dict[str, Any]:
        """
        예상 출력 스키마 반환
        
        Returns:
            Dict[str, Any]: 출력 데이터 스키마
        """
        return {
            'selected_analysis': {
                'method': dict,
                'parameters': dict,
                'customizations': dict
            },
            'analysis_plan': {
                'steps': list,
                'validations': list,
                'adjustments': list
            },
            'user_preferences': {
                'visualization_preferences': dict,
                'reporting_preferences': dict,
                'additional_requirements': list
            },
            'conversation_summary': {
                'key_decisions': list,
                'clarifications': list,
                'final_confirmations': list
            },
            'execution_context': {
                'parameters': dict,
                'constraints': dict,
                'special_instructions': list
            }
        }
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        사용자 피드백 기반 분석 방식 구체화 파이프라인 실행
        
        Args:
            input_data: 파이프라인 실행 컨텍스트
                - analysis_proposals: 분석 제안
                - statistical_context: 통계적 컨텍스트
                - domain_insights: 도메인 인사이트
                - execution_plan: 실행 계획
                - visualization_suggestions: 시각화 제안
            
        Returns:
            Dict: 실행 결과
        """
        self.logger.info("5단계: 사용자 피드백 기반 분석 방식 구체화 시작")
        
        try:
            # 1. 분석 제안 표시 및 사용자 선택 처리
            selected_analysis = self._handle_analysis_selection(input_data)
            
            # 2. 선택된 분석에 대한 상세 설정
            analysis_plan = self._refine_analysis_plan(
                selected_analysis, input_data
            )
            
            # 3. 사용자 선호도 수집
            user_preferences = self._collect_user_preferences(
                selected_analysis, input_data
            )
            
            # 4. 대화 내용 요약
            conversation_summary = self._summarize_conversation()
            
            # 5. 실행 컨텍스트 구성
            execution_context = self._build_execution_context(
                selected_analysis, analysis_plan, user_preferences
            )
            
            self.logger.info("분석 방식 구체화 완료")
            
            return {
                'selected_analysis': selected_analysis,
                'analysis_plan': analysis_plan,
                'user_preferences': user_preferences,
                'conversation_summary': conversation_summary,
                'execution_context': execution_context,
                'success_message': "✅ 분석 방식이 확정되었습니다."
            }
                
        except Exception as e:
            self.logger.error(f"분석 방식 구체화 파이프라인 오류: {e}")
            return {
                'error': True,
                'error_message': str(e),
                'error_type': 'selection_error'
            }
    
    def _handle_analysis_selection(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """분석 제안 표시 및 사용자 선택 처리"""
        # 1. 제안된 분석 방법 표시
        self._display_analysis_proposals(input_data['analysis_proposals'])
        
        # 2. 사용자 선택 처리
        selected_method = self._process_user_selection(
            input_data['analysis_proposals']
        )
        
        # 3. 선택된 방법 상세화
        detailed_selection = self._detail_selected_method(
            selected_method,
            input_data['statistical_context'],
            input_data['domain_insights']
        )
        
        return detailed_selection
    
    def _display_analysis_proposals(self, proposals: Dict[str, Any]) -> None:
        """분석 제안 표시"""
        print("\n" + "="*60)
        print("📊 제안된 분석 방법")
        print("="*60)
        
        # 추천 방법 표시
        print("\n🌟 추천 분석 방법:")
        for i, method in enumerate(proposals['recommended_methods'], 1):
            print(f"\n{i}. {method['name']}")
            print(f"   📝 설명: {method['description']}")
            print(f"   ✅ 장점: {', '.join(method.get('advantages', []))}")
            if method.get('limitations'):
                print(f"   ⚠️ 제한사항: {', '.join(method['limitations'])}")
        
        # 대체 방법 표시
        if proposals['alternative_methods']:
            print("\n📌 대체 분석 방법:")
            for i, method in enumerate(proposals['alternative_methods'], 1):
                print(f"\n{i}. {method['name']}")
                print(f"   📝 설명: {method['description']}")
    
    def _process_user_selection(self, proposals: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 선택 처리"""
        # 사용자 입력 대기 (실제 구현에서는 UI/CLI 통합 필요)
        print("\n💡 분석 방법을 선택해주세요 (번호 입력):")
        
        # 실제 사용자 입력 처리 로직 구현
        try:
            # 추천 방법 선택지 표시
            recommended_methods = proposals['recommended_methods']
            while True:
                try:
                    # 사용자 입력 받기
                    user_input = input(f"선택 (1-{len(recommended_methods)}): ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        # 기본 선택으로 첫 번째 방법 반환
                        return recommended_methods[0]
                    
                    # 숫자 입력 처리
                    choice_idx = int(user_input) - 1
                    if 0 <= choice_idx < len(recommended_methods):
                        selected_method = recommended_methods[choice_idx]
                        print(f"\n✅ '{selected_method['name']}' 방법이 선택되었습니다.")
                        return selected_method
                    else:
                        print(f"❌ 1부터 {len(recommended_methods)} 사이의 숫자를 입력해주세요.")
                        
                except ValueError:
                    print("❌ 유효한 숫자를 입력해주세요.")
                except KeyboardInterrupt:
                    print("\n\n🔄 기본 분석 방법을 선택합니다.")
                    return recommended_methods[0]
                    
        except Exception as e:
            self.logger.warning(f"사용자 입력 처리 중 오류: {e}, 기본 방법 선택")
            # 오류 발생 시 첫 번째 추천 방법을 자동 선택
            return recommended_methods[0]
    
    def _detail_selected_method(self, selected_method: Dict[str, Any],
                              statistical_context: Dict[str, Any],
                              domain_insights: Dict[str, Any]) -> Dict[str, Any]:
        """선택된 방법 상세화"""
        # LLM을 사용하여 선택된 방법 상세화
        prompt = self.prompt_engine.create_method_detailing_prompt(
            method=selected_method,
            statistical_context=statistical_context,
            domain_insights=domain_insights
        )
        
        llm_response = self.llm_client.generate(prompt)
        
        # 응답 파싱 및 구조화
        detailed_method = self._parse_method_details(llm_response)
        
        return {
            'method': selected_method,
            'parameters': detailed_method.get('parameters', {}),
            'customizations': detailed_method.get('customizations', {})
        }
    
    def _refine_analysis_plan(self, selected_analysis: Dict[str, Any],
                            input_data: Dict[str, Any]) -> Dict[str, Any]:
        """선택된 분석에 대한 상세 계획 수립"""
        # 1. 기본 실행 단계 정의
        execution_steps = self._define_execution_steps(
            selected_analysis, input_data['execution_plan']
        )
        
        # 2. 필요한 검증 단계 식별
        validation_steps = self._identify_validation_steps(
            selected_analysis, input_data['statistical_context']
        )
        
        # 3. 잠재적 조정사항 정의
        adjustment_options = self._define_adjustment_options(
            selected_analysis, input_data['domain_insights']
        )
        
        return {
            'steps': execution_steps,
            'validations': validation_steps,
            'adjustments': adjustment_options
        }
    
    def _collect_user_preferences(self, selected_analysis: Dict[str, Any],
                                input_data: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 선호도 수집"""
        # 1. 시각화 선호도 수집
        viz_preferences = self._collect_visualization_preferences(
            input_data['visualization_suggestions']
        )
        
        # 2. 보고서 형식 선호도 수집
        reporting_preferences = self._collect_reporting_preferences()
        
        # 3. 추가 요구사항 수집
        additional_requirements = self._collect_additional_requirements(
            selected_analysis
        )
        
        return {
            'visualization_preferences': viz_preferences,
            'reporting_preferences': reporting_preferences,
            'additional_requirements': additional_requirements
        }
    
    def _summarize_conversation(self) -> Dict[str, Any]:
        """대화 내용 요약"""
        try:
            # conversation_history가 주입되지 않은 경우 기본 동작
            if self.conversation_history is None:
                self.logger.warning("ConversationHistory가 주입되지 않았습니다. 기본 요약을 반환합니다.")
                return {
                    'key_decisions': ["분석 방법이 선택되었습니다"],
                    'clarifications': [],
                    'final_confirmations': ["사용자가 최종 분석 방법을 확정했습니다"]
                }
            
            # 현재 세션의 대화 이력 가져오기
            session_id = self.conversation_history.get_current_session_id() if hasattr(self.conversation_history, 'get_current_session_id') else None
            if not session_id:
                self.logger.warning("활성 세션이 없습니다.")
                return {
                    'key_decisions': [],
                    'clarifications': [],
                    'final_confirmations': []
                }
            
            # 대화 이력 요약 생성
            history = self.conversation_history.get_session_history(session_id, last_n_turns=10)
            
            if not history:
                return {
                    'key_decisions': [],
                    'clarifications': [],
                    'final_confirmations': []
                }
            
            # LLM을 사용한 대화 요약
            summary_prompt = self.prompt_engine.create_conversation_summary_prompt(history)
            summary_response = self.llm_client.generate_response(summary_prompt)
            
            # 요약 파싱
            parsed_summary = self._parse_conversation_summary(summary_response)
            
            return parsed_summary
            
        except Exception as e:
            self.logger.error(f"대화 요약 오류: {e}")
            return {
                'key_decisions': [],
                'clarifications': [],
                'final_confirmations': []
            }
    
    def _build_execution_context(self, selected_analysis: Dict[str, Any],
                               analysis_plan: Dict[str, Any],
                               user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """실행 컨텍스트 구성"""
        # 1. 분석 파라미터 구성
        parameters = self._build_analysis_parameters(
            selected_analysis, analysis_plan
        )
        
        # 2. 제약사항 정의
        constraints = self._define_execution_constraints(
            selected_analysis, user_preferences
        )
        
        # 3. 특별 지침 작성
        special_instructions = self._create_special_instructions(
            selected_analysis, user_preferences
        )
        
        return {
            'parameters': parameters,
            'constraints': constraints,
            'special_instructions': special_instructions
        }
    
    def _parse_method_details(self, llm_response: str) -> Dict[str, Any]:
        """LLM 응답에서 방법 상세 정보 파싱"""
        from services.llm.llm_response_parser import LLMResponseParser, ResponseType
        
        try:
            parser = LLMResponseParser()
            parsed = parser.parse_response(llm_response, expected_type=ResponseType.JSON)
            
            if parsed.confidence > 0.5 and isinstance(parsed.content, dict):
                return parsed.content
            else:
                # JSON 파싱 실패 시 텍스트에서 정보 추출
                return self._extract_method_details_from_text(llm_response)
                
        except Exception as e:
            self.logger.warning(f"방법 상세 정보 파싱 오류: {e}")
            return {
                'parameters': {},
                'customizations': {},
                'notes': llm_response
            }
    
    def _define_execution_steps(self, selected_analysis: Dict[str, Any],
                              execution_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """실행 단계 정의"""
        try:
            method_name = selected_analysis.get('method', {}).get('name', '')
            analysis_type = selected_analysis.get('method', {}).get('type', '')
            
            # 기본 실행 단계 템플릿
            base_steps = [
                {
                    'step_id': 'data_preparation',
                    'name': '데이터 준비',
                    'description': '데이터 로드 및 기본 전처리',
                    'required': True,
                    'estimated_time': '1-2분'
                },
                {
                    'step_id': 'assumption_check',
                    'name': '가정 검증',
                    'description': '통계적 가정 확인',
                    'required': True,
                    'estimated_time': '2-3분'
                }
            ]
            
            # 분석 유형별 특화 단계 추가
            if 't_test' in analysis_type.lower() or 't-test' in method_name.lower():
                base_steps.extend([
                    {
                        'step_id': 'normality_test',
                        'name': '정규성 검정',
                        'description': 'Shapiro-Wilk 또는 K-S 검정',
                        'required': True,
                        'estimated_time': '1분'
                    },
                    {
                        'step_id': 't_test_execution',
                        'name': 'T-검정 실행',
                        'description': '독립/대응표본 t-검정 수행',
                        'required': True,
                        'estimated_time': '1분'
                    }
                ])
            elif 'anova' in analysis_type.lower() or 'anova' in method_name.lower():
                base_steps.extend([
                    {
                        'step_id': 'homoscedasticity_test',
                        'name': '등분산성 검정',
                        'description': 'Levene 또는 Bartlett 검정',
                        'required': True,
                        'estimated_time': '1분'
                    },
                    {
                        'step_id': 'anova_execution',
                        'name': 'ANOVA 실행',
                        'description': '일원 또는 이원 분산분석',
                        'required': True,
                        'estimated_time': '2분'
                    },
                    {
                        'step_id': 'post_hoc',
                        'name': '사후 검정',
                        'description': 'Tukey HSD 등 다중비교',
                        'required': False,
                        'estimated_time': '1-2분'
                    }
                ])
            elif 'correlation' in analysis_type.lower():
                base_steps.extend([
                    {
                        'step_id': 'correlation_analysis',
                        'name': '상관분석',
                        'description': 'Pearson 또는 Spearman 상관분석',
                        'required': True,
                        'estimated_time': '1분'
                    }
                ])
            elif 'regression' in analysis_type.lower():
                base_steps.extend([
                    {
                        'step_id': 'regression_analysis',
                        'name': '회귀분석',
                        'description': '선형 또는 로지스틱 회귀분석',
                        'required': True,
                        'estimated_time': '2-3분'
                    },
                    {
                        'step_id': 'residual_analysis',
                        'name': '잔차분석',
                        'description': '모델 진단 및 가정 확인',
                        'required': True,
                        'estimated_time': '1-2분'
                    }
                ])
            
            # 공통 마무리 단계
            base_steps.extend([
                {
                    'step_id': 'result_interpretation',
                    'name': '결과 해석',
                    'description': '통계적 결과 해석 및 의미 도출',
                    'required': True,
                    'estimated_time': '2-3분'
                },
                {
                    'step_id': 'visualization',
                    'name': '시각화',
                    'description': '결과 차트 및 그래프 생성',
                    'required': False,
                    'estimated_time': '1-2분'
                }
            ])
            
            return base_steps
            
        except Exception as e:
            self.logger.error(f"실행 단계 정의 오류: {e}")
            return [
                {
                    'step_id': 'basic_analysis',
                    'name': '기본 분석',
                    'description': '선택된 통계 분석 수행',
                    'required': True,
                    'estimated_time': '5분'
                }
            ]
    
    def _identify_validation_steps(self, selected_analysis: Dict[str, Any],
                                 statistical_context: Dict[str, Any]) -> List[str]:
        """검증 단계 식별"""
        try:
            method_info = selected_analysis.get('method', {})
            analysis_type = method_info.get('type', '').lower()
            
            validation_steps = []
            
            # 데이터 품질 검증 (모든 분석에 공통)
            validation_steps.extend([
                'data_completeness_check',  # 데이터 완성도 확인
                'outlier_detection',        # 이상치 탐지
                'data_type_validation'      # 데이터 타입 검증
            ])
            
            # 분석별 특화 검증
            if any(test in analysis_type for test in ['t_test', 'anova', 'regression']):
                validation_steps.extend([
                    'normality_assumption',      # 정규성 가정
                    'independence_assumption'    # 독립성 가정
                ])
            
            if 'anova' in analysis_type or 'regression' in analysis_type:
                validation_steps.append('homoscedasticity_assumption')  # 등분산성 가정
            
            if 'regression' in analysis_type:
                validation_steps.extend([
                    'linearity_assumption',      # 선형성 가정
                    'multicollinearity_check'    # 다중공선성 확인
                ])
            
            if 'correlation' in analysis_type:
                validation_steps.extend([
                    'relationship_linearity',    # 관계의 선형성
                    'influential_points_check'   # 영향점 확인
                ])
            
            # 샘플 크기 관련 검증
            sample_size = statistical_context.get('sample_size', 0)
            if sample_size < 30:
                validation_steps.append('small_sample_considerations')
            if sample_size < 5:
                validation_steps.append('very_small_sample_warning')
            
            return validation_steps
            
        except Exception as e:
            self.logger.error(f"검증 단계 식별 오류: {e}")
            return ['basic_data_validation', 'assumption_check']
    
    def _define_adjustment_options(self, selected_analysis: Dict[str, Any],
                                 domain_insights: Dict[str, Any]) -> List[str]:
        """조정 옵션 정의"""
        try:
            method_info = selected_analysis.get('method', {})
            analysis_type = method_info.get('type', '').lower()
            
            adjustment_options = []
            
            # 기본 조정 옵션
            adjustment_options.extend([
                'significance_level_adjustment',  # 유의수준 조정 (0.05, 0.01, 0.001)
                'effect_size_reporting',          # 효과크기 보고 옵션
                'confidence_interval_level'       # 신뢰구간 수준 조정
            ])
            
            # 분석별 특화 조정 옵션
            if 't_test' in analysis_type:
                adjustment_options.extend([
                    'equal_variance_assumption',  # 등분산 가정 여부
                    'one_vs_two_tailed_test',    # 단측/양측 검정 선택
                    'welch_correction'           # Welch 보정 적용
                ])
            
            elif 'anova' in analysis_type:
                adjustment_options.extend([
                    'post_hoc_correction_method', # 사후검정 보정 방법
                    'effect_size_calculation',    # 효과크기 계산 방법 (eta², omega²)
                    'assumption_violation_handling' # 가정 위배 시 대안
                ])
            
            elif 'regression' in analysis_type:
                adjustment_options.extend([
                    'variable_selection_method',  # 변수 선택 방법
                    'regularization_options',     # 정규화 옵션 (Ridge, Lasso)
                    'cross_validation_folds',     # 교차검증 폴드 수
                    'outlier_handling_strategy'   # 이상치 처리 전략
                ])
            
            elif 'correlation' in analysis_type:
                adjustment_options.extend([
                    'correlation_method',         # 상관계수 방법 (Pearson, Spearman, Kendall)
                    'partial_correlation_control', # 편상관 제어변수
                    'bootstrap_confidence_interval' # 부트스트랩 신뢰구간
                ])
            
            # 도메인별 특화 옵션
            domain = domain_insights.get('domain', '').lower()
            if 'medical' in domain or 'health' in domain:
                adjustment_options.extend([
                    'clinical_significance_threshold',
                    'survival_analysis_considerations'
                ])
            elif 'business' in domain or 'marketing' in domain:
                adjustment_options.extend([
                    'business_impact_weighting',
                    'cost_benefit_considerations'
                ])
            elif 'psychology' in domain or 'social' in domain:
                adjustment_options.extend([
                    'cultural_context_adjustment',
                    'demographic_stratification'
                ])
            
            return adjustment_options
            
        except Exception as e:
            self.logger.error(f"조정 옵션 정의 오류: {e}")
            return ['significance_level_adjustment', 'basic_options']
    
    def _collect_visualization_preferences(self, viz_suggestions: Dict[str, Any]) -> Dict[str, Any]:
        """시각화 선호도 수집"""
        try:
            print("\n📊 시각화 옵션을 선택해주세요:")
            
            # 기본 시각화 옵션
            viz_options = viz_suggestions.get('suggested_plots', [])
            selected_plots = []
            
            # 사용자에게 시각화 옵션 표시
            for i, plot in enumerate(viz_options, 1):
                print(f"{i}. {plot.get('name', 'Unknown Plot')} - {plot.get('description', '')}")
            
            print("\n선택할 시각화 번호를 입력하세요 (여러 개 선택 시 쉼표로 구분, 예: 1,3,4):")
            
            try:
                user_input = input("시각화 선택: ").strip()
                if user_input:
                    choices = [int(x.strip()) for x in user_input.split(',')]
                    selected_plots = [viz_options[i-1] for i in choices if 1 <= i <= len(viz_options)]
                else:
                    # 기본 시각화 선택
                    selected_plots = viz_options[:2] if len(viz_options) >= 2 else viz_options
            except (ValueError, IndexError, KeyboardInterrupt):
                print("기본 시각화 옵션을 선택합니다.")
                selected_plots = viz_options[:2] if len(viz_options) >= 2 else viz_options
            
            # 시각화 스타일 선택
            print("\n시각화 스타일을 선택해주세요:")
            styles = ['간단한 스타일', '상세한 스타일', '학술적 스타일', '비즈니스 스타일']
            for i, style in enumerate(styles, 1):
                print(f"{i}. {style}")
            
            try:
                style_choice = int(input("스타일 선택 (1-4): ").strip())
                selected_style = styles[style_choice - 1] if 1 <= style_choice <= 4 else styles[0]
            except (ValueError, KeyboardInterrupt):
                selected_style = styles[0]
            
            return {
                'selected_plots': selected_plots,
                'style': selected_style,
                'interactive': True,  # 기본적으로 인터랙티브 차트
                'color_scheme': 'default',
                'export_formats': ['png', 'html']
            }
            
        except Exception as e:
            self.logger.error(f"시각화 선호도 수집 오류: {e}")
            return {
                'selected_plots': viz_suggestions.get('suggested_plots', [])[:2],
                'style': '간단한 스타일',
                'interactive': True,
                'color_scheme': 'default',
                'export_formats': ['png']
            }
    
    def _collect_reporting_preferences(self) -> Dict[str, Any]:
        """보고서 형식 선호도 수집"""
        try:
            print("\n📋 보고서 형식을 선택해주세요:")
            
            # 보고서 형식 옵션
            report_formats = [
                {'name': '간단 요약', 'description': '핵심 결과만 포함'},
                {'name': '상세 보고서', 'description': '분석 과정과 해석 포함'},
                {'name': '기술 보고서', 'description': '통계적 세부사항 포함'},
                {'name': '비즈니스 보고서', 'description': '실무진을 위한 형식'}
            ]
            
            for i, fmt in enumerate(report_formats, 1):
                print(f"{i}. {fmt['name']} - {fmt['description']}")
            
            try:
                choice = int(input("보고서 형식 선택 (1-4): ").strip())
                selected_format = report_formats[choice - 1] if 1 <= choice <= 4 else report_formats[1]
            except (ValueError, KeyboardInterrupt):
                selected_format = report_formats[1]  # 기본: 상세 보고서
            
            # 출력 형식 선택
            print("\n출력 형식을 선택해주세요 (여러 개 선택 가능):")
            output_formats = ['HTML', 'PDF', 'Markdown', 'Excel']
            for i, fmt in enumerate(output_formats, 1):
                print(f"{i}. {fmt}")
            
            try:
                output_input = input("출력 형식 선택 (예: 1,3): ").strip()
                if output_input:
                    choices = [int(x.strip()) for x in output_input.split(',')]
                    selected_outputs = [output_formats[i-1] for i in choices if 1 <= i <= len(output_formats)]
                else:
                    selected_outputs = ['HTML']
            except (ValueError, KeyboardInterrupt):
                selected_outputs = ['HTML']
            
            return {
                'format': selected_format,
                'output_formats': selected_outputs,
                'include_code': True,
                'include_data_summary': True,
                'include_assumptions': True,
                'include_interpretation': True,
                'language': 'korean'
            }
            
        except Exception as e:
            self.logger.error(f"보고서 선호도 수집 오류: {e}")
            return {
                'format': {'name': '상세 보고서', 'description': '분석 과정과 해석 포함'},
                'output_formats': ['HTML'],
                'include_code': True,
                'include_data_summary': True,
                'include_assumptions': True,
                'include_interpretation': True,
                'language': 'korean'
            }
    
    def _collect_additional_requirements(self, selected_analysis: Dict[str, Any]) -> List[str]:
        """추가 요구사항 수집"""
        try:
            print("\n📝 추가 요구사항이 있으시면 입력해주세요 (없으면 Enter):")
            
            requirements = []
            
            # 일반적인 추가 요구사항 옵션 제시
            common_requirements = [
                '특정 변수 간 상호작용 효과 분석',
                '서브그룹 분석 (성별, 연령대별 등)',
                '민감도 분석 (outlier 제거 후 재분석)',
                '효과크기의 실질적 의미 해석',
                '비즈니스 임팩트 추정',
                '추가 시각화 (heatmap, 3D plot 등)',
                '결과의 통계적 검정력 분석'
            ]
            
            print("\n일반적인 추가 요구사항:")
            for i, req in enumerate(common_requirements, 1):
                print(f"{i}. {req}")
            
            # 사용자 직접 입력
            try:
                custom_input = input("\n직접 입력하거나 위 번호 선택 (예: 1,3 또는 직접 입력): ").strip()
                
                if custom_input:
                    # 숫자 입력인지 확인
                    if ',' in custom_input or custom_input.isdigit():
                        try:
                            choices = [int(x.strip()) for x in custom_input.split(',')]
                            requirements = [common_requirements[i-1] for i in choices 
                                          if 1 <= i <= len(common_requirements)]
                        except (ValueError, IndexError):
                            # 숫자가 아니면 직접 입력으로 처리
                            requirements = [custom_input]
                    else:
                        # 직접 입력 텍스트
                        requirements = [custom_input]
                
                # 추가 요구사항이 있는지 확인
                if requirements:
                    print("\n추가로 더 입력하시겠습니까? (없으면 Enter)")
                    additional = input("추가 요구사항: ").strip()
                    if additional:
                        requirements.append(additional)
                        
            except KeyboardInterrupt:
                print("\n추가 요구사항 입력을 건너뜁니다.")
            
            return requirements
            
        except Exception as e:
            self.logger.error(f"추가 요구사항 수집 오류: {e}")
            return []
    
    def _parse_conversation_summary(self, llm_response: str) -> Dict[str, Any]:
        """LLM 응답에서 대화 요약 파싱"""
        from services.llm.llm_response_parser import LLMResponseParser, ResponseType
        
        try:
            parser = LLMResponseParser()
            parsed = parser.parse_response(llm_response, expected_type=ResponseType.JSON)
            
            if parsed.confidence > 0.5 and isinstance(parsed.content, dict):
                return parsed.content
            else:
                # JSON 파싱 실패 시 텍스트에서 정보 추출
                return self._extract_summary_from_text(llm_response)
                
        except Exception as e:
            self.logger.warning(f"대화 요약 파싱 오류: {e}")
            return {
                'key_decisions': ['분석 방법 선택 완료'],
                'clarifications': [],
                'final_confirmations': ['사용자 선택사항 확정'],
                'raw_summary': llm_response
            }
    
    def _build_analysis_parameters(self, selected_analysis: Dict[str, Any],
                                 analysis_plan: Dict[str, Any]) -> Dict[str, Any]:
        """분석 파라미터 구성"""
        try:
            method_info = selected_analysis.get('method', {})
            analysis_type = method_info.get('type', '').lower()
            
            # 기본 파라미터
            parameters = {
                'alpha': 0.05,  # 기본 유의수준
                'confidence_level': 0.95,  # 기본 신뢰수준
                'missing_value_handling': 'listwise_deletion',
                'outlier_handling': 'identify_only'
            }
            
            # 분석별 특화 파라미터
            if 't_test' in analysis_type:
                parameters.update({
                    'equal_var': True,  # 등분산 가정
                    'alternative': 'two-sided',  # 양측 검정
                    'paired': False  # 독립표본 기본
                })
            
            elif 'anova' in analysis_type:
                parameters.update({
                    'post_hoc_method': 'tukey',
                    'effect_size_method': 'eta_squared',
                    'correction_method': 'bonferroni'
                })
            
            elif 'correlation' in analysis_type:
                parameters.update({
                    'method': 'pearson',  # 기본 피어슨
                    'alternative': 'two-sided'
                })
            
            elif 'regression' in analysis_type:
                parameters.update({
                    'fit_intercept': True,
                    'normalize': False,
                    'cv_folds': 5,
                    'feature_selection': 'none'
                })
            
            # 사용자 커스터마이제이션 적용
            customizations = selected_analysis.get('customizations', {})
            parameters.update(customizations)
            
            return parameters
            
        except Exception as e:
            self.logger.error(f"분석 파라미터 구성 오류: {e}")
            return {'alpha': 0.05, 'confidence_level': 0.95}
    
    def _define_execution_constraints(self, selected_analysis: Dict[str, Any],
                                    user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """실행 제약사항 정의"""
        try:
            constraints = {
                'max_execution_time': 300,  # 최대 5분
                'max_memory_usage': 1024,   # 최대 1GB
                'allowed_file_operations': ['read_csv', 'save_plot', 'save_report'],
                'restricted_imports': ['os', 'subprocess', 'sys'],
                'safe_mode': True
            }
            
            # 분석 복잡도에 따른 제약사항 조정
            method_info = selected_analysis.get('method', {})
            complexity = method_info.get('complexity', 'medium')
            
            if complexity == 'high':
                constraints['max_execution_time'] = 600  # 10분
                constraints['max_memory_usage'] = 2048  # 2GB
            elif complexity == 'low':
                constraints['max_execution_time'] = 120  # 2분
                constraints['max_memory_usage'] = 512   # 512MB
            
            # 사용자 선호도 반영
            report_prefs = user_preferences.get('reporting_preferences', {})
            if 'PDF' in report_prefs.get('output_formats', []):
                constraints['allowed_file_operations'].append('save_pdf')
            if 'Excel' in report_prefs.get('output_formats', []):
                constraints['allowed_file_operations'].extend(['save_excel', 'read_excel'])
            
            return constraints
            
        except Exception as e:
            self.logger.error(f"실행 제약사항 정의 오류: {e}")
            return {
                'max_execution_time': 300,
                'max_memory_usage': 1024,
                'safe_mode': True
            }
    
    def _create_special_instructions(self, selected_analysis: Dict[str, Any],
                                   user_preferences: Dict[str, Any]) -> List[str]:
        """특별 지침 작성"""
        try:
            instructions = []
            
            # 기본 지침
            instructions.extend([
                "모든 가정을 명시적으로 확인하고 보고하세요",
                "결과 해석 시 통계적 유의성과 실질적 유의성을 구분하세요",
                "시각화는 명확하고 이해하기 쉽게 작성하세요"
            ])
            
            # 분석별 특별 지침
            method_info = selected_analysis.get('method', {})
            analysis_type = method_info.get('type', '').lower()
            
            if 't_test' in analysis_type:
                instructions.extend([
                    "정규성 가정 위배 시 비모수 검정 대안을 제시하세요",
                    "효과크기(Cohen's d)를 계산하고 해석하세요"
                ])
            
            elif 'anova' in analysis_type:
                instructions.extend([
                    "사후 검정 결과를 명확히 해석하세요",
                    "효과크기(eta squared)와 검정력을 보고하세요",
                    "그룹 간 차이의 실질적 의미를 설명하세요"
                ])
            
            elif 'regression' in analysis_type:
                instructions.extend([
                    "회귀 가정을 철저히 확인하세요",
                    "다중공선성 문제를 점검하세요",
                    "모델의 예측력과 설명력을 구분하여 보고하세요"
                ])
            
            # 사용자 추가 요구사항 반영
            additional_reqs = user_preferences.get('additional_requirements', [])
            for req in additional_reqs:
                instructions.append(f"사용자 요구사항: {req}")
            
            # 보고서 형식에 따른 지침
            report_format = user_preferences.get('reporting_preferences', {}).get('format', {})
            if report_format.get('name') == '비즈니스 보고서':
                instructions.extend([
                    "비즈니스 임팩트를 명확히 제시하세요",
                    "의사결정을 위한 구체적인 권고사항을 포함하세요",
                    "기술적 용어는 최소화하고 이해하기 쉽게 설명하세요"
                ])
            elif report_format.get('name') == '기술 보고서':
                instructions.extend([
                    "통계적 세부사항을 상세히 기록하세요",
                    "방법론의 타당성을 논증하세요",
                    "한계점과 추가 연구 방향을 제시하세요"
                ])
            
            return instructions
            
        except Exception as e:
            self.logger.error(f"특별 지침 작성 오류: {e}")
            return [
                "분석 과정을 명확히 문서화하세요",
                "결과를 객관적으로 해석하세요"
            ]
    
    def _extract_method_details_from_text(self, text: str) -> Dict[str, Any]:
        """텍스트에서 방법 상세 정보 추출"""
        try:
            import re
            
            details = {
                'parameters': {},
                'customizations': {},
                'notes': text
            }
            
            # 파라미터 추출 패턴
            param_patterns = {
                'alpha': r'alpha[=:]?\s*([0-9.]+)',
                'confidence': r'confidence[=:]?\s*([0-9.]+)',
                'method': r'method[=:]?\s*([a-zA-Z_]+)',
                'alternative': r'alternative[=:]?\s*([a-zA-Z_-]+)'
            }
            
            for param, pattern in param_patterns.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1)) if '.' in match.group(1) else match.group(1)
                        details['parameters'][param] = value
                    except ValueError:
                        details['parameters'][param] = match.group(1)
            
            return details
            
        except Exception as e:
            self.logger.warning(f"텍스트에서 방법 상세 정보 추출 오류: {e}")
            return {'parameters': {}, 'customizations': {}, 'notes': text}
    
    def _extract_summary_from_text(self, text: str) -> Dict[str, Any]:
        """텍스트에서 대화 요약 추출"""
        try:
            import re
            
            summary = {
                'key_decisions': [],
                'clarifications': [],
                'final_confirmations': [],
                'raw_summary': text
            }
            
            # 키워드 기반 추출
            decision_keywords = ['선택', '결정', '채택', '승인']
            clarification_keywords = ['명확화', '설명', '확인', '질문']
            confirmation_keywords = ['확정', '승인', '동의', '최종']
            
            sentences = re.split(r'[.!?]', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if any(keyword in sentence for keyword in decision_keywords):
                    summary['key_decisions'].append(sentence)
                elif any(keyword in sentence for keyword in clarification_keywords):
                    summary['clarifications'].append(sentence)
                elif any(keyword in sentence for keyword in confirmation_keywords):
                    summary['final_confirmations'].append(sentence)
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"텍스트에서 대화 요약 추출 오류: {e}")
            return {
                'key_decisions': ['분석 방법 선택 완료'],
                'clarifications': [],
                'final_confirmations': ['사용자 선택사항 확정'],
                'raw_summary': text
            }


# 단계 등록
PipelineStepRegistry.register_step(5, UserSelectionStep) 