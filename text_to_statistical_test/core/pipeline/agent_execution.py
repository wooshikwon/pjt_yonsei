"""
Agent Execution Pipeline

7단계: Agentic LLM의 자율적 통계 검정 및 동적 조정
Agent가 통계 검정 전 과정을 자율적으로 수행하며, 필요시 동적으로 분석 방법을 조정합니다.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import time
import re

from .base_pipeline_step import BasePipelineStep, PipelineStepRegistry
from core.rag.rag_manager import RAGManager
from services.llm.llm_client import LLMClient
from services.llm.prompt_engine import PromptEngine
from services.code_executor.safe_code_runner import SafeCodeRunner
from services.visualization.plot_generator import PlotGenerator
from services.statistics.stats_executor import StatsExecutor


class AgentExecutionStep(BasePipelineStep):
    """7단계: Agentic LLM의 통계 분석 실행 및 결과 해석"""
    
    def __init__(self):
        """AgentExecutionStep 초기화"""
        super().__init__("Agentic LLM의 통계 분석 실행 및 결과 해석", 7)
        self.rag_manager = RAGManager()
        self.llm_client = LLMClient()
        self.prompt_engine = PromptEngine()
        self.code_runner = SafeCodeRunner()
        self.plot_generator = PlotGenerator()
        self.stats_executor = StatsExecutor()
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        입력 데이터 유효성 검증
        
        Args:
            input_data: 6단계에서 전달받은 데이터
            
        Returns:
            bool: 유효성 검증 결과
        """
        required_fields = [
            'analysis_code', 'execution_plan', 'data_requirements',
            'statistical_design', 'visualization_plan', 'documentation'
        ]
        return all(field in input_data for field in required_fields)
    
    def get_expected_output_schema(self) -> Dict[str, Any]:
        """
        예상 출력 스키마 반환
        
        Returns:
            Dict[str, Any]: 출력 데이터 스키마
        """
        return {
            'execution_results': {
                'status': str,
                'statistics': dict,
                'metrics': dict,
                'validation_results': list
            },
            'analysis_insights': {
                'key_findings': list,
                'statistical_significance': dict,
                'limitations': list
            },
            'visualizations': {
                'plots': list,
                'interactive_elements': list,
                'plot_descriptions': dict
            },
            'interpretation': {
                'summary': str,
                'detailed_analysis': dict,
                'recommendations': list
            },
            'quality_metrics': {
                'reliability_scores': dict,
                'validation_metrics': dict,
                'confidence_intervals': dict
            },
            'execution_metadata': {
                'runtime_stats': dict,
                'resource_usage': dict,
                'error_logs': list
            }
        }
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agentic LLM의 통계 분석 실행 및 결과 해석 파이프라인 실행
        
        Args:
            input_data: 파이프라인 실행 컨텍스트
                - analysis_code: 분석 코드
                - execution_plan: 실행 계획
                - data_requirements: 데이터 요구사항
                - statistical_design: 통계적 설계
                - visualization_plan: 시각화 계획
                - documentation: 문서화
            
        Returns:
            Dict: 실행 결과
        """
        self.logger.info("7단계: Agentic LLM의 통계 분석 실행 및 결과 해석 시작")
        
        try:
            # 1. 통계 분석 실행
            execution_results = self._execute_statistical_analysis(input_data)
            
            # 2. RAG 기반 결과 해석
            analysis_insights = self._interpret_results_with_rag(
                execution_results, input_data
            )
            
            # 3. 시각화 생성
            visualizations = self._generate_visualizations(
                execution_results, input_data['visualization_plan']
            )
            
            # 4. 결과 해석 및 요약
            interpretation = self._create_interpretation(
                execution_results, analysis_insights, visualizations
            )
            
            # 5. 품질 메트릭 계산
            quality_metrics = self._calculate_quality_metrics(
                execution_results, input_data['statistical_design']
            )
            
            # 6. 실행 메타데이터 수집
            execution_metadata = self._collect_execution_metadata()
            
            self.logger.info("통계 분석 실행 및 결과 해석 완료")
            
            return {
                'execution_results': execution_results,
                'analysis_insights': analysis_insights,
                'visualizations': visualizations,
                'interpretation': interpretation,
                'quality_metrics': quality_metrics,
                'execution_metadata': execution_metadata,
                'success_message': "📊 통계 분석이 성공적으로 완료되었습니다."
            }
                
        except Exception as e:
            self.logger.error(f"통계 분석 실행 파이프라인 오류: {e}")
            return {
                'error': True,
                'error_message': str(e),
                'error_type': 'execution_error'
            }
    
    def _execute_statistical_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """통계 분석 실행"""
        try:
            # 1. 코드 실행 준비
            execution_context = self._prepare_execution_context(input_data)
            
            # 2. 데이터 전처리 실행
            preprocessed_data = self._execute_preprocessing(
                input_data['data_requirements']
            )
            
            # 3. 통계 분석 실행
            statistics = self.stats_executor.execute_analysis(
                code=input_data['analysis_code']['main_script'],
                data=preprocessed_data,
                parameters=input_data['statistical_design']['parameters']
            )
            
            # 4. 메트릭 계산
            metrics = self._calculate_analysis_metrics(statistics)
            
            # 5. 검증 실행
            validation_results = self._execute_validations(
                statistics, input_data['execution_plan']['validation_checks']
            )
            
            return {
                'status': 'success',
                'statistics': statistics,
                'metrics': metrics,
                'validation_results': validation_results
            }
            
        except Exception as e:
            self.logger.error(f"통계 분석 실행 오류: {e}")
            return {
                'status': 'error',
                'error_message': str(e)
            }
    
    def _interpret_results_with_rag(self, execution_results: Dict[str, Any],
                                  input_data: Dict[str, Any]) -> Dict[str, Any]:
        """RAG 기반 결과 해석"""
        try:
            # 1. 관련 지식 검색
            interpretation_knowledge = self.rag_manager.search(
                query=self._build_interpretation_query(execution_results)
            )
            
            # 2. 컨텍스트 구축
            interpretation_context = self.rag_manager.build_context(
                query=self._build_interpretation_query(execution_results)
            )
            
            # 3. LLM을 통한 해석 생성
            prompt = self.prompt_engine.create_interpretation_prompt(
                context=interpretation_context
            )
            
            llm_response = self.llm_client.generate(prompt)
            
            # 4. 해석 결과 구조화
            interpretation = self._parse_interpretation_response(llm_response)
            
            return {
                'key_findings': interpretation.get('key_findings', []),
                'statistical_significance': interpretation.get('significance', {}),
                'limitations': interpretation.get('limitations', [])
            }
            
        except Exception as e:
            self.logger.error(f"결과 해석 오류: {e}")
            return {
                'error': True,
                'error_message': str(e)
            }
    
    def _generate_visualizations(self, execution_results: Dict[str, Any],
                               visualization_plan: Dict[str, Any]) -> Dict[str, Any]:
        """시각화 생성"""
        try:
            # 1. 기본 플롯 생성
            plots = self.plot_generator.generate_plots(
                data=execution_results['statistics'],
                plot_specs=visualization_plan['plots']
            )
            
            # 2. 인터랙티브 요소 추가
            interactive_elements = self.plot_generator.add_interactivity(
                plots=plots,
                interactive_specs=visualization_plan['interactive_elements']
            )
            
            # 3. 플롯 설명 생성
            plot_descriptions = self._generate_plot_descriptions(
                plots, execution_results
            )
            
            return {
                'plots': plots,
                'interactive_elements': interactive_elements,
                'plot_descriptions': plot_descriptions
            }
            
        except Exception as e:
            self.logger.error(f"시각화 생성 오류: {e}")
            return {
                'error': True,
                'error_message': str(e)
            }
    
    def _create_interpretation(self, execution_results: Dict[str, Any],
                             analysis_insights: Dict[str, Any],
                             visualizations: Dict[str, Any]) -> Dict[str, Any]:
        """결과 해석 및 요약"""
        try:
            # 1. 요약 생성
            summary = self._generate_analysis_summary(
                execution_results, analysis_insights
            )
            
            # 2. 상세 분석
            detailed_analysis = self._generate_detailed_analysis(
                execution_results, analysis_insights, visualizations
            )
            
            # 3. 추천사항 도출
            recommendations = self._generate_recommendations(
                detailed_analysis, analysis_insights
            )
            
            return {
                'summary': summary,
                'detailed_analysis': detailed_analysis,
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.error(f"해석 생성 오류: {e}")
            return {
                'error': True,
                'error_message': str(e)
            }
    
    def _calculate_quality_metrics(self, execution_results: Dict[str, Any],
                                 statistical_design: Dict[str, Any]) -> Dict[str, Any]:
        """품질 메트릭 계산"""
        try:
            # 1. 신뢰도 점수 계산
            reliability_scores = self._calculate_reliability_scores(
                execution_results
            )
            
            # 2. 검증 메트릭 계산
            validation_metrics = self._calculate_validation_metrics(
                execution_results, statistical_design
            )
            
            # 3. 신뢰구간 계산
            confidence_intervals = self._calculate_confidence_intervals(
                execution_results, statistical_design
            )
            
            return {
                'reliability_scores': reliability_scores,
                'validation_metrics': validation_metrics,
                'confidence_intervals': confidence_intervals
            }
            
        except Exception as e:
            self.logger.error(f"품질 메트릭 계산 오류: {e}")
            return {
                'error': True,
                'error_message': str(e)
            }
    
    def _collect_execution_metadata(self) -> Dict[str, Any]:
        """실행 메타데이터 수집"""
        try:
            # 1. 런타임 통계 수집
            runtime_stats = self.stats_executor.get_runtime_statistics()
            
            # 2. 리소스 사용량 수집
            resource_usage = self.stats_executor.get_resource_usage()
            
            # 3. 오류 로그 수집
            error_logs = self.stats_executor.get_error_logs()
            
            return {
                'runtime_stats': runtime_stats,
                'resource_usage': resource_usage,
                'error_logs': error_logs
            }
            
        except Exception as e:
            self.logger.error(f"메타데이터 수집 오류: {e}")
            return {
                'error': True,
                'error_message': str(e)
            }
    
    def _prepare_execution_context(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """실행 컨텍스트 준비"""
        try:
            execution_context = input_data.get('execution_context', {})
            analysis_code = input_data.get('analysis_code', {})
            data_requirements = input_data.get('data_requirements', {})
            
            # 실행 환경 설정
            context = {
                'workspace_path': '/tmp/statistical_analysis',
                'data_path': data_requirements.get('data_path', ''),
                'output_path': '/tmp/analysis_output',
                'temp_path': '/tmp/analysis_temp'
            }
            
            # 파라미터 설정
            context['parameters'] = execution_context.get('parameters', {})
            context['constraints'] = execution_context.get('constraints', {})
            context['special_instructions'] = execution_context.get('special_instructions', [])
            
            # 코드 실행 설정
            context['code_settings'] = {
                'timeout': context['constraints'].get('max_execution_time', 300),
                'memory_limit': context['constraints'].get('max_memory_usage', 1024),
                'safe_mode': context['constraints'].get('safe_mode', True),
                'allowed_imports': ['pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'statsmodels'],
                'restricted_imports': context['constraints'].get('restricted_imports', ['os', 'subprocess', 'sys'])
            }
            
            # 데이터 설정
            context['data_settings'] = {
                'encoding': data_requirements.get('encoding', 'utf-8'),
                'separator': data_requirements.get('separator', ','),
                'missing_values': data_requirements.get('missing_values', ['', 'NA', 'NULL']),
                'data_types': data_requirements.get('data_types', {})
            }
            
            # 출력 설정
            context['output_settings'] = {
                'save_plots': True,
                'plot_format': ['png', 'html'],
                'save_data': True,
                'save_results': True,
                'generate_report': True
            }
            
            return context
            
        except Exception as e:
            self.logger.error(f"실행 컨텍스트 준비 오류: {e}")
            return {
                'workspace_path': '/tmp/statistical_analysis',
                'parameters': {},
                'constraints': {'max_execution_time': 300, 'safe_mode': True}
            }
    
    def _execute_preprocessing(self, data_requirements: Dict[str, Any]) -> Any:
        """데이터 전처리 실행"""
        try:
            from services.statistics.data_preprocessor import DataPreprocessor
            
            preprocessor = DataPreprocessor()
            
            # 데이터 로드
            data_path = data_requirements.get('data_path', '')
            if not data_path:
                raise ValueError("데이터 경로가 지정되지 않았습니다.")
            
            # 전처리 옵션 설정
            preprocessing_options = {
                'handle_missing': data_requirements.get('missing_value_handling', 'drop'),
                'outlier_treatment': data_requirements.get('outlier_handling', 'identify'),
                'normalization': data_requirements.get('normalization', None),
                'encoding': data_requirements.get('categorical_encoding', 'label')
            }
            
            # 데이터 전처리 실행
            preprocessed_data = preprocessor.preprocess_data(
                data_path=data_path,
                target_columns=data_requirements.get('target_columns', []),
                options=preprocessing_options
            )
            
            # 전처리 결과 검증
            if preprocessed_data is None:
                raise ValueError("데이터 전처리에 실패했습니다.")
            
            self.logger.info(f"데이터 전처리 완료: {preprocessed_data.shape}")
            
            return preprocessed_data
            
        except Exception as e:
            self.logger.error(f"데이터 전처리 실행 오류: {e}")
            return None
    
    def _calculate_analysis_metrics(self, statistics: Dict[str, Any]) -> Dict[str, Any]:
        """분석 메트릭 계산"""
        try:
            metrics = {}
            
            # 기본 통계량 메트릭
            if 'descriptive_stats' in statistics:
                desc_stats = statistics['descriptive_stats']
                metrics['data_quality'] = {
                    'completeness': desc_stats.get('completeness_ratio', 0.0),
                    'sample_size': desc_stats.get('count', 0),
                    'missing_percentage': desc_stats.get('missing_percentage', 0.0)
                }
            
            # 통계 검정 메트릭
            if 'test_results' in statistics:
                test_results = statistics['test_results']
                metrics['statistical_significance'] = {
                    'p_value': test_results.get('p_value', 1.0),
                    'test_statistic': test_results.get('statistic', 0.0),
                    'degrees_of_freedom': test_results.get('df', 0),
                    'is_significant': test_results.get('p_value', 1.0) < 0.05
                }
                
                # 효과크기 계산
                if 'effect_size' in test_results:
                    metrics['effect_size'] = {
                        'value': test_results['effect_size'],
                        'interpretation': self._interpret_effect_size(
                            test_results['effect_size'], 
                            test_results.get('test_type', '')
                        )
                    }
            
            # 모델 성능 메트릭 (회귀분석인 경우)
            if 'model_summary' in statistics:
                model_summary = statistics['model_summary']
                metrics['model_performance'] = {
                    'r_squared': model_summary.get('r_squared', 0.0),
                    'adjusted_r_squared': model_summary.get('adj_r_squared', 0.0),
                    'f_statistic': model_summary.get('f_statistic', 0.0),
                    'aic': model_summary.get('aic', float('inf')),
                    'bic': model_summary.get('bic', float('inf'))
                }
            
            # 가정 검증 메트릭
            if 'assumption_tests' in statistics:
                assumption_tests = statistics['assumption_tests']
                metrics['assumption_validity'] = {}
                
                for test_name, test_result in assumption_tests.items():
                    metrics['assumption_validity'][test_name] = {
                        'passed': test_result.get('p_value', 0.0) > 0.05,
                        'p_value': test_result.get('p_value', 1.0),
                        'test_statistic': test_result.get('statistic', 0.0)
                    }
            
            # 신뢰도 메트릭
            metrics['reliability'] = {
                'sample_adequacy': self._assess_sample_adequacy(statistics),
                'assumption_score': self._calculate_assumption_score(metrics.get('assumption_validity', {})),
                'overall_confidence': self._calculate_overall_confidence(metrics)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"분석 메트릭 계산 오류: {e}")
            return {
                'data_quality': {'completeness': 0.0, 'sample_size': 0},
                'reliability': {'overall_confidence': 0.0}
            }
    
    def _execute_validations(self, statistics: Dict[str, Any],
                           validation_checks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """검증 실행"""
        try:
            validation_results = []
            
            for check in validation_checks:
                check_type = check.get('type', '')
                check_name = check.get('name', check_type)
                
                result = {
                    'check_name': check_name,
                    'check_type': check_type,
                    'passed': False,
                    'message': '',
                    'details': {}
                }
                
                try:
                    if check_type == 'data_completeness_check':
                        result.update(self._validate_data_completeness(statistics, check))
                    elif check_type == 'outlier_detection':
                        result.update(self._validate_outliers(statistics, check))
                    elif check_type == 'normality_assumption':
                        result.update(self._validate_normality(statistics, check))
                    elif check_type == 'independence_assumption':
                        result.update(self._validate_independence(statistics, check))
                    elif check_type == 'homoscedasticity_assumption':
                        result.update(self._validate_homoscedasticity(statistics, check))
                    elif check_type == 'linearity_assumption':
                        result.update(self._validate_linearity(statistics, check))
                    elif check_type == 'multicollinearity_check':
                        result.update(self._validate_multicollinearity(statistics, check))
                    elif check_type == 'sample_size_adequacy':
                        result.update(self._validate_sample_size(statistics, check))
                    else:
                        result['message'] = f"알 수 없는 검증 유형: {check_type}"
                        
                except Exception as check_error:
                    result['message'] = f"검증 실행 오류: {str(check_error)}"
                    result['details']['error'] = str(check_error)
                
                validation_results.append(result)
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"검증 실행 오류: {e}")
            return [{
                'check_name': 'error',
                'check_type': 'error',
                'passed': False,
                'message': f"검증 실행 중 오류 발생: {str(e)}"
            }]
    
    def _build_interpretation_query(self, execution_results: Dict[str, Any]) -> str:
        """해석 쿼리 생성"""
        try:
            statistics = execution_results.get('statistics', {})
            test_results = statistics.get('test_results', {})
            
            # 기본 정보
            sample_size = statistics.get('descriptive_stats', {}).get('sample_size', 'N/A')
            num_tests = len(test_results)
            
            # 유의한 결과 집계
            significant_tests = []
            for test_name, result in test_results.items():
                if result.get('p_value', 1.0) < 0.05:
                    significant_tests.append(test_name)
            
            # 요약 문서 구성
            query = f"""
다음 통계 분석 결과를 해석해주세요:

### 기본 정보
- 표본 크기: {sample_size}
- 실행된 검정 수: {num_tests}
- 유의한 결과: {len(significant_tests)}개

### 주요 발견사항
"""
            
            # 유의한 결과들 요약
            if significant_tests:
                query += "**통계적으로 유의한 결과:**\n"
                for test_name in significant_tests[:5]:  # 최대 5개
                    result = test_results[test_name]
                    p_value = result.get('p_value', 1.0)
                    effect_size = result.get('effect_size', 'N/A')
                    query += f"- {test_name}: p = {p_value:.4f}"
                    if effect_size != 'N/A':
                        query += f", 효과크기 = {effect_size}"
                    query += "\n"
            else:
                query += "**통계적으로 유의한 결과가 발견되지 않았습니다.**\n"
            
            # 해석 인사이트 포함
            if 'key_insights' in execution_results:
                insights = execution_results['key_insights']
                if insights:
                    query += "\n### 핵심 인사이트\n"
                    for insight in insights[:3]:  # 최대 3개
                        query += f"- {insight}\n"
            
            # 결론
            if 'conclusion' in execution_results:
                query += f"\n### 결론\n{execution_results['conclusion']}\n"
            
            # 권고사항
            if 'follow_up_suggestions' in execution_results:
                suggestions = execution_results['follow_up_suggestions']
                if suggestions:
                    query += "\n### 권고사항\n"
                    for suggestion in suggestions[:3]:  # 최대 3개
                        query += f"- {suggestion}\n"
            
            query += "\n### 주의사항\n"
            query += "- 이 분석 결과는 제공된 데이터와 가정에 기반합니다.\n"
            query += "- 실무적 의사결정시 도메인 전문가와의 상담을 권장합니다.\n"
            query += "- 추가적인 데이터 수집이나 분석이 필요할 수 있습니다.\n"
            
            return query
            
        except Exception as e:
            self.logger.error(f"해석 쿼리 생성 오류: {e}")
            return "통계 분석 결과를 일반적인 관점에서 해석해주세요."
    
    def _parse_interpretation_response(self, llm_response: str) -> Dict[str, Any]:
        """해석 응답 파싱"""
        try:
            import json
            import re
            
            interpretation = {
                'statistical_significance': '',
                'practical_significance': '',
                'reliability_assessment': '',
                'limitations': [],
                'follow_up_suggestions': [],
                'key_insights': [],
                'conclusion': ''
            }
            
            # JSON 형태 응답 시도
            json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    return parsed
                except json.JSONDecodeError:
                    # JSON 파싱 실패시 기본 구조로 텍스트 파싱 시도
                    self.logger.warning("JSON 파싱 실패, 텍스트 파싱으로 전환")
                    return self._fallback_interpretation_parsing(llm_response)
            
            # 구조화된 텍스트 파싱
            response_lower = llm_response.lower()
            
            # 1. 통계적 유의성 섹션 추출
            stat_sig_pattern = r'통계적\s*유의성[:\s]*(.*?)(?=실무적|실용적|신뢰성|한계|결론|$)'
            stat_sig_match = re.search(stat_sig_pattern, llm_response, re.IGNORECASE | re.DOTALL)
            if stat_sig_match:
                interpretation['statistical_significance'] = stat_sig_match.group(1).strip()
            
            # 2. 실무적 유의성 섹션 추출
            practical_sig_pattern = r'(?:실무적|실용적)\s*유의성[:\s]*(.*?)(?=신뢰성|한계|결론|$)'
            practical_sig_match = re.search(practical_sig_pattern, llm_response, re.IGNORECASE | re.DOTALL)
            if practical_sig_match:
                interpretation['practical_significance'] = practical_sig_match.group(1).strip()
            
            # 3. 신뢰성 평가 섹션 추출
            reliability_pattern = r'신뢰성[:\s]*(.*?)(?=한계|결론|제안|$)'
            reliability_match = re.search(reliability_pattern, llm_response, re.IGNORECASE | re.DOTALL)
            if reliability_match:
                interpretation['reliability_assessment'] = reliability_match.group(1).strip()
            
            # 4. 한계점 추출
            limitation_pattern = r'한계[점]?[:\s]*(.*?)(?=제안|권고|결론|$)'
            limitation_match = re.search(limitation_pattern, llm_response, re.IGNORECASE | re.DOTALL)
            if limitation_match:
                limitation_text = limitation_match.group(1)
                limitations = re.findall(r'[-•]\s*([^-•\n]+)', limitation_text)
                interpretation['limitations'] = [lim.strip() for lim in limitations if lim.strip()]
            
            # 5. 후속 제안 추출
            follow_up_pattern = r'(?:후속|추가|제안|권고)[:\s]*(.*?)(?=결론|$)'
            follow_up_match = re.search(follow_up_pattern, llm_response, re.IGNORECASE | re.DOTALL)
            if follow_up_match:
                follow_up_text = follow_up_match.group(1)
                suggestions = re.findall(r'[-•]\s*([^-•\n]+)', follow_up_text)
                interpretation['follow_up_suggestions'] = [sug.strip() for sug in suggestions if sug.strip()]
            
            # 6. 핵심 인사이트 추출 (주요 포인트들)
            insight_keywords = ['중요한', '핵심', '주목할', '특히', '결과적으로']
            insights = []
            sentences = re.split(r'[.!?]', llm_response)
            for sentence in sentences:
                if any(keyword in sentence for keyword in insight_keywords):
                    clean_sentence = sentence.strip()
                    if len(clean_sentence) > 10:  # 너무 짧은 문장 제외
                        insights.append(clean_sentence)
            interpretation['key_insights'] = insights[:5]  # 최대 5개
            
            # 7. 결론 추출
            conclusion_pattern = r'결론[:\s]*(.*?)$'
            conclusion_match = re.search(conclusion_pattern, llm_response, re.IGNORECASE | re.DOTALL)
            if conclusion_match:
                interpretation['conclusion'] = conclusion_match.group(1).strip()
            
            # 빈 필드들에 대한 기본값 설정
            if not interpretation['statistical_significance']:
                interpretation['statistical_significance'] = "통계 분석 결과에 대한 기본적인 해석을 참조하세요."
            
            if not interpretation['conclusion']:
                # 마지막 문단을 결론으로 사용
                paragraphs = llm_response.split('\n\n')
                if paragraphs:
                    interpretation['conclusion'] = paragraphs[-1].strip()
            
            return interpretation
            
        except Exception as e:
            self.logger.error(f"해석 응답 파싱 오류: {e}")
            return {
                'statistical_significance': '파싱 오류로 인해 해석을 생성할 수 없습니다.',
                'practical_significance': '',
                'reliability_assessment': '',
                'limitations': ['응답 파싱 중 오류 발생'],
                'follow_up_suggestions': ['전문가 상담 권장'],
                'key_insights': [],
                'conclusion': '결과 해석에 주의가 필요합니다.',
                'error': str(e)
            }
    
    def _generate_plot_descriptions(self, plots: List[Dict[str, Any]],
                                  execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """플롯 설명 생성"""
        try:
            plot_descriptions = {
                'individual_plots': {},
                'plot_relationships': [],
                'visual_insights': [],
                'interpretation_guide': {}
            }
            
            statistics = execution_results.get('statistics', {})
            test_results = statistics.get('test_results', {})
            
            # 개별 플롯 설명 생성
            for plot in plots:
                plot_id = plot.get('id', f"plot_{len(plot_descriptions['individual_plots']) + 1}")
                plot_type = plot.get('type', 'unknown')
                plot_title = plot.get('title', 'Untitled Plot')
                
                description = self._generate_single_plot_description(
                    plot, test_results
                )
                
                plot_descriptions['individual_plots'][plot_id] = {
                    'title': plot_title,
                    'type': plot_type,
                    'description': description,
                    'key_features': self._identify_plot_key_features(plot, test_results),
                    'interpretation_tips': self._generate_plot_interpretation_tips(plot_type)
                }
            
            # 플롯 간 관계 분석
            plot_descriptions['plot_relationships'] = self._analyze_plot_relationships(plots)
            
            # 시각적 인사이트 도출
            plot_descriptions['visual_insights'] = self._derive_visual_insights(plots, test_results)
            
            # 해석 가이드 생성
            plot_descriptions['interpretation_guide'] = self._create_interpretation_guide(plots)
            
            return plot_descriptions
            
        except Exception as e:
            self.logger.error(f"플롯 설명 생성 오류: {e}")
            return {
                'individual_plots': {},
                'plot_relationships': [],
                'visual_insights': ['플롯 설명 생성 중 오류가 발생했습니다.'],
                'interpretation_guide': {'error': str(e)}
            }
    
    def _generate_single_plot_description(self, plot: Dict[str, Any], 
                                        test_results: Dict[str, Any]) -> str:
        """단일 플롯 설명 생성"""
        plot_type = plot.get('type', 'unknown')
        
        if plot_type == 'histogram':
            return self._describe_histogram(plot, test_results)
        elif plot_type == 'boxplot':
            return self._describe_boxplot(plot, test_results)
        elif plot_type == 'scatter':
            return self._describe_scatterplot(plot, test_results)
        elif plot_type == 'bar':
            return self._describe_barplot(plot, test_results)
        elif plot_type == 'line':
            return self._describe_lineplot(plot, test_results)
        elif 'correlation' in plot_type:
            return self._describe_correlation_plot(plot, test_results)
        elif 'residual' in plot_type:
            return self._describe_residual_plot(plot, test_results)
        else:
            return f"이 {plot_type} 플롯은 데이터의 패턴과 관계를 시각적으로 보여줍니다."
    
    def _describe_histogram(self, plot: Dict[str, Any], test_results: Dict[str, Any]) -> str:
        """히스토그램 설명"""
        variable = plot.get('variables', ['변수'])[0] if plot.get('variables') else '변수'
        
        description = f"{variable}의 분포를 보여주는 히스토그램입니다. "
        description += "막대의 높이는 해당 구간에 속하는 데이터 포인트의 빈도를 나타냅니다. "
        
        # 정규성 검정 결과가 있다면 포함
        normality_tests = ['shapiro_wilk', 'kolmogorov_smirnov', 'anderson_darling']
        for test_name in normality_tests:
            if test_name in test_results:
                result = test_results[test_name]
                p_value = result.get('p_value', 1.0)
                if p_value < 0.05:
                    description += f"정규성 검정 결과 정규분포를 따르지 않는 것으로 나타났습니다 (p = {p_value:.3f}). "
                else:
                    description += f"정규성 검정 결과 정규분포 가정을 만족하는 것으로 보입니다 (p = {p_value:.3f}). "
                break
        
        description += "분포의 모양, 중심 경향성, 그리고 이상값 여부를 확인할 수 있습니다."
        
        return description
    
    def _describe_boxplot(self, plot: Dict[str, Any], test_results: Dict[str, Any]) -> str:
        """박스플롯 설명"""
        variables = plot.get('variables', ['변수'])
        
        description = f"{', '.join(variables)}의 분포 특성을 요약한 박스플롯입니다. "
        description += "박스는 1사분위수(Q1)부터 3사분위수(Q3)까지의 구간을 나타내며, "
        description += "박스 내부의 선은 중앙값(median)을 표시합니다. "
        description += "박스에서 연장된 선(whiskers)은 데이터의 범위를 보여주고, "
        description += "개별 점들은 이상값(outliers)을 나타냅니다."
        
        return description
    
    def _describe_scatterplot(self, plot: Dict[str, Any], test_results: Dict[str, Any]) -> str:
        """산점도 설명"""
        variables = plot.get('variables', ['X변수', 'Y변수'])
        x_var = variables[0] if len(variables) > 0 else 'X변수'
        y_var = variables[1] if len(variables) > 1 else 'Y변수'
        
        description = f"{x_var}와 {y_var} 간의 관계를 보여주는 산점도입니다. "
        
        # 상관분석 결과가 있다면 포함
        correlation_tests = ['pearson_correlation', 'spearman_correlation']
        for test_name in correlation_tests:
            if test_name in test_results:
                result = test_results[test_name]
                correlation = result.get('correlation', 0)
                p_value = result.get('p_value', 1.0)
                
                strength = "약한" if abs(correlation) < 0.3 else "보통" if abs(correlation) < 0.7 else "강한"
                direction = "양의" if correlation > 0 else "음의"
                
                if p_value < 0.05:
                    description += f"두 변수 간에는 {strength} {direction} 상관관계가 있습니다 "
                    description += f"(r = {correlation:.3f}, p = {p_value:.3f}). "
                else:
                    description += f"두 변수 간의 상관관계는 통계적으로 유의하지 않습니다 "
                    description += f"(r = {correlation:.3f}, p = {p_value:.3f}). "
                break
        
        description += "점들의 패턴을 통해 선형성, 이상값, 그리고 관계의 강도를 파악할 수 있습니다."
        
        return description
    
    def _describe_barplot(self, plot: Dict[str, Any], test_results: Dict[str, Any]) -> str:
        """막대그래프 설명"""
        variables = plot.get('variables', ['범주변수'])
        variable = variables[0] if variables else '범주변수'
        
        description = f"{variable}의 각 범주별 빈도 또는 평균값을 보여주는 막대그래프입니다. "
        description += "막대의 높이는 해당 범주의 값의 크기를 나타내며, "
        description += "범주 간 차이를 직관적으로 비교할 수 있습니다."
        
        # 카이제곱 검정이나 ANOVA 결과가 있다면 포함
        group_tests = ['chi_square', 'one_way_anova', 'kruskal_wallis']
        for test_name in group_tests:
            if test_name in test_results:
                result = test_results[test_name]
                p_value = result.get('p_value', 1.0)
                
                if p_value < 0.05:
                    description += f" 범주 간 차이는 통계적으로 유의합니다 (p = {p_value:.3f}). "
                else:
                    description += f" 범주 간 차이는 통계적으로 유의하지 않습니다 (p = {p_value:.3f}). "
                break
        
        return description
    
    def _describe_lineplot(self, plot: Dict[str, Any], test_results: Dict[str, Any]) -> str:
        """선그래프 설명"""
        description = "시간에 따른 변화나 순서형 변수의 패턴을 보여주는 선그래프입니다. "
        description += "선의 기울기와 변화 패턴을 통해 트렌드와 주기적 변동을 파악할 수 있습니다."
        
        return description
    
    def _describe_correlation_plot(self, plot: Dict[str, Any], test_results: Dict[str, Any]) -> str:
        """상관관계 플롯 설명"""
        description = "변수들 간의 상관관계를 시각적으로 표현한 플롯입니다. "
        description += "색상의 강도와 방향은 상관계수의 크기와 방향을 나타내며, "
        description += "변수들 간의 복잡한 관계 패턴을 한눈에 파악할 수 있습니다."
        
        return description
    
    def _describe_residual_plot(self, plot: Dict[str, Any], test_results: Dict[str, Any]) -> str:
        """잔차 플롯 설명"""
        description = "회귀 모델의 잔차(residuals)를 시각화한 플롯입니다. "
        description += "잔차의 패턴을 통해 모델의 가정 위배 여부를 확인할 수 있으며, "
        description += "등분산성, 선형성, 독립성 등의 회귀 가정을 검토하는 데 사용됩니다."
        
        return description
    
    def _identify_plot_key_features(self, plot: Dict[str, Any], 
                                  test_results: Dict[str, Any]) -> List[str]:
        """플롯의 주요 특징 식별"""
        features = []
        plot_type = plot.get('type', 'unknown')
        
        if plot_type == 'histogram':
            features.extend(['분포의 모양', '중심 경향성', '산포도', '이상값'])
        elif plot_type == 'boxplot':
            features.extend(['중앙값', '사분위수', '이상값', '분포의 대칭성'])
        elif plot_type == 'scatter':
            features.extend(['선형 관계', '상관 강도', '이상값', '패턴'])
        elif plot_type == 'bar':
            features.extend(['범주별 빈도', '그룹 간 차이', '상대적 크기'])
        elif 'correlation' in plot_type:
            features.extend(['상관 강도', '상관 방향', '변수 간 관계'])
        elif 'residual' in plot_type:
            features.extend(['잔차 패턴', '등분산성', '선형성', '가정 위배'])
        
        return features
    
    def _generate_plot_interpretation_tips(self, plot_type: str) -> List[str]:
        """플롯 해석 팁 생성"""
        tips = []
        
        if plot_type == 'histogram':
            tips.extend([
                "정규분포에 가까운 종 모양인지 확인하세요",
                "치우침(skewness)이 있는지 관찰하세요",
                "이상값이나 다중 모드가 있는지 확인하세요"
            ])
        elif plot_type == 'boxplot':
            tips.extend([
                "박스의 위치와 크기로 분포 특성을 파악하세요",
                "이상값들이 의미가 있는지 검토하세요",
                "그룹 간 박스의 차이를 비교하세요"
            ])
        elif plot_type == 'scatter':
            tips.extend([
                "점들이 직선 패턴을 보이는지 확인하세요",
                "이상값이 상관관계에 미치는 영향을 고려하세요",
                "비선형 관계의 가능성을 검토하세요"
            ])
        elif plot_type == 'bar':
            tips.extend([
                "범주 간 차이의 크기와 방향을 확인하세요",
                "표본 크기의 차이를 고려하세요",
                "실질적 유의성을 함께 평가하세요"
            ])
        
        return tips
    
    def _analyze_plot_relationships(self, plots: List[Dict[str, Any]]) -> List[str]:
        """플롯 간 관계 분석"""
        relationships = []
        
        if len(plots) < 2:
            return relationships
        
        # 상보적 관계 식별
        plot_types = [plot.get('type', '') for plot in plots]
        
        if 'histogram' in plot_types and 'boxplot' in plot_types:
            relationships.append("히스토그램과 박스플롯이 함께 분포의 전체 특성을 보여줍니다")
        
        if 'scatter' in plot_types and any('correlation' in ptype for ptype in plot_types):
            relationships.append("산점도와 상관분석 결과가 변수 간 관계를 다각도로 보여줍니다")
        
        if any('residual' in ptype for ptype in plot_types) and 'scatter' in plot_types:
            relationships.append("잔차 플롯이 회귀 모델의 적합성을 검증합니다")
        
        return relationships
    
    def _derive_visual_insights(self, plots: List[Dict[str, Any]], 
                              test_results: Dict[str, Any]) -> List[str]:
        """시각적 인사이트 도출"""
        insights = []
        
        # 플롯 유형별 인사이트
        for plot in plots:
            plot_type = plot.get('type', '')
            
            if 'distribution' in plot.get('insights', {}):
                insights.append("데이터 분포의 특성이 분석 방법 선택에 중요한 영향을 미칩니다")
            
            if 'outliers' in plot.get('insights', {}):
                insights.append("이상값의 존재가 분석 결과에 미치는 영향을 신중히 고려해야 합니다")
            
            if 'pattern' in plot.get('insights', {}):
                insights.append("데이터의 패턴이 예상과 다를 경우 추가 탐색이 필요합니다")
        
        # 일반적 인사이트
        if len(plots) > 2:
            insights.append("다양한 시각화를 통해 데이터의 서로 다른 측면을 이해할 수 있습니다")
        
        return insights[:5]  # 최대 5개로 제한
    
    def _create_interpretation_guide(self, plots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """해석 가이드 생성"""
        guide = {
            'reading_order': [],
            'key_questions': [],
            'common_mistakes': [],
            'best_practices': []
        }
        
        # 읽기 순서 제안
        plot_priorities = {
            'histogram': 1,
            'boxplot': 2,
            'scatter': 3,
            'correlation': 4,
            'residual': 5,
            'bar': 3
        }
        
        sorted_plots = sorted(plots, key=lambda p: plot_priorities.get(p.get('type', ''), 99))
        guide['reading_order'] = [f"{i+1}. {plot.get('title', f'Plot {i+1}')}" 
                                 for i, plot in enumerate(sorted_plots)]
        
        # 핵심 질문들
        guide['key_questions'] = [
            "이 시각화가 보여주는 주요 패턴은 무엇인가?",
            "데이터의 가정들이 만족되고 있는가?",
            "이상값이나 특이점들이 있는가?",
            "시각적 패턴이 통계 검정 결과와 일치하는가?"
        ]
        
        # 흔한 실수들
        guide['common_mistakes'] = [
            "시각적 패턴만으로 인과관계를 추론하는 것",
            "이상값을 무조건 제거하려는 시도",
            "척도의 차이를 고려하지 않은 비교",
            "표본 크기를 고려하지 않은 해석"
        ]
        
        # 모범 관행들
        guide['best_practices'] = [
            "여러 시각화를 종합적으로 검토하기",
            "통계 검정 결과와 시각화를 함께 고려하기",
            "도메인 지식을 바탕으로 해석하기",
            "결과의 한계점을 명확히 인식하기"
        ]
        
        return guide
    
    def _generate_analysis_summary(self, execution_results: Dict[str, Any],
                                 analysis_insights: Dict[str, Any]) -> str:
        """분석 요약 생성"""
        try:
            statistics = execution_results.get('statistics', {})
            test_results = statistics.get('test_results', {})
            
            # 기본 정보
            sample_size = statistics.get('descriptive_stats', {}).get('sample_size', 'N/A')
            num_tests = len(test_results)
            
            # 유의한 결과 집계
            significant_tests = []
            for test_name, result in test_results.items():
                if result.get('p_value', 1.0) < 0.05:
                    significant_tests.append(test_name)
            
            # 요약 문서 구성
            summary = f"""
## 통계 분석 요약 보고서

### 기본 정보
- 표본 크기: {sample_size}
- 실행된 검정 수: {num_tests}
- 유의한 결과: {len(significant_tests)}개

### 주요 발견사항
"""
            
            # 유의한 결과들 요약
            if significant_tests:
                summary += "**통계적으로 유의한 결과:**\n"
                for test_name in significant_tests[:5]:  # 최대 5개
                    result = test_results[test_name]
                    p_value = result.get('p_value', 1.0)
                    effect_size = result.get('effect_size', 'N/A')
                    summary += f"- {test_name}: p = {p_value:.4f}"
                    if effect_size != 'N/A':
                        summary += f", 효과크기 = {effect_size}"
                    summary += "\n"
            else:
                summary += "**통계적으로 유의한 결과가 발견되지 않았습니다.**\n"
            
            # 해석 인사이트 포함
            if 'key_insights' in analysis_insights:
                insights = analysis_insights['key_insights']
                if insights:
                    summary += "\n### 핵심 인사이트\n"
                    for insight in insights[:3]:  # 최대 3개
                        summary += f"- {insight}\n"
            
            # 결론
            if 'conclusion' in analysis_insights:
                summary += f"\n### 결론\n{analysis_insights['conclusion']}\n"
            
            # 권고사항
            if 'follow_up_suggestions' in analysis_insights:
                suggestions = analysis_insights['follow_up_suggestions']
                if suggestions:
                    summary += "\n### 권고사항\n"
                    for suggestion in suggestions[:3]:  # 최대 3개
                        summary += f"- {suggestion}\n"
            
            summary += "\n### 주의사항\n"
            summary += "- 이 분석 결과는 제공된 데이터와 가정에 기반합니다.\n"
            summary += "- 실무적 의사결정시 도메인 전문가와의 상담을 권장합니다.\n"
            summary += "- 추가적인 데이터 수집이나 분석이 필요할 수 있습니다.\n"
            
            return summary
            
        except Exception as e:
            self.logger.error(f"분석 요약 생성 오류: {e}")
            return f"""
## 분석 요약 생성 오류

분석 요약을 생성하는 중 오류가 발생했습니다: {str(e)}

기본적인 분석 결과를 검토하시고, 필요시 전문가와 상담하시기 바랍니다.
"""
    
    def _generate_detailed_analysis(self, execution_results: Dict[str, Any],
                                  analysis_insights: Dict[str, Any],
                                  visualizations: Dict[str, Any]) -> Dict[str, Any]:
        """상세 분석 생성"""
        try:
            detailed_analysis = {
                'statistical_analysis': {},
                'data_characteristics': {},
                'assumption_analysis': {},
                'effect_analysis': {},
                'visualization_analysis': {},
                'methodological_notes': {}
            }
            
            statistics = execution_results.get('statistics', {})
            
            # 1. 통계 분석 세부사항
            detailed_analysis['statistical_analysis'] = self._analyze_statistical_details(statistics)
            
            # 2. 데이터 특성 분석
            detailed_analysis['data_characteristics'] = self._analyze_data_characteristics(statistics)
            
            # 3. 가정 분석
            detailed_analysis['assumption_analysis'] = self._analyze_assumptions(statistics)
            
            # 4. 효과 분석
            detailed_analysis['effect_analysis'] = self._analyze_effects(statistics)
            
            # 5. 시각화 분석
            detailed_analysis['visualization_analysis'] = self._analyze_visualizations(visualizations)
            
            # 6. 방법론적 노트
            detailed_analysis['methodological_notes'] = self._generate_methodological_notes(
                execution_results, analysis_insights
            )
            
            return detailed_analysis
            
        except Exception as e:
            self.logger.error(f"상세 분석 생성 오류: {e}")
            return {
                'error': True,
                'error_message': str(e),
                'statistical_analysis': {},
                'data_characteristics': {},
                'assumption_analysis': {},
                'effect_analysis': {},
                'visualization_analysis': {},
                'methodological_notes': {}
            }
    
    def _generate_recommendations(self, detailed_analysis: Dict[str, Any],
                                analysis_insights: Dict[str, Any]) -> List[str]:
        """추천사항 도출"""
        try:
            recommendations = []
            
            # 1. 통계적 권고사항
            statistical_recs = self._generate_statistical_recommendations(detailed_analysis)
            recommendations.extend(statistical_recs)
            
            # 2. 방법론적 권고사항
            methodological_recs = self._generate_methodological_recommendations(detailed_analysis)
            recommendations.extend(methodological_recs)
            
            # 3. 데이터 품질 권고사항
            data_quality_recs = self._generate_data_quality_recommendations(detailed_analysis)
            recommendations.extend(data_quality_recs)
            
            # 4. 후속 분석 권고사항
            follow_up_recs = self._generate_follow_up_recommendations(detailed_analysis)
            recommendations.extend(follow_up_recs)
            
            # 5. 실무적 권고사항
            practical_recs = self._generate_practical_recommendations(
                detailed_analysis, analysis_insights
            )
            recommendations.extend(practical_recs)
            
            # 중복 제거 및 우선순위 정렬
            unique_recommendations = list(set(recommendations))
            prioritized_recommendations = self._prioritize_recommendations(unique_recommendations)
            
            return prioritized_recommendations
            
        except Exception as e:
            self.logger.error(f"추천사항 도출 오류: {e}")
            return [
                "분석 결과를 신중하게 해석하시기 바랍니다.",
                "추가적인 데이터 검토가 필요할 수 있습니다.",
                "전문가와의 상담을 고려해보세요."
            ]
    
    def _calculate_reliability_scores(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """신뢰도 점수 계산"""
        try:
            reliability_scores = {
                'data_quality_score': 0.0,
                'statistical_validity_score': 0.0,
                'assumption_compliance_score': 0.0,
                'effect_reliability_score': 0.0,
                'overall_reliability_score': 0.0
            }
            
            statistics = execution_results.get('statistics', {})
            
            # 1. 데이터 품질 점수
            reliability_scores['data_quality_score'] = self._calculate_data_quality_score(statistics)
            
            # 2. 통계적 타당성 점수
            reliability_scores['statistical_validity_score'] = self._calculate_statistical_validity_score(statistics)
            
            # 3. 가정 준수 점수
            reliability_scores['assumption_compliance_score'] = self._calculate_assumption_compliance_score(statistics)
            
            # 4. 효과 신뢰도 점수
            reliability_scores['effect_reliability_score'] = self._calculate_effect_reliability_score(statistics)
            
            # 5. 전체 신뢰도 점수 (가중평균)
            weights = {
                'data_quality_score': 0.3,
                'statistical_validity_score': 0.3,
                'assumption_compliance_score': 0.2,
                'effect_reliability_score': 0.2
            }
            
            overall_score = sum(
                reliability_scores[key] * weight 
                for key, weight in weights.items()
            )
            reliability_scores['overall_reliability_score'] = overall_score
            
            # 신뢰도 등급 부여
            reliability_scores['reliability_grade'] = self._assign_reliability_grade(overall_score)
            
            # 상세 분석
            reliability_scores['detailed_analysis'] = {
                'strengths': self._identify_reliability_strengths(reliability_scores),
                'weaknesses': self._identify_reliability_weaknesses(reliability_scores),
                'improvement_suggestions': self._suggest_reliability_improvements(reliability_scores)
            }
            
            return reliability_scores
            
        except Exception as e:
            self.logger.error(f"신뢰도 점수 계산 오류: {e}")
            return {
                'data_quality_score': 0.0,
                'statistical_validity_score': 0.0,
                'assumption_compliance_score': 0.0,
                'effect_reliability_score': 0.0,
                'overall_reliability_score': 0.0,
                'reliability_grade': 'F',
                'error': str(e)
            }
    
    def _calculate_validation_metrics(self, execution_results: Dict[str, Any],
                                    statistical_design: Dict[str, Any]) -> Dict[str, Any]:
        """검증 메트릭 계산"""
        try:
            validation_metrics = {
                'power_analysis': {},
                'sensitivity_analysis': {},
                'robustness_analysis': {},
                'cross_validation': {},
                'bootstrap_analysis': {}
            }
            
            statistics = execution_results.get('statistics', {})
            test_results = statistics.get('test_results', {})
            
            # 1. 검정력 분석
            validation_metrics['power_analysis'] = self._perform_power_analysis(
                test_results, statistical_design
            )
            
            # 2. 민감도 분석
            validation_metrics['sensitivity_analysis'] = self._perform_sensitivity_analysis(
                statistics, statistical_design
            )
            
            # 3. 강건성 분석
            validation_metrics['robustness_analysis'] = self._perform_robustness_analysis(
                statistics, statistical_design
            )
            
            # 4. 교차 검증 (해당하는 경우)
            if self._is_cross_validation_applicable(statistical_design):
                validation_metrics['cross_validation'] = self._perform_cross_validation(
                    statistics, statistical_design
                )
            
            # 5. 부트스트랩 분석
            validation_metrics['bootstrap_analysis'] = self._perform_bootstrap_analysis(
                test_results, statistical_design
            )
            
            # 전체 검증 점수
            validation_metrics['overall_validation_score'] = self._calculate_overall_validation_score(
                validation_metrics
            )
            
            return validation_metrics
            
        except Exception as e:
            self.logger.error(f"검증 메트릭 계산 오류: {e}")
            return {
                'power_analysis': {'power': 0.0, 'message': '계산 실패'},
                'sensitivity_analysis': {'stable': False, 'message': '계산 실패'},
                'robustness_analysis': {'robust': False, 'message': '계산 실패'},
                'overall_validation_score': 0.0,
                'error': str(e)
            }
    
    def _calculate_confidence_intervals(self, execution_results: Dict[str, Any],
                                     statistical_design: Dict[str, Any]) -> Dict[str, Any]:
        """신뢰구간 계산"""
        try:
            confidence_intervals = {}
            
            statistics = execution_results.get('statistics', {})
            test_results = statistics.get('test_results', {})
            
            # 기본 신뢰수준들
            confidence_levels = [0.90, 0.95, 0.99]
            
            # 1. 모수 신뢰구간
            if 'parameter_estimates' in test_results:
                confidence_intervals['parameter_intervals'] = {}
                for level in confidence_levels:
                    confidence_intervals['parameter_intervals'][f'{int(level*100)}%'] = \
                        self._calculate_parameter_confidence_intervals(test_results, level)
            
            # 2. 예측 신뢰구간 (회귀분석인 경우)
            if self._is_regression_analysis(statistical_design):
                confidence_intervals['prediction_intervals'] = {}
                for level in confidence_levels:
                    confidence_intervals['prediction_intervals'][f'{int(level*100)}%'] = \
                        self._calculate_prediction_intervals(test_results, level)
            
            # 3. 효과크기 신뢰구간
            if 'effect_size' in test_results:
                confidence_intervals['effect_size_intervals'] = {}
                for level in confidence_levels:
                    confidence_intervals['effect_size_intervals'][f'{int(level*100)}%'] = \
                        self._calculate_effect_size_confidence_intervals(test_results, level)
            
            # 4. 부트스트랩 신뢰구간
            confidence_intervals['bootstrap_intervals'] = self._calculate_bootstrap_confidence_intervals(
                test_results, confidence_levels
            )
            
            # 5. 베이지안 신용구간 (해당하는 경우)
            if statistical_design.get('bayesian_analysis', False):
                confidence_intervals['credible_intervals'] = self._calculate_credible_intervals(
                    test_results, confidence_levels
                )
            
            # 신뢰구간 해석
            confidence_intervals['interpretation'] = self._interpret_confidence_intervals(
                confidence_intervals, statistical_design
            )
            
            return confidence_intervals
            
        except Exception as e:
            self.logger.error(f"신뢰구간 계산 오류: {e}")
            return {
                'parameter_intervals': {},
                'effect_size_intervals': {},
                'bootstrap_intervals': {},
                'interpretation': f"신뢰구간 계산 중 오류 발생: {str(e)}",
                'error': str(e)
            }
    
    def get_step_info(self) -> Dict[str, Any]:
        """단계 정보 반환 (부모 클래스 메서드 확장)"""
        base_info = super().get_step_info()
        base_info.update({
            'description': 'Agentic LLM의 통계 분석 실행 및 결과 해석',
            'input_requirements': [
                'analysis_code', 'execution_plan', 'data_requirements',
                'statistical_design', 'visualization_plan', 'documentation'
            ],
            'output_provides': [
                'execution_results', 'analysis_insights', 'visualizations',
                'interpretation', 'quality_metrics', 'execution_metadata'
            ],
            'capabilities': [
                '통계 분석 실행', 'RAG 기반 결과 해석', '시각화 생성',
                '결과 해석 및 요약', '품질 메트릭 계산', '실행 메타데이터 수집'
            ]
        })
        return base_info

    # 헬퍼 메서드들
    def _interpret_effect_size(self, effect_size: float, test_type: str) -> str:
        """효과크기 해석"""
        try:
            if test_type.lower() in ['t_test', 't-test']:
                # Cohen's d 기준
                if abs(effect_size) < 0.2:
                    return "작은 효과"
                elif abs(effect_size) < 0.5:
                    return "중간 효과"
                elif abs(effect_size) < 0.8:
                    return "큰 효과"
                else:
                    return "매우 큰 효과"
            elif 'correlation' in test_type.lower():
                # 상관계수 기준
                if abs(effect_size) < 0.1:
                    return "무시할 만한 상관"
                elif abs(effect_size) < 0.3:
                    return "약한 상관"
                elif abs(effect_size) < 0.5:
                    return "중간 상관"
                elif abs(effect_size) < 0.7:
                    return "강한 상관"
                else:
                    return "매우 강한 상관"
            elif 'regression' in test_type.lower():
                # R² 기준
                if effect_size < 0.02:
                    return "작은 설명력"
                elif effect_size < 0.13:
                    return "중간 설명력"
                elif effect_size < 0.26:
                    return "큰 설명력"
                else:
                    return "매우 큰 설명력"
            else:
                # 일반적인 기준
                if abs(effect_size) < 0.1:
                    return "작은 효과"
                elif abs(effect_size) < 0.3:
                    return "중간 효과"
                else:
                    return "큰 효과"
                    
        except Exception:
            return "해석 불가"
    
    def _assess_sample_adequacy(self, statistics: Dict[str, Any]) -> float:
        """샘플 적절성 평가"""
        try:
            desc_stats = statistics.get('descriptive_stats', {})
            sample_size = desc_stats.get('count', 0)
            
            # 기본 샘플 크기 기준
            if sample_size >= 100:
                return 1.0  # 매우 적절
            elif sample_size >= 50:
                return 0.8  # 적절
            elif sample_size >= 30:
                return 0.6  # 보통
            elif sample_size >= 10:
                return 0.4  # 부족
            else:
                return 0.2  # 매우 부족
                
        except Exception:
            return 0.0
    
    def _calculate_assumption_score(self, assumption_validity: Dict[str, Any]) -> float:
        """가정 점수 계산"""
        try:
            if not assumption_validity:
                return 0.5  # 가정 검증 안 됨
            
            passed_count = sum(1 for test in assumption_validity.values() if test.get('passed', False))
            total_count = len(assumption_validity)
            
            return passed_count / total_count if total_count > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_overall_confidence(self, metrics: Dict[str, Any]) -> float:
        """전체 신뢰도 계산"""
        try:
            data_quality = metrics.get('data_quality', {}).get('completeness', 0.0)
            reliability = metrics.get('reliability', {})
            sample_adequacy = reliability.get('sample_adequacy', 0.0)
            assumption_score = reliability.get('assumption_score', 0.0)
            
            # 가중평균
            weights = [0.3, 0.4, 0.3]  # 데이터품질, 샘플적절성, 가정점수
            scores = [data_quality, sample_adequacy, assumption_score]
            
            return sum(w * s for w, s in zip(weights, scores)) / sum(weights)
            
        except Exception:
            return 0.0
    
    def _validate_data_completeness(self, statistics: Dict[str, Any], check: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 완성도 검증"""
        try:
            desc_stats = statistics.get('descriptive_stats', {})
            completeness = desc_stats.get('completeness_ratio', 0.0)
            threshold = check.get('threshold', 0.95)
            
            passed = completeness >= threshold
            
            return {
                'passed': passed,
                'message': f"데이터 완성도: {completeness:.2%} (기준: {threshold:.2%})",
                'details': {
                    'completeness_ratio': completeness,
                    'threshold': threshold,
                    'missing_count': desc_stats.get('missing_count', 0)
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'message': f"데이터 완성도 검증 오류: {str(e)}",
                'details': {'error': str(e)}
            }
    
    def _validate_outliers(self, statistics: Dict[str, Any], check: Dict[str, Any]) -> Dict[str, Any]:
        """이상치 검증"""
        try:
            outliers = statistics.get('outliers', {})
            outlier_count = outliers.get('count', 0)
            total_count = statistics.get('descriptive_stats', {}).get('count', 1)
            outlier_ratio = outlier_count / total_count
            
            threshold = check.get('threshold', 0.05)  # 5% 기준
            passed = outlier_ratio <= threshold
            
            return {
                'passed': passed,
                'message': f"이상치 비율: {outlier_ratio:.2%} (기준: {threshold:.2%})",
                'details': {
                    'outlier_count': outlier_count,
                    'total_count': total_count,
                    'outlier_ratio': outlier_ratio,
                    'outlier_indices': outliers.get('indices', [])
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'message': f"이상치 검증 오류: {str(e)}",
                'details': {'error': str(e)}
            }
    
    def _validate_normality(self, statistics: Dict[str, Any], check: Dict[str, Any]) -> Dict[str, Any]:
        """정규성 검증"""
        try:
            assumption_tests = statistics.get('assumption_tests', {})
            normality_tests = ['shapiro_wilk', 'kolmogorov_smirnov', 'anderson_darling']
            
            for test_name in normality_tests:
                if test_name in assumption_tests:
                    test_result = assumption_tests[test_name]
                    p_value = test_result.get('p_value', 0.0)
                    alpha = check.get('alpha', 0.05)
                    
                    passed = p_value > alpha  # 귀무가설: 정규분포
                    
                    return {
                        'passed': passed,
                        'message': f"{test_name} 검정: p-value = {p_value:.4f} (α = {alpha})",
                        'details': {
                            'test_name': test_name,
                            'p_value': p_value,
                            'statistic': test_result.get('statistic', 0.0),
                            'alpha': alpha
                        }
                    }
            
            return {
                'passed': False,
                'message': "정규성 검정 결과를 찾을 수 없습니다.",
                'details': {'available_tests': list(assumption_tests.keys())}
            }
            
        except Exception as e:
            return {
                'passed': False,
                'message': f"정규성 검증 오류: {str(e)}",
                'details': {'error': str(e)}
            }
    
    def _validate_independence(self, statistics: Dict[str, Any], check: Dict[str, Any]) -> Dict[str, Any]:
        """독립성 검증"""
        try:
            assumption_tests = statistics.get('assumption_tests', {})
            
            if 'durbin_watson' in assumption_tests:
                dw_result = assumption_tests['durbin_watson']
                dw_statistic = dw_result.get('statistic', 2.0)
                
                # Durbin-Watson 통계량 해석 (1.5-2.5가 적절)
                passed = 1.5 <= dw_statistic <= 2.5
                
                return {
                    'passed': passed,
                    'message': f"Durbin-Watson 검정: {dw_statistic:.3f} (기준: 1.5-2.5)",
                    'details': {
                        'durbin_watson_statistic': dw_statistic,
                        'interpretation': self._interpret_durbin_watson(dw_statistic)
                    }
                }
            
            # 독립성 검정이 없는 경우 일반적인 가정
            return {
                'passed': True,  # 기본적으로 독립성 가정
                'message': "독립성 가정 (별도 검정 없음)",
                'details': {'assumption': 'independence_assumed'}
            }
            
        except Exception as e:
            return {
                'passed': False,
                'message': f"독립성 검증 오류: {str(e)}",
                'details': {'error': str(e)}
            }
    
    def _validate_homoscedasticity(self, statistics: Dict[str, Any], check: Dict[str, Any]) -> Dict[str, Any]:
        """등분산성 검증"""
        try:
            assumption_tests = statistics.get('assumption_tests', {})
            homoscedasticity_tests = ['levene', 'bartlett', 'breusch_pagan']
            
            for test_name in homoscedasticity_tests:
                if test_name in assumption_tests:
                    test_result = assumption_tests[test_name]
                    p_value = test_result.get('p_value', 0.0)
                    alpha = check.get('alpha', 0.05)
                    
                    passed = p_value > alpha  # 귀무가설: 등분산성
                    
                    return {
                        'passed': passed,
                        'message': f"{test_name} 검정: p-value = {p_value:.4f} (α = {alpha})",
                        'details': {
                            'test_name': test_name,
                            'p_value': p_value,
                            'statistic': test_result.get('statistic', 0.0),
                            'alpha': alpha
                        }
                    }
            
            return {
                'passed': False,
                'message': "등분산성 검정 결과를 찾을 수 없습니다.",
                'details': {'available_tests': list(assumption_tests.keys())}
            }
            
        except Exception as e:
            return {
                'passed': False,
                'message': f"등분산성 검증 오류: {str(e)}",
                'details': {'error': str(e)}
            }
    
    def _validate_linearity(self, statistics: Dict[str, Any], check: Dict[str, Any]) -> Dict[str, Any]:
        """선형성 검증"""
        try:
            assumption_tests = statistics.get('assumption_tests', {})
            
            if 'linearity_test' in assumption_tests:
                linearity_result = assumption_tests['linearity_test']
                p_value = linearity_result.get('p_value', 0.0)
                alpha = check.get('alpha', 0.05)
                
                passed = p_value > alpha
                
                return {
                    'passed': passed,
                    'message': f"선형성 검정: p-value = {p_value:.4f} (α = {alpha})",
                    'details': {
                        'p_value': p_value,
                        'statistic': linearity_result.get('statistic', 0.0),
                        'alpha': alpha
                    }
                }
            
            # R² 기반 선형성 평가
            model_summary = statistics.get('model_summary', {})
            r_squared = model_summary.get('r_squared', 0.0)
            
            threshold = check.get('r_squared_threshold', 0.1)
            passed = r_squared >= threshold
            
            return {
                'passed': passed,
                'message': f"모델 설명력 기반 선형성: R² = {r_squared:.3f} (기준: {threshold})",
                'details': {
                    'r_squared': r_squared,
                    'threshold': threshold,
                    'method': 'r_squared_based'
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'message': f"선형성 검증 오류: {str(e)}",
                'details': {'error': str(e)}
            }
    
    def _validate_multicollinearity(self, statistics: Dict[str, Any], check: Dict[str, Any]) -> Dict[str, Any]:
        """다중공선성 검증"""
        try:
            model_diagnostics = statistics.get('model_diagnostics', {})
            
            if 'vif_values' in model_diagnostics:
                vif_values = model_diagnostics['vif_values']
                threshold = check.get('vif_threshold', 10.0)
                
                max_vif = max(vif_values.values()) if vif_values else 0.0
                passed = max_vif < threshold
                
                return {
                    'passed': passed,
                    'message': f"VIF 검정: 최대 VIF = {max_vif:.2f} (기준: < {threshold})",
                    'details': {
                        'vif_values': vif_values,
                        'max_vif': max_vif,
                        'threshold': threshold
                    }
                }
            
            return {
                'passed': True,  # VIF 값이 없으면 기본 통과
                'message': "다중공선성 검정 결과 없음 (단순 모델로 추정)",
                'details': {'method': 'not_applicable'}
            }
            
        except Exception as e:
            return {
                'passed': False,
                'message': f"다중공선성 검증 오류: {str(e)}",
                'details': {'error': str(e)}
            }
    
    def _validate_sample_size(self, statistics: Dict[str, Any], check: Dict[str, Any]) -> Dict[str, Any]:
        """샘플 크기 검증"""
        try:
            desc_stats = statistics.get('descriptive_stats', {})
            sample_size = desc_stats.get('count', 0)
            
            min_sample_size = check.get('min_sample_size', 30)
            passed = sample_size >= min_sample_size
            
            return {
                'passed': passed,
                'message': f"샘플 크기: {sample_size} (최소 기준: {min_sample_size})",
                'details': {
                    'sample_size': sample_size,
                    'min_required': min_sample_size,
                    'adequacy': self._assess_sample_adequacy(statistics)
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'message': f"샘플 크기 검증 오류: {str(e)}",
                'details': {'error': str(e)}
            }
    
    def _interpret_durbin_watson(self, dw_statistic: float) -> str:
        """Durbin-Watson 통계량 해석"""
        if dw_statistic < 1.5:
            return "양의 자기상관 의심"
        elif dw_statistic > 2.5:
            return "음의 자기상관 의심"
        else:
            return "자기상관 없음 (독립성 만족)"

    def _fallback_interpretation_parsing(self, text: str) -> Dict[str, Any]:
        """JSON 파싱 실패시 해석 결과 기본 파싱"""
        try:
            # 기본 구조 생성
            interpretation = {
                'statistical_significance': '',
                'practical_significance': '',
                'reliability_assessment': '',
                'limitations': [],
                'follow_up_suggestions': [],
                'key_insights': [],
                'conclusion': ''
            }
            
            # 텍스트를 문장 단위로 분할
            sentences = re.split(r'[.!?]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # 키워드 기반 분류
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                # 통계적 유의성 관련
                if any(keyword in sentence_lower for keyword in ['p-value', 'p값', '유의수준', '통계적', 'significant']):
                    if not interpretation['statistical_significance']:
                        interpretation['statistical_significance'] = sentence
                
                # 실무적 유의성 관련
                elif any(keyword in sentence_lower for keyword in ['실무', '실용', '효과크기', 'effect size', '실제']):
                    if not interpretation['practical_significance']:
                        interpretation['practical_significance'] = sentence
                
                # 한계점 관련
                elif any(keyword in sentence_lower for keyword in ['한계', '제한', 'limitation', '주의']):
                    interpretation['limitations'].append(sentence)
                
                # 제안사항 관련
                elif any(keyword in sentence_lower for keyword in ['제안', '권고', '추천', 'recommend', '후속']):
                    interpretation['follow_up_suggestions'].append(sentence)
                
                # 핵심 인사이트 관련
                elif any(keyword in sentence_lower for keyword in ['중요', '핵심', '주목', '결과적']):
                    interpretation['key_insights'].append(sentence)
            
            # 결론은 마지막 문장들 중에서
            if sentences:
                interpretation['conclusion'] = sentences[-1]
            
            # 빈 필드들에 대한 기본값 설정
            if not interpretation['statistical_significance']:
                interpretation['statistical_significance'] = "통계 분석 결과를 참조하세요."
            
            if not interpretation['practical_significance']:
                interpretation['practical_significance'] = "효과크기와 실무적 중요성을 고려하세요."
            
            if not interpretation['reliability_assessment']:
                interpretation['reliability_assessment'] = "분석 결과의 신뢰성을 검토하세요."
            
            if not interpretation['limitations']:
                interpretation['limitations'] = ["분석 결과 해석시 주의가 필요합니다."]
            
            if not interpretation['follow_up_suggestions']:
                interpretation['follow_up_suggestions'] = ["추가 분석이나 전문가 상담을 고려하세요."]
            
            if not interpretation['conclusion']:
                interpretation['conclusion'] = "분석 결과에 대한 종합적인 검토가 필요합니다."
            
            return interpretation
            
        except Exception as e:
            self.logger.error(f"Fallback 해석 파싱 오류: {e}")
            return {
                'statistical_significance': '파싱 오류로 인해 해석을 생성할 수 없습니다.',
                'practical_significance': '전문가 상담을 권장합니다.',
                'reliability_assessment': '결과 신뢰성 검토 필요',
                'limitations': ['응답 파싱 중 오류 발생'],
                'follow_up_suggestions': ['전문가 상담 권장'],
                'key_insights': ['파싱 오류 발생'],
                'conclusion': '결과 해석에 주의가 필요합니다.',
                'error': str(e)
            }


# 단계 등록
PipelineStepRegistry.register_step(7, AgentExecutionStep) 