"""
Decision Tree

지능적 의사결정 트리 (분석 방법 선택, 파라미터 최적화)
- 데이터 특성 기반 분석 방법 선택
- 통계적 가정 검증 및 대안 제시
- 파라미터 최적화 및 적응적 전략 생성
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
from datetime import datetime

from utils.error_handler import handle_error, StatisticalException


class DecisionCriteria(Enum):
    """의사결정 기준"""
    DATA_TYPE = "data_type"
    SAMPLE_SIZE = "sample_size"
    NORMALITY = "normality"
    HOMOSCEDASTICITY = "homoscedasticity"
    INDEPENDENCE = "independence"
    LINEARITY = "linearity"
    ANALYSIS_GOAL = "analysis_goal"
    USER_PREFERENCE = "user_preference"


class AnalysisMethod(Enum):
    """분석 방법"""
    DESCRIPTIVE = "descriptive"
    T_TEST = "t_test"
    ANOVA = "anova"
    CHI_SQUARE = "chi_square"
    CORRELATION = "correlation"
    REGRESSION = "regression"
    NONPARAMETRIC = "nonparametric"
    TIME_SERIES = "time_series"


@dataclass
class DecisionNode:
    """의사결정 노드"""
    node_id: str
    criteria: DecisionCriteria
    condition: str
    threshold: Optional[Union[float, int, str]] = None
    children: List['DecisionNode'] = field(default_factory=list)
    recommendation: Optional[AnalysisMethod] = None
    confidence: float = 0.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionPath:
    """의사결정 경로"""
    nodes: List[DecisionNode]
    final_recommendation: AnalysisMethod
    confidence_score: float
    reasoning_chain: List[str]
    alternative_paths: List['DecisionPath'] = field(default_factory=list)


@dataclass
class AnalysisPlan:
    """분석 계획"""
    primary_method: AnalysisMethod
    steps: List[Dict[str, Any]]
    assumptions_to_check: List[str]
    alternative_methods: List[AnalysisMethod]
    parameters: Dict[str, Any]
    expected_outputs: List[str]
    confidence: float
    reasoning: str


class DecisionTree:
    """지능적 의사결정 트리 (분석 방법 선택, 파라미터 최적화)"""
    
    def __init__(self):
        """DecisionTree 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 의사결정 트리 구조
        self.root_node: Optional[DecisionNode] = None
        self.decision_history: List[DecisionPath] = []
        
        # 학습된 패턴
        self.learned_patterns: Dict[str, Any] = {}
        self.success_patterns: List[Dict[str, Any]] = []
        self.failure_patterns: List[Dict[str, Any]] = []
        
        # 설정
        self.min_confidence_threshold = 0.6
        self.max_alternatives = 3
        
        # 트리 구축
        self._build_decision_tree()
        
        self.logger.info("DecisionTree 초기화 완료")
    
    async def prepare(self):
        """의사결정 트리 준비"""
        self.logger.info("의사결정 트리 준비 중...")
        
        try:
            # 학습된 패턴 로드
            await self._load_learned_patterns()
            
            # 트리 최적화
            await self._optimize_tree()
            
            self.logger.info("의사결정 트리 준비 완료")
            
        except Exception as e:
            self.logger.error(f"의사결정 트리 준비 오류: {e}")
            raise
    
    async def make_decision(self, context: Dict[str, Any]) -> DecisionPath:
        """
        의사결정 수행
        
        Args:
            context: 의사결정 컨텍스트
            
        Returns:
            DecisionPath: 의사결정 경로
        """
        self.logger.info("의사결정 수행 시작")
        
        try:
            # 컨텍스트 분석
            analyzed_context = await self._analyze_context(context)
            
            # 트리 탐색
            decision_path = await self._traverse_tree(analyzed_context)
            
            # 대안 경로 생성
            alternatives = await self._generate_alternatives(analyzed_context, decision_path)
            decision_path.alternative_paths = alternatives
            
            # 의사결정 기록
            self.decision_history.append(decision_path)
            
            self.logger.info(f"의사결정 완료: {decision_path.final_recommendation.value}")
            return decision_path
            
        except Exception as e:
            self.logger.error(f"의사결정 오류: {e}")
            raise
    
    async def create_analysis_plan(self, context: Dict[str, Any]) -> AnalysisPlan:
        """
        분석 계획 생성
        
        Args:
            context: 분석 컨텍스트
            
        Returns:
            AnalysisPlan: 분석 계획
        """
        self.logger.info("분석 계획 생성 시작")
        
        try:
            # 의사결정 수행
            decision_path = await self.make_decision(context)
            
            # 분석 계획 구성
            plan = await self._build_analysis_plan(decision_path, context)
            
            self.logger.info(f"분석 계획 생성 완료: {plan.primary_method.value}")
            return plan
            
        except Exception as e:
            self.logger.error(f"분석 계획 생성 오류: {e}")
            raise
    
    async def optimize_parameters(self, method: AnalysisMethod, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        파라미터 최적화
        
        Args:
            method: 분석 방법
            context: 컨텍스트
            
        Returns:
            Dict: 최적화된 파라미터
        """
        self.logger.info(f"파라미터 최적화 시작: {method.value}")
        
        try:
            # 기본 파라미터 설정
            base_params = self._get_default_parameters(method)
            
            # 컨텍스트 기반 조정
            optimized_params = await self._optimize_for_context(base_params, context)
            
            # 학습된 패턴 적용
            final_params = await self._apply_learned_optimizations(optimized_params, method, context)
            
            self.logger.info("파라미터 최적화 완료")
            return final_params
            
        except Exception as e:
            self.logger.error(f"파라미터 최적화 오류: {e}")
            raise
    
    async def learn_from_outcome(self, decision_path: DecisionPath, 
                               outcome: Dict[str, Any]):
        """
        결과로부터 학습
        
        Args:
            decision_path: 의사결정 경로
            outcome: 결과
        """
        self.logger.info("결과로부터 학습 시작")
        
        try:
            # 성공/실패 패턴 분석
            pattern = self._extract_pattern(decision_path, outcome)
            
            if outcome.get('success', False):
                self.success_patterns.append(pattern)
                await self._update_success_weights(decision_path)
            else:
                self.failure_patterns.append(pattern)
                await self._update_failure_weights(decision_path)
            
            # 학습된 패턴 업데이트
            await self._update_learned_patterns(pattern)
            
            self.logger.info("학습 완료")
            
        except Exception as e:
            self.logger.error(f"학습 오류: {e}")
    
    async def _analyze_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """컨텍스트 분석"""
        analyzed = {
            'original_context': context,
            'data_characteristics': {},
            'statistical_properties': {},
            'user_requirements': {},
            'constraints': []
        }
        
        # 데이터 특성 분석
        if 'data' in context:
            analyzed['data_characteristics'] = await self._analyze_data_characteristics(context['data'])
        
        # 통계적 속성 분석
        if 'data' in context:
            analyzed['statistical_properties'] = await self._analyze_statistical_properties(context['data'])
        
        # 사용자 요구사항 분석
        if 'user_requirements' in context:
            analyzed['user_requirements'] = await self._analyze_user_requirements(context['user_requirements'])
        
        # 제약사항 분석
        if 'constraints' in context:
            analyzed['constraints'] = context['constraints']
        
        return analyzed
    
    async def _traverse_tree(self, context: Dict[str, Any]) -> DecisionPath:
        """트리 탐색"""
        current_node = self.root_node
        path_nodes = []
        reasoning_chain = []
        
        while current_node:
            path_nodes.append(current_node)
            
            # 노드 조건 평가
            evaluation_result = await self._evaluate_node_condition(current_node, context)
            reasoning_chain.append(evaluation_result['reasoning'])
            
            # 다음 노드 선택
            next_node = await self._select_next_node(current_node, evaluation_result, context)
            
            if next_node is None or current_node.recommendation:
                break
            
            current_node = next_node
        
        # 최종 추천 결정
        final_recommendation = current_node.recommendation if current_node else AnalysisMethod.DESCRIPTIVE
        confidence = await self._calculate_path_confidence(path_nodes, context)
        
        return DecisionPath(
            nodes=path_nodes,
            final_recommendation=final_recommendation,
            confidence_score=confidence,
            reasoning_chain=reasoning_chain
        )
    
    async def _evaluate_node_condition(self, node: DecisionNode, 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """노드 조건 평가"""
        criteria = node.criteria
        condition = node.condition
        threshold = node.threshold
        
        if criteria == DecisionCriteria.DATA_TYPE:
            return await self._evaluate_data_type_condition(condition, threshold, context)
        elif criteria == DecisionCriteria.SAMPLE_SIZE:
            return await self._evaluate_sample_size_condition(condition, threshold, context)
        elif criteria == DecisionCriteria.NORMALITY:
            return await self._evaluate_normality_condition(condition, threshold, context)
        elif criteria == DecisionCriteria.ANALYSIS_GOAL:
            return await self._evaluate_analysis_goal_condition(condition, threshold, context)
        else:
            return {'result': True, 'reasoning': f'기본 조건 통과: {criteria.value}'}
    
    async def _select_next_node(self, current_node: DecisionNode, 
                              evaluation_result: Dict[str, Any],
                              context: Dict[str, Any]) -> Optional[DecisionNode]:
        """다음 노드 선택"""
        if not current_node.children:
            return None
        
        # 조건 결과에 따른 자식 노드 선택
        if evaluation_result.get('result', False):
            # 조건이 참인 경우 첫 번째 자식
            return current_node.children[0] if current_node.children else None
        else:
            # 조건이 거짓인 경우 두 번째 자식 (있다면)
            return current_node.children[1] if len(current_node.children) > 1 else None
    
    async def _generate_alternatives(self, context: Dict[str, Any], 
                                   primary_path: DecisionPath) -> List[DecisionPath]:
        """대안 경로 생성"""
        alternatives = []
        
        # 다른 분석 방법들 고려
        all_methods = list(AnalysisMethod)
        primary_method = primary_path.final_recommendation
        
        for method in all_methods:
            if method != primary_method and len(alternatives) < self.max_alternatives:
                # 대안 경로 생성
                alt_path = await self._create_alternative_path(method, context)
                if alt_path.confidence_score >= self.min_confidence_threshold:
                    alternatives.append(alt_path)
        
        # 신뢰도 순으로 정렬
        alternatives.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return alternatives[:self.max_alternatives]
    
    async def _build_analysis_plan(self, decision_path: DecisionPath, 
                                 context: Dict[str, Any]) -> AnalysisPlan:
        """분석 계획 구축"""
        method = decision_path.final_recommendation
        
        # 기본 단계 정의
        steps = await self._define_analysis_steps(method, context)
        
        # 가정 검증 목록
        assumptions = await self._identify_assumptions_to_check(method, context)
        
        # 대안 방법들
        alternatives = [path.final_recommendation for path in decision_path.alternative_paths]
        
        # 파라미터 최적화
        parameters = await self.optimize_parameters(method, context)
        
        # 예상 출력
        expected_outputs = await self._define_expected_outputs(method, context)
        
        return AnalysisPlan(
            primary_method=method,
            steps=steps,
            assumptions_to_check=assumptions,
            alternative_methods=alternatives,
            parameters=parameters,
            expected_outputs=expected_outputs,
            confidence=decision_path.confidence_score,
            reasoning=" -> ".join(decision_path.reasoning_chain)
        )
    
    def _build_decision_tree(self):
        """의사결정 트리 구축"""
        # 루트 노드: 분석 목표
        self.root_node = DecisionNode(
            node_id="root",
            criteria=DecisionCriteria.ANALYSIS_GOAL,
            condition="analysis_type",
            reasoning="분석 목표에 따른 방법 선택"
        )
        
        # 비교 분석 브랜치
        comparison_node = DecisionNode(
            node_id="comparison",
            criteria=DecisionCriteria.DATA_TYPE,
            condition="numeric_data",
            reasoning="수치형 데이터 여부 확인"
        )
        
        # 수치형 비교 분석
        numeric_comparison_node = DecisionNode(
            node_id="numeric_comparison",
            criteria=DecisionCriteria.SAMPLE_SIZE,
            condition="sample_size",
            threshold=30,
            reasoning="표본 크기에 따른 방법 선택"
        )
        
        # 대표본 t-test/ANOVA
        large_sample_node = DecisionNode(
            node_id="large_sample",
            criteria=DecisionCriteria.NORMALITY,
            condition="normality_test",
            threshold=0.05,
            reasoning="정규성 검정 결과에 따른 선택"
        )
        
        # 정규성 만족 시 parametric test
        parametric_node = DecisionNode(
            node_id="parametric",
            criteria=DecisionCriteria.ANALYSIS_GOAL,
            condition="group_count",
            threshold=2,
            recommendation=AnalysisMethod.T_TEST,
            reasoning="두 그룹 비교는 t-test"
        )
        
        # 다중 그룹 비교
        anova_node = DecisionNode(
            node_id="anova",
            criteria=DecisionCriteria.ANALYSIS_GOAL,
            condition="group_count",
            recommendation=AnalysisMethod.ANOVA,
            reasoning="다중 그룹 비교는 ANOVA"
        )
        
        # 정규성 불만족 시 nonparametric test
        nonparametric_node = DecisionNode(
            node_id="nonparametric",
            criteria=DecisionCriteria.ANALYSIS_GOAL,
            condition="group_count",
            recommendation=AnalysisMethod.NONPARAMETRIC,
            reasoning="정규성 불만족 시 비모수 검정"
        )
        
        # 소표본 nonparametric
        small_sample_node = DecisionNode(
            node_id="small_sample",
            criteria=DecisionCriteria.ANALYSIS_GOAL,
            condition="group_count",
            recommendation=AnalysisMethod.NONPARAMETRIC,
            reasoning="소표본은 비모수 검정"
        )
        
        # 범주형 비교 분석
        categorical_comparison_node = DecisionNode(
            node_id="categorical_comparison",
            criteria=DecisionCriteria.ANALYSIS_GOAL,
            condition="independence_test",
            recommendation=AnalysisMethod.CHI_SQUARE,
            reasoning="범주형 데이터는 카이제곱 검정"
        )
        
        # 관계 분석 브랜치
        relationship_node = DecisionNode(
            node_id="relationship",
            criteria=DecisionCriteria.DATA_TYPE,
            condition="numeric_data",
            reasoning="수치형 데이터 여부 확인"
        )
        
        # 상관관계 분석
        correlation_node = DecisionNode(
            node_id="correlation",
            criteria=DecisionCriteria.LINEARITY,
            condition="linearity_test",
            recommendation=AnalysisMethod.CORRELATION,
            reasoning="선형 관계 분석"
        )
        
        # 회귀 분석
        regression_node = DecisionNode(
            node_id="regression",
            criteria=DecisionCriteria.ANALYSIS_GOAL,
            condition="prediction_goal",
            recommendation=AnalysisMethod.REGRESSION,
            reasoning="예측 목적의 회귀 분석"
        )
        
        # 기술통계 (기본값)
        descriptive_node = DecisionNode(
            node_id="descriptive",
            criteria=DecisionCriteria.ANALYSIS_GOAL,
            condition="default",
            recommendation=AnalysisMethod.DESCRIPTIVE,
            reasoning="기본 기술통계 분석"
        )
        
        # 트리 구조 연결
        self.root_node.children = [comparison_node, relationship_node, descriptive_node]
        
        comparison_node.children = [numeric_comparison_node, categorical_comparison_node]
        numeric_comparison_node.children = [large_sample_node, small_sample_node]
        large_sample_node.children = [parametric_node, nonparametric_node]
        parametric_node.children = [anova_node]
        
        relationship_node.children = [correlation_node, regression_node]
    
    # 조건 평가 메서드들
    async def _evaluate_data_type_condition(self, condition: str, threshold: Any, 
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 타입 조건 평가"""
        data_chars = context.get('data_characteristics', {})
        
        if condition == "numeric_data":
            numeric_columns = data_chars.get('numeric_columns', [])
            result = len(numeric_columns) > 0
            reasoning = f"수치형 컬럼 {len(numeric_columns)}개 발견"
        else:
            result = True
            reasoning = f"데이터 타입 조건: {condition}"
        
        return {'result': result, 'reasoning': reasoning}
    
    async def _evaluate_sample_size_condition(self, condition: str, threshold: Any,
                                            context: Dict[str, Any]) -> Dict[str, Any]:
        """표본 크기 조건 평가"""
        data_chars = context.get('data_characteristics', {})
        sample_size = data_chars.get('sample_size', 0)
        
        if condition == "sample_size" and threshold:
            result = sample_size >= threshold
            reasoning = f"표본 크기 {sample_size} (기준: {threshold})"
        else:
            result = True
            reasoning = f"표본 크기: {sample_size}"
        
        return {'result': result, 'reasoning': reasoning}
    
    async def _evaluate_normality_condition(self, condition: str, threshold: Any,
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """정규성 조건 평가"""
        stats_props = context.get('statistical_properties', {})
        normality_p = stats_props.get('normality_p_value', 1.0)
        
        if condition == "normality_test" and threshold:
            result = normality_p > threshold
            reasoning = f"정규성 검정 p-value: {normality_p:.4f} (기준: {threshold})"
        else:
            result = True
            reasoning = f"정규성 p-value: {normality_p:.4f}"
        
        return {'result': result, 'reasoning': reasoning}
    
    async def _evaluate_analysis_goal_condition(self, condition: str, threshold: Any,
                                              context: Dict[str, Any]) -> Dict[str, Any]:
        """분석 목표 조건 평가"""
        user_reqs = context.get('user_requirements', {})
        analysis_type = user_reqs.get('analysis_type', 'unknown')
        
        if condition == "analysis_type":
            if analysis_type in ['comparison', 'group_comparison']:
                result = True
                reasoning = "그룹 비교 분석 요청"
            elif analysis_type in ['relationship', 'correlation']:
                result = False
                reasoning = "관계 분석 요청"
            else:
                result = False
                reasoning = "기본 분석 요청"
        elif condition == "group_count":
            group_count = user_reqs.get('group_count', 2)
            result = group_count > threshold if threshold else True
            reasoning = f"그룹 수: {group_count}"
        else:
            result = True
            reasoning = f"분석 목표: {analysis_type}"
        
        return {'result': result, 'reasoning': reasoning}
    
    # 헬퍼 메서드들
    async def _analyze_data_characteristics(self, data) -> Dict[str, Any]:
        """데이터 특성 분석"""
        if data is None:
            return {}
        
        characteristics = {}
        
        if hasattr(data, 'shape'):
            characteristics['sample_size'] = data.shape[0]
            characteristics['feature_count'] = data.shape[1]
        
        if hasattr(data, 'dtypes'):
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            characteristics['numeric_columns'] = numeric_columns
            characteristics['categorical_columns'] = categorical_columns
            characteristics['numeric_count'] = len(numeric_columns)
            characteristics['categorical_count'] = len(categorical_columns)
        
        return characteristics
    
    async def _analyze_statistical_properties(self, data) -> Dict[str, Any]:
        """통계적 속성 분석"""
        properties = {}
        
        if data is None or not hasattr(data, 'select_dtypes'):
            return properties
        
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if not numeric_data.empty:
                # 정규성 검정 (첫 번째 수치형 컬럼)
                first_numeric = numeric_data.iloc[:, 0].dropna()
                if len(first_numeric) > 3:
                    from scipy import stats
                    _, p_value = stats.shapiro(first_numeric[:5000])  # 최대 5000개 샘플
                    properties['normality_p_value'] = p_value
                
                # 기본 통계량
                properties['mean'] = numeric_data.mean().to_dict()
                properties['std'] = numeric_data.std().to_dict()
                properties['skewness'] = numeric_data.skew().to_dict()
                properties['kurtosis'] = numeric_data.kurtosis().to_dict()
        
        except Exception as e:
            self.logger.warning(f"통계적 속성 분석 오류: {e}")
        
        return properties
    
    async def _analyze_user_requirements(self, requirements) -> Dict[str, Any]:
        """사용자 요구사항 분석"""
        if isinstance(requirements, str):
            # 텍스트 분석을 통한 요구사항 추출
            analysis_type = self._extract_analysis_type_from_text(requirements)
            return {
                'analysis_type': analysis_type,
                'original_text': requirements
            }
        elif isinstance(requirements, dict):
            return requirements
        else:
            return {'analysis_type': 'unknown'}
    
    def _extract_analysis_type_from_text(self, text: str) -> str:
        """텍스트에서 분석 유형 추출"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['비교', 'compare', '차이', 'difference']):
            return 'comparison'
        elif any(word in text_lower for word in ['관계', 'relationship', '상관', 'correlation']):
            return 'relationship'
        elif any(word in text_lower for word in ['예측', 'predict', '회귀', 'regression']):
            return 'prediction'
        else:
            return 'descriptive'
    
    async def _calculate_path_confidence(self, nodes: List[DecisionNode], 
                                       context: Dict[str, Any]) -> float:
        """경로 신뢰도 계산"""
        if not nodes:
            return 0.0
        
        # 기본 신뢰도
        base_confidence = 0.7
        
        # 노드별 신뢰도 조정
        confidence_adjustments = []
        
        for node in nodes:
            if node.confidence > 0:
                confidence_adjustments.append(node.confidence)
            else:
                # 기본 조정값
                confidence_adjustments.append(0.8)
        
        # 평균 신뢰도 계산
        if confidence_adjustments:
            avg_confidence = sum(confidence_adjustments) / len(confidence_adjustments)
            final_confidence = (base_confidence + avg_confidence) / 2
        else:
            final_confidence = base_confidence
        
        return min(final_confidence, 1.0)
    
    async def _create_alternative_path(self, method: AnalysisMethod, 
                                     context: Dict[str, Any]) -> DecisionPath:
        """대안 경로 생성"""
        # 간단한 대안 경로 생성
        alt_node = DecisionNode(
            node_id=f"alt_{method.value}",
            criteria=DecisionCriteria.ANALYSIS_GOAL,
            condition="alternative",
            recommendation=method,
            reasoning=f"대안 방법: {method.value}"
        )
        
        confidence = await self._calculate_alternative_confidence(method, context)
        
        return DecisionPath(
            nodes=[alt_node],
            final_recommendation=method,
            confidence_score=confidence,
            reasoning_chain=[f"대안 방법: {method.value}"]
        )
    
    async def _calculate_alternative_confidence(self, method: AnalysisMethod,
                                              context: Dict[str, Any]) -> float:
        """대안 방법 신뢰도 계산"""
        # 기본 대안 신뢰도
        base_confidence = 0.5
        
        # 방법별 조정
        method_adjustments = {
            AnalysisMethod.DESCRIPTIVE: 0.8,
            AnalysisMethod.NONPARAMETRIC: 0.7,
            AnalysisMethod.T_TEST: 0.6,
            AnalysisMethod.ANOVA: 0.6,
            AnalysisMethod.CORRELATION: 0.6,
            AnalysisMethod.REGRESSION: 0.5,
            AnalysisMethod.CHI_SQUARE: 0.6,
            AnalysisMethod.TIME_SERIES: 0.4
        }
        
        return method_adjustments.get(method, base_confidence)
    
    async def _define_analysis_steps(self, method: AnalysisMethod, 
                                   context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """분석 단계 정의"""
        steps = []
        
        # 공통 단계
        steps.append({
            'type': 'data_exploration',
            'description': '데이터 탐색 및 기술통계',
            'parameters': {}
        })
        
        # 방법별 단계
        if method == AnalysisMethod.T_TEST:
            steps.extend([
                {
                    'type': 'assumption_check',
                    'description': '정규성 및 등분산성 검정',
                    'parameters': {'tests': ['normality', 'homoscedasticity']}
                },
                {
                    'type': 't_test',
                    'description': 't-검정 수행',
                    'parameters': {'alpha': 0.05}
                }
            ])
        elif method == AnalysisMethod.ANOVA:
            steps.extend([
                {
                    'type': 'assumption_check',
                    'description': 'ANOVA 가정 검정',
                    'parameters': {'tests': ['normality', 'homoscedasticity', 'independence']}
                },
                {
                    'type': 'anova',
                    'description': '분산분석 수행',
                    'parameters': {'alpha': 0.05}
                },
                {
                    'type': 'post_hoc',
                    'description': '사후검정',
                    'parameters': {'method': 'tukey'}
                }
            ])
        elif method == AnalysisMethod.CORRELATION:
            steps.extend([
                {
                    'type': 'correlation_analysis',
                    'description': '상관관계 분석',
                    'parameters': {'method': 'pearson'}
                }
            ])
        elif method == AnalysisMethod.REGRESSION:
            steps.extend([
                {
                    'type': 'regression_analysis',
                    'description': '회귀분석',
                    'parameters': {'method': 'linear'}
                }
            ])
        elif method == AnalysisMethod.NONPARAMETRIC:
            steps.extend([
                {
                    'type': 'nonparametric_test',
                    'description': '비모수 검정',
                    'parameters': {'method': 'mann_whitney'}
                }
            ])
        
        # 결과 해석 단계
        steps.append({
            'type': 'interpretation',
            'description': '결과 해석 및 보고서 생성',
            'parameters': {}
        })
        
        return steps
    
    async def _identify_assumptions_to_check(self, method: AnalysisMethod,
                                           context: Dict[str, Any]) -> List[str]:
        """검증할 가정 식별"""
        assumptions = []
        
        if method in [AnalysisMethod.T_TEST, AnalysisMethod.ANOVA]:
            assumptions.extend(['normality', 'homoscedasticity', 'independence'])
        elif method == AnalysisMethod.CORRELATION:
            assumptions.extend(['linearity', 'normality'])
        elif method == AnalysisMethod.REGRESSION:
            assumptions.extend(['linearity', 'independence', 'homoscedasticity', 'normality'])
        elif method == AnalysisMethod.CHI_SQUARE:
            assumptions.extend(['independence', 'expected_frequency'])
        
        return assumptions
    
    def _get_default_parameters(self, method: AnalysisMethod) -> Dict[str, Any]:
        """기본 파라미터 반환"""
        defaults = {
            AnalysisMethod.T_TEST: {
                'alpha': 0.05,
                'alternative': 'two-sided',
                'equal_var': True
            },
            AnalysisMethod.ANOVA: {
                'alpha': 0.05,
                'post_hoc': 'tukey'
            },
            AnalysisMethod.CORRELATION: {
                'method': 'pearson',
                'alpha': 0.05
            },
            AnalysisMethod.REGRESSION: {
                'alpha': 0.05,
                'fit_intercept': True
            },
            AnalysisMethod.NONPARAMETRIC: {
                'alpha': 0.05,
                'alternative': 'two-sided'
            },
            AnalysisMethod.CHI_SQUARE: {
                'alpha': 0.05,
                'correction': True
            }
        }
        
        return defaults.get(method, {'alpha': 0.05})
    
    async def _optimize_for_context(self, base_params: Dict[str, Any],
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """컨텍스트 기반 파라미터 최적화"""
        optimized = base_params.copy()
        
        # 표본 크기 기반 조정
        data_chars = context.get('data_characteristics', {})
        sample_size = data_chars.get('sample_size', 0)
        
        if sample_size < 30:
            # 소표본의 경우 더 보수적인 설정
            optimized['alpha'] = min(optimized.get('alpha', 0.05), 0.01)
        
        # 데이터 특성 기반 조정
        stats_props = context.get('statistical_properties', {})
        normality_p = stats_props.get('normality_p_value', 1.0)
        
        if normality_p < 0.05:
            # 정규성 불만족 시 비모수 방법 권장
            if 'method' in optimized:
                if optimized['method'] == 'pearson':
                    optimized['method'] = 'spearman'
        
        return optimized
    
    async def _apply_learned_optimizations(self, params: Dict[str, Any],
                                         method: AnalysisMethod,
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """학습된 최적화 적용"""
        # 성공 패턴에서 학습된 파라미터 적용
        for pattern in self.success_patterns:
            if pattern.get('method') == method:
                learned_params = pattern.get('parameters', {})
                for key, value in learned_params.items():
                    if key in params:
                        # 학습된 값과 현재 값의 가중 평균
                        current_value = params[key]
                        if isinstance(current_value, (int, float)) and isinstance(value, (int, float)):
                            params[key] = (current_value + value) / 2
        
        return params
    
    async def _define_expected_outputs(self, method: AnalysisMethod,
                                     context: Dict[str, Any]) -> List[str]:
        """예상 출력 정의"""
        outputs = ['기술통계량', '시각화']
        
        if method == AnalysisMethod.T_TEST:
            outputs.extend(['t-통계량', 'p-값', '신뢰구간', '효과크기'])
        elif method == AnalysisMethod.ANOVA:
            outputs.extend(['F-통계량', 'p-값', '효과크기', '사후검정 결과'])
        elif method == AnalysisMethod.CORRELATION:
            outputs.extend(['상관계수', 'p-값', '산점도'])
        elif method == AnalysisMethod.REGRESSION:
            outputs.extend(['회귀계수', 'R-squared', '잔차분석', '예측값'])
        elif method == AnalysisMethod.NONPARAMETRIC:
            outputs.extend(['검정통계량', 'p-값', '순위합'])
        elif method == AnalysisMethod.CHI_SQUARE:
            outputs.extend(['카이제곱 통계량', 'p-값', '기대빈도', '잔차'])
        
        return outputs
    
    # 학습 관련 메서드들
    async def _load_learned_patterns(self):
        """학습된 패턴 로드"""
        # 실제 구현에서는 파일이나 데이터베이스에서 로드
        self.learned_patterns = {}
        self.success_patterns = []
        self.failure_patterns = []
    
    async def _optimize_tree(self):
        """트리 최적화"""
        # 학습된 패턴을 바탕으로 노드 가중치 조정
        pass
    
    def _extract_pattern(self, decision_path: DecisionPath, 
                        outcome: Dict[str, Any]) -> Dict[str, Any]:
        """패턴 추출"""
        return {
            'method': decision_path.final_recommendation,
            'confidence': decision_path.confidence_score,
            'success': outcome.get('success', False),
            'reasoning': decision_path.reasoning_chain,
            'timestamp': datetime.now(),
            'parameters': outcome.get('parameters', {})
        }
    
    async def _update_success_weights(self, decision_path: DecisionPath):
        """성공 가중치 업데이트"""
        for node in decision_path.nodes:
            node.confidence = min(node.confidence + 0.1, 1.0)
    
    async def _update_failure_weights(self, decision_path: DecisionPath):
        """실패 가중치 업데이트"""
        for node in decision_path.nodes:
            node.confidence = max(node.confidence - 0.1, 0.0)
    
    async def _update_learned_patterns(self, pattern: Dict[str, Any]):
        """학습된 패턴 업데이트"""
        method = pattern.get('method')
        if method:
            if method.value not in self.learned_patterns:
                self.learned_patterns[method.value] = []
            self.learned_patterns[method.value].append(pattern)
    
    # 공개 인터페이스 메서드들
    def get_decision_history(self) -> List[DecisionPath]:
        """의사결정 이력 반환"""
        return self.decision_history
    
    def get_learned_patterns(self) -> Dict[str, Any]:
        """학습된 패턴 반환"""
        return self.learned_patterns
    
    def get_tree_structure(self) -> Dict[str, Any]:
        """트리 구조 반환"""
        return self._serialize_node(self.root_node) if self.root_node else {}
    
    def _serialize_node(self, node: DecisionNode) -> Dict[str, Any]:
        """노드 직렬화"""
        return {
            'node_id': node.node_id,
            'criteria': node.criteria.value,
            'condition': node.condition,
            'threshold': node.threshold,
            'recommendation': node.recommendation.value if node.recommendation else None,
            'confidence': node.confidence,
            'reasoning': node.reasoning,
            'children': [self._serialize_node(child) for child in node.children]
        } 