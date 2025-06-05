"""
LLMAgent: Multi-turn 대화형 통계 분석 워크플로우 오케스트레이션 및 상태 관리

Enhanced RAG 시스템 (비즈니스 지식 + DB 스키마)을 활용한 Multi-turn 대화형 통계 분석 프로세스의 
중앙 컨트롤 타워 역할을 하며, AI 추천 기반 워크플로우의 각 단계를 실행하고 세션 상태를 관리합니다.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
import json
import re
from pathlib import Path


class LLMAgent:
    """
    Enhanced RAG 기반 Multi-turn LLM Agent - 통계 검정 자동화 시스템의 핵심 클래스
    
    비즈니스 컨텍스트 인식 AI 추천 기반 워크플로우 상태 기계의 실행자 역할을 하며, 
    각 노드별 작업을 처리하고 전체 대화형 분석 세션을 관리합니다.
    """
    
    def __init__(self, workflow_manager, decision_engine, context_manager, 
                 llm_client, prompt_crafter, data_loader, 
                 business_retriever, schema_retriever, rag_manager,
                 analysis_recommender, safe_code_executor, report_generator):
        """
        Enhanced RAG 기반 Multi-turn LLMAgent 초기화
        
        Args:
            workflow_manager: Multi-turn 워크플로우 관리자
            decision_engine: 의사결정 엔진
            context_manager: 세션 컨텍스트 관리자
            llm_client: LLM 클라이언트
            prompt_crafter: 프롬프트 생성기
            data_loader: 데이터 로더
            business_retriever: 비즈니스 지식 검색기
            schema_retriever: DB 스키마 구조 검색기
            rag_manager: RAG 통합 관리자
            analysis_recommender: AI 분석 추천 엔진
            safe_code_executor: 안전 코드 실행기
            report_generator: 보고서 생성기
        """
        self.workflow_manager = workflow_manager
        self.decision_engine = decision_engine
        self.context_manager = context_manager
        self.llm_client = llm_client
        self.prompt_crafter = prompt_crafter
        self.data_loader = data_loader
        
        # Enhanced RAG 시스템 컴포넌트
        self.business_retriever = business_retriever
        self.schema_retriever = schema_retriever
        self.rag_manager = rag_manager
        self.analysis_recommender = analysis_recommender
        
        self.safe_code_executor = safe_code_executor
        self.report_generator = report_generator
        
        # Multi-turn 세션 상태 관리
        self.current_node_id = None
        self.session_active = False
        self.session_data = {}
        self.raw_data = None
        self.processed_data = None
        self.user_interaction_history = []
        
        # Enhanced RAG 및 AI 추천 관련 상태
        self.business_context = {}
        self.schema_context = {}
        self.ai_recommendations = []
        self.selected_recommendation = None
        self.natural_language_request = ""
        
        # 대화형 모드 관련 상태
        self.interactive_mode = True  # Multi-turn은 항상 대화형
        self.pending_user_confirmation = None
        self.workflow_paused = False
        self.conversation_context = []
        
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        
    def start_session(self, data_path: Optional[str] = None) -> Dict:
        """
        Enhanced RAG 기반 Multi-turn 분석 세션 시작
        
        Args:
            data_path: 분석할 데이터 파일 경로 (선택사항)
            
        Returns:
            Dict: 세션 시작 결과
        """
        self.logger.info("Enhanced RAG 기반 Multi-turn 분석 세션 시작")
        
        # 세션 초기화
        self.session_active = True
        self.current_node_id = self.workflow_manager.get_initial_node_id()
        self.session_data = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'start_time': datetime.now(),
            'data_path': data_path
        }
        
        # RAG 시스템 초기화
        self.business_context = {}
        self.schema_context = {}
        
        # 컨텍스트 초기화
        self.context_manager.add_interaction(
            role="system",
            content="Enhanced RAG 기반 Multi-turn 통계 분석 세션이 시작되었습니다.",
            node_id=self.current_node_id
        )
        
        # 데이터가 제공된 경우 로딩
        if data_path:
            self._load_session_data(data_path)
        
        self.logger.info(f"세션 시작됨 - ID: {self.session_data['session_id']}")
        return {
            'session_started': True,
            'session_id': self.session_data['session_id'],
            'current_node': self.current_node_id,
            'message': "Enhanced RAG 기반 Multi-turn 분석 세션이 시작되었습니다.",
            'next_action': self._get_next_action_description()
        }
    
    def process_user_input(self, user_input: str) -> Dict:
        """
        사용자 입력 처리 및 Enhanced RAG 기반 워크플로우 진행
        
        Args:
            user_input: 사용자 입력 텍스트
            
        Returns:
            Dict: 처리 결과
        """
        if not self.session_active:
            return {'error': '활성 세션이 없습니다. 먼저 세션을 시작해주세요.'}
        
        self.logger.info(f"사용자 입력 처리: {user_input}")
        
        # 사용자 입력을 컨텍스트에 추가
        self.context_manager.add_interaction(
            role="user",
            content=user_input,
            node_id=self.current_node_id
        )
        
        # 현재 노드에 따른 처리 - 새로운 워크플로우 대응
        try:
            if self.current_node_id == 'data_selection':
                return self._handle_data_selection(user_input)
            elif self.current_node_id == 'natural_language_request':
                return self._handle_natural_language_request(user_input)
            elif self.current_node_id == 'rag_system_activation':
                return self._handle_rag_system_activation()
            elif self.current_node_id == 'ai_recommendation_generation':
                return self._handle_ai_recommendation_generation()
            elif self.current_node_id == 'recommendation_display':
                return self._handle_recommendation_display()
            elif self.current_node_id == 'method_confirmation':
                return self._handle_method_confirmation(user_input)
            elif self.current_node_id == 'session_continuation':
                return self._handle_session_continuation(user_input)
            else:
                return self._handle_general_node_processing(user_input)
                
        except Exception as e:
            self.logger.error(f"사용자 입력 처리 중 오류: {e}")
            return {
                'error': f'처리 중 오류가 발생했습니다: {str(e)}',
                'requires_input': True,
                'question': "다시 시도하시겠습니까? (y/n)"
            }
    
    def _handle_data_selection(self, user_input: str) -> Dict:
        """데이터 선택 노드 처리"""
        self.logger.info("데이터 선택 처리 시작")
        
        # 입력이 숫자인 경우 (데이터 파일 선택)
        if user_input.strip().isdigit():
            return self._handle_data_file_selection(int(user_input.strip()))
        
        # 파일 경로로 간주하고 데이터 로딩 시도
        try:
            self._load_session_data(user_input.strip())
            return self._transition_to_data_overview()
        except Exception as e:
            return {
                'error': f'데이터 로딩 실패: {str(e)}',
                'requires_input': True,
                'question': "올바른 데이터 파일 경로를 입력해주세요:"
            }
    
    def _handle_natural_language_request(self, user_input: str) -> Dict:
        """자연어 분석 요청 처리"""
        self.logger.info("자연어 분석 요청 처리 시작")
        
        self.natural_language_request = user_input.strip()
        
        # 자연어 요청 기본 검증
        if len(self.natural_language_request) < 10:
            return {
                'error': '분석 요청이 너무 짧습니다. 더 구체적으로 설명해주세요.',
                'requires_input': True,
                'question': "분석하고 싶은 내용을 자세히 설명해주세요:"
            }
        
        # RAG 시스템 활성화로 전환
        return self._transition_to_rag_activation()
    
    def _handle_rag_system_activation(self) -> Dict:
        """RAG 시스템 활성화 처리"""
        self.logger.info("Enhanced RAG 시스템 활성화 시작")
        
        try:
            # 비즈니스 지식 검색
            self.business_context = self.rag_manager.search_business_knowledge(
                self.natural_language_request
            )
            
            # 데이터 컨텍스트 준비
            data_context = self._prepare_data_context()
            
            # DB 스키마 구조 검색
            self.schema_context = self.rag_manager.search_schema_context(
                data_context
            )
            
            self.logger.info("RAG 시스템 활성화 완료")
            
            # AI 추천 생성으로 전환
            return self._transition_to_ai_recommendation()
            
        except Exception as e:
            self.logger.error(f"RAG 시스템 활성화 오류: {e}")
            return {
                'error': f'RAG 시스템 오류: {str(e)}',
                'requires_input': True,
                'question': "계속하시겠습니까? (y/n)"
            }
    
    def _handle_ai_recommendation_generation(self) -> Dict:
        """AI 기반 분석 방법 추천 생성"""
        self.logger.info("AI 분석 방법 추천 생성 시작")
        
        try:
            # 데이터 요약 준비
            data_summary = self._get_comprehensive_data_summary()
            
            # AI 추천 생성
            self.ai_recommendations = self.analysis_recommender.generate_recommendations(
                natural_language_request=self.natural_language_request,
                data_summary=data_summary,
                business_context=self.business_context,
                schema_context=self.schema_context
            )
            
            self.logger.info(f"AI 추천 {len(self.ai_recommendations)}개 생성 완료")
            
            # 추천 표시로 전환
            return self._transition_to_recommendation_display()
            
        except Exception as e:
            self.logger.error(f"AI 추천 생성 오류: {e}")
            return {
                'error': f'AI 추천 생성 실패: {str(e)}',
                'fallback_recommendations': self._get_fallback_recommendations()
            }
    
    def _handle_recommendation_display(self) -> Dict:
        """추천 방법 표시 및 사용자 선택 처리"""
        if not self.ai_recommendations:
            return {
                'error': '추천 방법이 없습니다.',
                'requires_input': True,
                'question': "분석 요청을 다시 입력해주세요:"
            }
        
        # 추천 결과 포맷팅
        formatted_recommendations = self._format_recommendations_for_display()
        
        return {
            'recommendations': formatted_recommendations,
            'requires_input': True,
            'question': "선택하실 방법 번호를 입력해주세요 (1-3):",
            'current_node': 'recommendation_display'
        }
    
    def _handle_method_confirmation(self, user_input: str) -> Dict:
        """선택된 분석 방법 확정"""
        try:
            selection = int(user_input.strip())
            if 1 <= selection <= len(self.ai_recommendations):
                self.selected_recommendation = self.ai_recommendations[selection - 1]
                
                # 분석 실행으로 전환
                return self._transition_to_analysis_execution()
            else:
                return {
                    'error': f'1-{len(self.ai_recommendations)} 범위의 숫자를 입력해주세요.',
                    'requires_input': True,
                    'question': "다시 선택해주세요:"
                }
        except ValueError:
            return {
                'error': '숫자를 입력해주세요.',
                'requires_input': True,
                'question': "방법 번호를 입력해주세요:"
            }
    
    def _handle_session_continuation(self, user_input: str) -> Dict:
        """세션 지속 또는 종료 처리"""
        user_input_lower = user_input.lower().strip()
        
        if user_input_lower in ['y', 'yes', '예', '네', '계속']:
            # 새로운 분석 요청으로 돌아가기
            return self._transition_to_natural_language_request()
        elif user_input_lower in ['n', 'no', '아니오', '종료', 'exit']:
            return self._end_session()
        elif user_input_lower in ['다른', 'other', '새로운', 'new']:
            # 다른 데이터로 분석
            return self._transition_to_data_selection()
        else:
            return {
                'error': 'y(계속)/n(종료)/other(다른 데이터) 중 하나를 선택해주세요.',
                'requires_input': True,
                'question': "추가 분석을 하시겠습니까? (y/n/other):"
            }

    def _prepare_data_context(self) -> Dict:
        """RAG 시스템을 위한 데이터 컨텍스트 준비"""
        if self.raw_data is None:
            return {}
        
        return {
            'columns': list(self.raw_data.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.raw_data.dtypes.items()},
            'shape': self.raw_data.shape,
            'sample_data': self.raw_data.head().to_dict()
        }
    
    def _format_recommendations_for_display(self) -> List[Dict]:
        """AI 추천을 사용자 표시용으로 포맷팅"""
        formatted = []
        for i, rec in enumerate(self.ai_recommendations, 1):
            formatted.append({
                'number': i,
                'method': rec.method_name,
                'confidence': rec.confidence_score,
                'reasoning': rec.reasoning,
                'business_context': rec.business_interpretation,
                'schema_considerations': rec.schema_considerations
            })
        return formatted
    
    def _transition_to_data_overview(self) -> Dict:
        """데이터 개요로 전환"""
        self.current_node_id = 'data_overview'
        return {
            'node_transition': 'data_overview',
            'message': '데이터 로딩이 완료되었습니다.',
            'data_info': self._get_data_summary(),
            'auto_proceed': True
        }
    
    def _transition_to_rag_activation(self) -> Dict:
        """RAG 시스템 활성화로 전환"""
        self.current_node_id = 'rag_system_activation'
        return {
            'node_transition': 'rag_system_activation',
            'message': 'Enhanced RAG 시스템을 활성화합니다...',
            'auto_proceed': True
        }
    
    def _transition_to_ai_recommendation(self) -> Dict:
        """AI 추천 생성으로 전환"""
        self.current_node_id = 'ai_recommendation_generation'
        return {
            'node_transition': 'ai_recommendation_generation',
            'message': 'AI가 최적의 분석 방법을 추천하고 있습니다...',
            'auto_proceed': True
        }
    
    def _transition_to_recommendation_display(self) -> Dict:
        """추천 표시로 전환"""
        self.current_node_id = 'recommendation_display'
        return self._handle_recommendation_display()
    
    def _transition_to_analysis_execution(self) -> Dict:
        """분석 실행으로 전환"""
        self.current_node_id = 'automated_preprocessing'
        return {
            'node_transition': 'automated_preprocessing',
            'message': f'선택된 방법({self.selected_recommendation.method_name})으로 분석을 시작합니다...',
            'selected_method': self.selected_recommendation.method_name,
            'auto_proceed': True
        }
    
    def _transition_to_natural_language_request(self) -> Dict:
        """자연어 요청으로 전환"""
        self.current_node_id = 'natural_language_request'
        return {
            'node_transition': 'natural_language_request',
            'message': '새로운 분석 요청을 입력해주세요.',
            'requires_input': True,
            'question': "분석하고 싶은 내용을 자연어로 설명해주세요:"
        }
    
    def _transition_to_data_selection(self) -> Dict:
        """데이터 선택으로 전환"""
        self.current_node_id = 'data_selection'
        return {
            'node_transition': 'data_selection',
            'message': '새로운 데이터를 선택해주세요.',
            'requires_input': True,
            'question': "분석할 데이터 파일을 선택해주세요:"
        }
    
    def _get_next_action_description(self) -> str:
        """현재 노드에서 다음 필요한 액션 설명"""
        if self.current_node_id == 'start':
            return "데이터 선택이 필요합니다."
        elif self.current_node_id == 'data_selection':
            return "분석할 데이터 파일을 선택해주세요."
        elif self.current_node_id == 'natural_language_request':
            return "분석 요청을 자연어로 입력해주세요."
        else:
            return f"현재 단계: {self.current_node_id}"
    
    def _get_fallback_recommendations(self) -> List[Dict]:
        """AI 추천 실패 시 기본 추천"""
        return [
            {
                'method': '기술통계 분석',
                'confidence': 0.8,
                'reasoning': '데이터의 기본적인 분포와 특성을 파악합니다.'
            },
            {
                'method': '상관관계 분석',
                'confidence': 0.7,
                'reasoning': '변수 간의 관계를 탐색합니다.'
            }
        ]

    # ... existing code ...

    def _get_data_summary(self) -> Dict:
        """현재 데이터 요약 정보 반환"""
        if self.raw_data is None:
            return {}
        
        return {
            'shape': self.raw_data.shape,
            'columns': list(self.raw_data.columns),
            'dtypes': self.raw_data.dtypes.to_dict(),
            'missing_values': self.raw_data.isnull().sum().to_dict()
        }
    
    def _format_recommendations_for_display(self) -> List[Dict]:
        """AI 추천을 사용자 표시용으로 포맷팅"""
        formatted = []
        for i, rec in enumerate(self.ai_recommendations, 1):
            formatted.append({
                'number': i,
                'method': rec.method_name,
                'confidence': rec.confidence_score,
                'reasoning': rec.reasoning,
                'business_context': rec.business_interpretation,
                'schema_considerations': rec.schema_considerations
            })
        return formatted
    
    def _transition_to_data_overview(self) -> Dict:
        """데이터 개요로 전환"""
        self.current_node_id = 'data_overview'
        return {
            'node_transition': 'data_overview',
            'message': '데이터 로딩이 완료되었습니다.',
            'data_info': self._get_data_summary(),
            'auto_proceed': True
        }
    
    def _transition_to_rag_activation(self) -> Dict:
        """RAG 시스템 활성화로 전환"""
        self.current_node_id = 'rag_system_activation'
        return {
            'node_transition': 'rag_system_activation',
            'message': 'Enhanced RAG 시스템을 활성화합니다...',
            'auto_proceed': True
        }
    
    def _transition_to_ai_recommendation(self) -> Dict:
        """AI 추천 생성으로 전환"""
        self.current_node_id = 'ai_recommendation_generation'
        return {
            'node_transition': 'ai_recommendation_generation',
            'message': 'AI가 최적의 분석 방법을 추천하고 있습니다...',
            'auto_proceed': True
        }
    
    def _transition_to_recommendation_display(self) -> Dict:
        """추천 표시로 전환"""
        self.current_node_id = 'recommendation_display'
        return self._handle_recommendation_display()
    
    def _transition_to_analysis_execution(self) -> Dict:
        """분석 실행으로 전환"""
        self.current_node_id = 'automated_preprocessing'
        return {
            'node_transition': 'automated_preprocessing',
            'message': f'선택된 방법({self.selected_recommendation.method_name})으로 분석을 시작합니다...',
            'selected_method': self.selected_recommendation.method_name,
            'auto_proceed': True
        }
    
    def _transition_to_natural_language_request(self) -> Dict:
        """자연어 요청으로 전환"""
        self.current_node_id = 'natural_language_request'
        return {
            'node_transition': 'natural_language_request',
            'message': '새로운 분석 요청을 입력해주세요.',
            'requires_input': True,
            'question': "분석하고 싶은 내용을 자연어로 설명해주세요:"
        }
    
    def _transition_to_data_selection(self) -> Dict:
        """데이터 선택으로 전환"""
        self.current_node_id = 'data_selection'
        return {
            'node_transition': 'data_selection',
            'message': '새로운 데이터를 선택해주세요.',
            'requires_input': True,
            'question': "분석할 데이터 파일을 선택해주세요:"
        }
    
    def _get_next_action_description(self) -> str:
        """현재 노드에서 다음 필요한 액션 설명"""
        if self.current_node_id == 'start':
            return "데이터 선택이 필요합니다."
        elif self.current_node_id == 'data_selection':
            return "분석할 데이터 파일을 선택해주세요."
        elif self.current_node_id == 'natural_language_request':
            return "분석 요청을 자연어로 입력해주세요."
        else:
            return f"현재 단계: {self.current_node_id}"
    
    def _get_fallback_recommendations(self) -> List[Dict]:
        """AI 추천 실패 시 기본 추천"""
        return [
            {
                'method': '기술통계 분석',
                'confidence': 0.8,
                'reasoning': '데이터의 기본적인 분포와 특성을 파악합니다.'
            },
            {
                'method': '상관관계 분석',
                'confidence': 0.7,
                'reasoning': '변수 간의 관계를 탐색합니다.'
            }
        ]

    # ... existing code continues with all other methods ... 