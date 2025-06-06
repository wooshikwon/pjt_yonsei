"""
대화 이력 관리 모듈
다중 턴 대화의 상태와 이력을 추적하고 관리합니다.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import uuid
from enum import Enum

from utils.error_handler import ErrorHandler
from utils.global_cache import GlobalCache

class ConversationState(Enum):
    """대화 상태 열거형"""
    INITIALIZING = "initializing"
    DATA_UPLOAD = "data_upload"
    ANALYSIS_PLANNING = "analysis_planning"
    USER_SELECTION = "user_selection"
    EXECUTION = "execution"
    INTERPRETATION = "interpretation"
    FOLLOWUP = "followup"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class ConversationTurn:
    """대화 턴 데이터 클래스"""
    turn_id: str
    timestamp: datetime
    user_message: Optional[str]
    agent_response: Optional[str]
    pipeline_step: Optional[str]
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    metadata: Dict[str, Any]
    state: ConversationState
    error_info: Optional[Dict[str, Any]] = None

@dataclass
class ConversationSession:
    """대화 세션 데이터 클래스"""
    session_id: str
    created_at: datetime
    updated_at: datetime
    user_id: Optional[str]
    current_state: ConversationState
    turns: List[ConversationTurn]
    context_data: Dict[str, Any]
    preferences: Dict[str, Any]
    metadata: Dict[str, Any]

class ConversationHistory:
    """대화 이력 관리자"""
    
    def __init__(self, 
                 max_history_length: int = 50,
                 session_timeout_hours: int = 24,
                 persistence_enabled: bool = True):
        """
        Args:
            max_history_length: 최대 이력 길이
            session_timeout_hours: 세션 타임아웃 시간 (시간)
            persistence_enabled: 영속성 활성화 여부
        """
        self.logger = logging.getLogger(__name__)
        self.error_handler = ErrorHandler()
        self.cache = GlobalCache()
        
        self.max_history_length = max_history_length
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self.persistence_enabled = persistence_enabled
        
        # 활성 세션들
        self.active_sessions: Dict[str, ConversationSession] = {}
        
        # 세션별 대화 이력 (메모리 캐시)
        self.session_histories: Dict[str, deque] = {}
    
    def create_session(self, 
                      user_id: Optional[str] = None,
                      initial_context: Optional[Dict[str, Any]] = None) -> str:
        """
        새로운 대화 세션 생성
        
        Args:
            user_id: 사용자 ID
            initial_context: 초기 컨텍스트
            
        Returns:
            생성된 세션 ID
        """
        try:
            session_id = str(uuid.uuid4())
            current_time = datetime.now()
            
            session = ConversationSession(
                session_id=session_id,
                created_at=current_time,
                updated_at=current_time,
                user_id=user_id,
                current_state=ConversationState.INITIALIZING,
                turns=[],
                context_data=initial_context or {},
                preferences={},
                metadata={
                    'created_by': 'system',
                    'version': '1.0'
                }
            )
            
            self.active_sessions[session_id] = session
            self.session_histories[session_id] = deque(maxlen=self.max_history_length)
            
            # 영속성 저장
            if self.persistence_enabled:
                self._persist_session(session)
            
            self.logger.info(f"새 대화 세션 생성: {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"세션 생성 오류: {e}")
            return self.error_handler.handle_error(e, default_return="")
    
    def add_turn(self,
                session_id: str,
                user_message: Optional[str] = None,
                agent_response: Optional[str] = None,
                pipeline_step: Optional[str] = None,
                input_data: Optional[Dict[str, Any]] = None,
                output_data: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None,
                new_state: Optional[ConversationState] = None,
                error_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        대화 턴 추가
        
        Args:
            session_id: 세션 ID
            user_message: 사용자 메시지
            agent_response: 에이전트 응답
            pipeline_step: 파이프라인 단계
            input_data: 입력 데이터
            output_data: 출력 데이터
            metadata: 메타데이터
            new_state: 새로운 상태
            error_info: 오류 정보
            
        Returns:
            성공 여부
        """
        try:
            if session_id not in self.active_sessions:
                self.logger.error(f"존재하지 않는 세션: {session_id}")
                return False
            
            session = self.active_sessions[session_id]
            current_time = datetime.now()
            
            # 세션 타임아웃 확인
            if current_time - session.updated_at > self.session_timeout:
                self.logger.warning(f"세션 타임아웃: {session_id}")
                self._archive_session(session_id)
                return False
            
            turn_id = f"{session_id}_{len(session.turns)}"
            
            turn = ConversationTurn(
                turn_id=turn_id,
                timestamp=current_time,
                user_message=user_message,
                agent_response=agent_response,
                pipeline_step=pipeline_step,
                input_data=input_data or {},
                output_data=output_data or {},
                metadata=metadata or {},
                state=new_state or session.current_state,
                error_info=error_info
            )
            
            # 세션에 턴 추가
            session.turns.append(turn)
            session.updated_at = current_time
            
            # 상태 업데이트
            if new_state:
                session.current_state = new_state
            
            # 이력에 추가 (메모리 캐시)
            self.session_histories[session_id].append(turn)
            
            # 영속성 저장
            if self.persistence_enabled:
                self._persist_turn(session_id, turn)
            
            self.logger.debug(f"턴 추가 완료: {turn_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"턴 추가 오류: {e}")
            return self.error_handler.handle_error(e, default_return=False)
    
    def get_session_history(self, 
                           session_id: str,
                           last_n_turns: Optional[int] = None,
                           include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        세션의 대화 이력 조회
        
        Args:
            session_id: 세션 ID
            last_n_turns: 최근 N턴만 반환
            include_metadata: 메타데이터 포함 여부
            
        Returns:
            대화 이력 리스트
        """
        try:
            if session_id not in self.active_sessions:
                # 아카이브된 세션에서 조회 시도
                return self._get_archived_history(session_id, last_n_turns, include_metadata)
            
            session = self.active_sessions[session_id]
            turns = session.turns
            
            if last_n_turns:
                turns = turns[-last_n_turns:]
            
            history = []
            for turn in turns:
                turn_dict = asdict(turn)
                
                # datetime 객체를 ISO 형식 문자열로 변환
                turn_dict['timestamp'] = turn.timestamp.isoformat()
                turn_dict['state'] = turn.state.value
                
                if not include_metadata:
                    turn_dict.pop('metadata', None)
                
                history.append(turn_dict)
            
            return history
            
        except Exception as e:
            self.logger.error(f"이력 조회 오류: {e}")
            return self.error_handler.handle_error(e, default_return=[])
    
    def get_conversation_context(self, session_id: str) -> Dict[str, Any]:
        """
        대화의 현재 컨텍스트 조회
        
        Args:
            session_id: 세션 ID
            
        Returns:
            대화 컨텍스트
        """
        try:
            if session_id not in self.active_sessions:
                return {}
            
            session = self.active_sessions[session_id]
            
            # 최근 턴들에서 중요한 컨텍스트 추출
            context = {
                'session_id': session_id,
                'current_state': session.current_state.value,
                'user_id': session.user_id,
                'session_duration': (datetime.now() - session.created_at).total_seconds(),
                'turn_count': len(session.turns),
                'context_data': session.context_data,
                'user_preferences': session.preferences
            }
            
            # 최근 데이터 분석 정보
            recent_data_info = self._extract_recent_data_info(session)
            if recent_data_info:
                context['recent_data_analysis'] = recent_data_info
            
            # 최근 사용자 의도
            recent_intent = self._extract_recent_intent(session)
            if recent_intent:
                context['recent_user_intent'] = recent_intent
            
            # 진행 중인 파이프라인 단계
            current_pipeline_step = self._get_current_pipeline_step(session)
            if current_pipeline_step:
                context['current_pipeline_step'] = current_pipeline_step
            
            return context
            
        except Exception as e:
            self.logger.error(f"컨텍스트 조회 오류: {e}")
            return self.error_handler.handle_error(e, default_return={})
    
    def update_session_context(self, 
                              session_id: str, 
                              context_updates: Dict[str, Any]) -> bool:
        """
        세션 컨텍스트 업데이트
        
        Args:
            session_id: 세션 ID
            context_updates: 업데이트할 컨텍스트
            
        Returns:
            성공 여부
        """
        try:
            if session_id not in self.active_sessions:
                self.logger.error(f"존재하지 않는 세션: {session_id}")
                return False
            
            session = self.active_sessions[session_id]
            session.context_data.update(context_updates)
            session.updated_at = datetime.now()
            
            # 영속성 저장
            if self.persistence_enabled:
                self._persist_session(session)
            
            return True
            
        except Exception as e:
            self.logger.error(f"컨텍스트 업데이트 오류: {e}")
            return self.error_handler.handle_error(e, default_return=False)
    
    def update_user_preferences(self, 
                               session_id: str, 
                               preferences: Dict[str, Any]) -> bool:
        """
        사용자 선호도 업데이트
        
        Args:
            session_id: 세션 ID
            preferences: 선호도 설정
            
        Returns:
            성공 여부
        """
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            session.preferences.update(preferences)
            session.updated_at = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"선호도 업데이트 오류: {e}")
            return False
    
    def get_similar_conversations(self, 
                                 session_id: str, 
                                 similarity_threshold: float = 0.7,
                                 max_results: int = 5) -> List[Dict[str, Any]]:
        """
        유사한 대화 검색
        
        Args:
            session_id: 현재 세션 ID
            similarity_threshold: 유사도 임계값
            max_results: 최대 결과 수
            
        Returns:
            유사한 대화 리스트
        """
        try:
            if session_id not in self.active_sessions:
                return []
            
            current_session = self.active_sessions[session_id]
            current_context = self._extract_session_features(current_session)
            
            similar_conversations = []
            
            # 다른 활성 세션과 비교
            for other_session_id, other_session in self.active_sessions.items():
                if other_session_id == session_id:
                    continue
                
                other_context = self._extract_session_features(other_session)
                similarity = self._calculate_session_similarity(current_context, other_context)
                
                if similarity >= similarity_threshold:
                    similar_conversations.append({
                        'session_id': other_session_id,
                        'similarity_score': similarity,
                        'summary': self._generate_session_summary(other_session),
                        'relevant_insights': self._extract_relevant_insights(other_session)
                    })
            
            # 유사도 순으로 정렬
            similar_conversations.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return similar_conversations[:max_results]
            
        except Exception as e:
            self.logger.error(f"유사 대화 검색 오류: {e}")
            return []
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        세션 요약 정보 생성
        
        Args:
            session_id: 세션 ID
            
        Returns:
            세션 요약
        """
        try:
            if session_id not in self.active_sessions:
                return {}
            
            session = self.active_sessions[session_id]
            
            summary = {
                'session_id': session_id,
                'created_at': session.created_at.isoformat(),
                'updated_at': session.updated_at.isoformat(),
                'current_state': session.current_state.value,
                'turn_count': len(session.turns),
                'duration_minutes': (session.updated_at - session.created_at).total_seconds() / 60,
                'pipeline_progress': self._calculate_pipeline_progress(session),
                'key_topics': self._extract_key_topics(session),
                'data_analyzed': self._get_analyzed_data_summary(session),
                'user_preferences': session.preferences
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"세션 요약 생성 오류: {e}")
            return {}
    
    def cleanup_expired_sessions(self) -> int:
        """
        만료된 세션 정리
        
        Returns:
            정리된 세션 수
        """
        try:
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, session in self.active_sessions.items():
                if current_time - session.updated_at > self.session_timeout:
                    expired_sessions.append(session_id)
            
            cleaned_count = 0
            for session_id in expired_sessions:
                self._archive_session(session_id)
                cleaned_count += 1
            
            self.logger.info(f"만료된 세션 {cleaned_count}개 정리 완료")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"세션 정리 오류: {e}")
            return 0
    
    def _extract_recent_data_info(self, session: ConversationSession) -> Optional[Dict[str, Any]]:
        """최근 데이터 분석 정보 추출"""
        for turn in reversed(session.turns):
            if 'data_analysis' in turn.output_data:
                return turn.output_data['data_analysis']
            if 'data_overview' in turn.output_data:
                return turn.output_data['data_overview']
        return None
    
    def _extract_recent_intent(self, session: ConversationSession) -> Optional[str]:
        """최근 사용자 의도 추출"""
        for turn in reversed(session.turns):
            if turn.user_message:
                # 간단한 의도 추출 (실제로는 더 정교한 NLP 기법 사용 가능)
                message_lower = turn.user_message.lower()
                if any(word in message_lower for word in ['분석', 'analyze', 'test']):
                    return 'analysis_request'
                elif any(word in message_lower for word in ['비교', 'compare', 'difference']):
                    return 'comparison_request'
                elif any(word in message_lower for word in ['예측', 'predict', 'forecast']):
                    return 'prediction_request'
        return None
    
    def _get_current_pipeline_step(self, session: ConversationSession) -> Optional[str]:
        """현재 파이프라인 단계 조회"""
        if session.turns:
            last_turn = session.turns[-1]
            return last_turn.pipeline_step
        return None
    
    def _extract_session_features(self, session: ConversationSession) -> Dict[str, Any]:
        """세션 특성 추출"""
        features = {
            'state': session.current_state.value,
            'turn_count': len(session.turns),
            'has_data_upload': False,
            'analysis_types': set(),
            'data_types': set(),
            'user_topics': set()
        }
        
        for turn in session.turns:
            # 데이터 업로드 확인
            if 'data_upload' in turn.output_data:
                features['has_data_upload'] = True
            
            # 분석 유형 추출
            if 'analysis_type' in turn.metadata:
                features['analysis_types'].add(turn.metadata['analysis_type'])
            
            # 데이터 유형 추출
            if 'data_types' in turn.output_data:
                features['data_types'].update(turn.output_data['data_types'])
            
            # 사용자 토픽 추출
            if turn.user_message:
                # 간단한 키워드 추출
                keywords = self._extract_keywords(turn.user_message)
                features['user_topics'].update(keywords)
        
        # set을 list로 변환 (JSON 직렬화를 위해)
        features['analysis_types'] = list(features['analysis_types'])
        features['data_types'] = list(features['data_types'])
        features['user_topics'] = list(features['user_topics'])
        
        return features
    
    def _calculate_session_similarity(self, 
                                    features1: Dict[str, Any], 
                                    features2: Dict[str, Any]) -> float:
        """세션 간 유사도 계산"""
        similarity_score = 0.0
        weight_sum = 0.0
        
        # 상태 유사도 (가중치: 0.2)
        if features1['state'] == features2['state']:
            similarity_score += 0.2
        weight_sum += 0.2
        
        # 분석 유형 유사도 (가중치: 0.3)
        analysis_overlap = len(set(features1['analysis_types']) & set(features2['analysis_types']))
        analysis_union = len(set(features1['analysis_types']) | set(features2['analysis_types']))
        if analysis_union > 0:
            analysis_similarity = analysis_overlap / analysis_union
            similarity_score += 0.3 * analysis_similarity
        weight_sum += 0.3
        
        # 데이터 유형 유사도 (가중치: 0.3)
        data_overlap = len(set(features1['data_types']) & set(features2['data_types']))
        data_union = len(set(features1['data_types']) | set(features2['data_types']))
        if data_union > 0:
            data_similarity = data_overlap / data_union
            similarity_score += 0.3 * data_similarity
        weight_sum += 0.3
        
        # 토픽 유사도 (가중치: 0.2)
        topic_overlap = len(set(features1['user_topics']) & set(features2['user_topics']))
        topic_union = len(set(features1['user_topics']) | set(features2['user_topics']))
        if topic_union > 0:
            topic_similarity = topic_overlap / topic_union
            similarity_score += 0.2 * topic_similarity
        weight_sum += 0.2
        
        return similarity_score / weight_sum if weight_sum > 0 else 0.0
    
    def _generate_session_summary(self, session: ConversationSession) -> str:
        """세션 요약 텍스트 생성"""
        summary_parts = []
        
        # 기본 정보
        duration = session.updated_at - session.created_at
        summary_parts.append(f"Duration: {duration.total_seconds()/60:.1f}분")
        summary_parts.append(f"Turns: {len(session.turns)}")
        summary_parts.append(f"State: {session.current_state.value}")
        
        # 주요 활동
        activities = []
        for turn in session.turns:
            if turn.pipeline_step:
                activities.append(turn.pipeline_step)
        
        if activities:
            unique_activities = list(set(activities))
            summary_parts.append(f"Activities: {', '.join(unique_activities)}")
        
        return " | ".join(summary_parts)
    
    def _extract_relevant_insights(self, session: ConversationSession) -> List[str]:
        """관련 인사이트 추출"""
        insights = []
        
        for turn in session.turns:
            # 분석 결과에서 인사이트 추출
            if 'analysis_insights' in turn.output_data:
                insights.extend(turn.output_data['analysis_insights'])
            
            # 추천사항 추출
            if 'recommendations' in turn.output_data:
                insights.extend(turn.output_data['recommendations'])
        
        return insights[:5]  # 최대 5개만 반환
    
    def _calculate_pipeline_progress(self, session: ConversationSession) -> float:
        """파이프라인 진행률 계산"""
        total_steps = 8  # 전체 파이프라인 단계 수
        completed_steps = set()
        
        for turn in session.turns:
            if turn.pipeline_step:
                completed_steps.add(turn.pipeline_step)
        
        return len(completed_steps) / total_steps
    
    def _extract_key_topics(self, session: ConversationSession) -> List[str]:
        """주요 토픽 추출"""
        all_topics = []
        
        for turn in session.turns:
            if turn.user_message:
                keywords = self._extract_keywords(turn.user_message)
                all_topics.extend(keywords)
        
        # 빈도수 계산하여 상위 토픽 반환
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics[:5]]
    
    def _get_analyzed_data_summary(self, session: ConversationSession) -> Dict[str, Any]:
        """분석된 데이터 요약"""
        data_summary = {}
        
        for turn in session.turns:
            if 'data_overview' in turn.output_data:
                data_overview = turn.output_data['data_overview']
                data_summary.update({
                    'dataset_name': data_overview.get('dataset_name', 'Unknown'),
                    'rows': data_overview.get('shape', [0, 0])[0],
                    'columns': data_overview.get('shape', [0, 0])[1],
                    'data_types': list(data_overview.get('column_types', {}).values())
                })
                break
        
        return data_summary
    
    def _extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 키워드 추출 (간단한 버전)"""
        # 실제로는 더 정교한 NLP 기법 사용 가능
        words = text.lower().split()
        
        # 불용어 제거
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'by', '은', '는', '이', '가', '을', '를', '에', '의', '와', '과'}
        
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        return keywords[:10]  # 최대 10개 키워드
    
    def _archive_session(self, session_id: str) -> bool:
        """세션 아카이브"""
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                
                # 영속성 저장
                if self.persistence_enabled:
                    self._persist_session(session, archived=True)
                
                # 메모리에서 제거
                del self.active_sessions[session_id]
                if session_id in self.session_histories:
                    del self.session_histories[session_id]
                
                self.logger.info(f"세션 아카이브 완료: {session_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"세션 아카이브 오류: {e}")
            return False
    
    def _persist_session(self, session: ConversationSession, archived: bool = False) -> bool:
        """세션 영속성 저장"""
        try:
            # 실제 구현에서는 데이터베이스나 파일 시스템에 저장
            cache_key = f"session:{'archived' if archived else 'active'}:{session.session_id}"
            session_data = asdict(session)
            
            # datetime 객체를 ISO 문자열로 변환
            session_data['created_at'] = session.created_at.isoformat()
            session_data['updated_at'] = session.updated_at.isoformat()
            session_data['current_state'] = session.current_state.value
            
            # 턴 데이터도 직렬화
            for turn_data in session_data['turns']:
                turn_data['timestamp'] = turn_data['timestamp'].isoformat() if isinstance(turn_data['timestamp'], datetime) else turn_data['timestamp']
                turn_data['state'] = turn_data['state'].value if isinstance(turn_data['state'], ConversationState) else turn_data['state']
            
            self.cache.set(cache_key, session_data, ttl=86400 * 7)  # 7일 보관
            return True
            
        except Exception as e:
            self.logger.error(f"세션 저장 오류: {e}")
            return False
    
    def _persist_turn(self, session_id: str, turn: ConversationTurn) -> bool:
        """턴 영속성 저장"""
        try:
            cache_key = f"turn:{session_id}:{turn.turn_id}"
            turn_data = asdict(turn)
            turn_data['timestamp'] = turn.timestamp.isoformat()
            turn_data['state'] = turn.state.value
            
            self.cache.set(cache_key, turn_data, ttl=86400 * 7)
            return True
            
        except Exception as e:
            self.logger.error(f"턴 저장 오류: {e}")
            return False
    
    def _get_archived_history(self, 
                             session_id: str, 
                             last_n_turns: Optional[int] = None,
                             include_metadata: bool = True) -> List[Dict[str, Any]]:
        """아카이브된 세션 이력 조회"""
        try:
            cache_key = f"session:archived:{session_id}"
            session_data = self.cache.get(cache_key)
            
            if not session_data:
                return []
            
            turns = session_data.get('turns', [])
            
            if last_n_turns:
                turns = turns[-last_n_turns:]
            
            if not include_metadata:
                for turn in turns:
                    turn.pop('metadata', None)
            
            return turns
            
        except Exception as e:
            self.logger.error(f"아카이브 이력 조회 오류: {e}")
            return []
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """대화 이력 통계 정보"""
        try:
            active_count = len(self.active_sessions)
            total_turns = sum(len(session.turns) for session in self.active_sessions.values())
            
            # 상태별 세션 수
            state_counts = {}
            for session in self.active_sessions.values():
                state = session.current_state.value
                state_counts[state] = state_counts.get(state, 0) + 1
            
            # 평균 세션 지속 시간
            if active_count > 0:
                avg_duration = sum(
                    (session.updated_at - session.created_at).total_seconds()
                    for session in self.active_sessions.values()
                ) / active_count / 60  # 분 단위
            else:
                avg_duration = 0
            
            return {
                'active_sessions': active_count,
                'total_turns': total_turns,
                'average_turns_per_session': total_turns / active_count if active_count > 0 else 0,
                'average_session_duration_minutes': avg_duration,
                'session_states': state_counts,
                'max_history_length': self.max_history_length,
                'session_timeout_hours': self.session_timeout.total_seconds() / 3600
            }
            
        except Exception as e:
            self.logger.error(f"통계 생성 오류: {e}")
            return {} 