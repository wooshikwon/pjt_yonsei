"""
State Manager

대화 상태, 분석 진행 상태, 세션 데이터 등을 관리하는 상태 관리자
"""

import json
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import logging
import threading

from utils.global_cache import get_global_cache
from utils.helpers import generate_unique_id, safe_json_dumps, safe_json_loads

logger = logging.getLogger(__name__)

class StateManager:
    """
    세션 상태 및 대화 이력을 관리하는 클래스
    """
    
    def __init__(self, session_dir: Optional[str] = None):
        """
        상태 관리자 초기화
        
        Args:
            session_dir: 세션 저장 디렉토리 (None이면 메모리만 사용)
        """
        self.sessions = {}  # 활성 세션들
        self.session_metadata = {}  # 세션 메타데이터
        self.lock = threading.RLock()
        self.cache = get_global_cache()
        
        # 세션 저장 디렉토리 설정
        if session_dir:
            self.session_dir = Path(session_dir)
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self.persistent = True
        else:
            self.session_dir = None
            self.persistent = False
        
        logger.info(f"상태 관리자 초기화 - 영속성: {self.persistent}")
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        새 세션 생성
        
        Args:
            session_id: 세션 ID (None이면 자동 생성)
            
        Returns:
            str: 생성된 세션 ID
        """
        with self.lock:
            if not session_id:
                session_id = generate_unique_id("session")
            
            if session_id in self.sessions:
                logger.warning(f"세션 {session_id}가 이미 존재합니다. 기존 세션을 반환합니다.")
                return session_id
            
            # 세션 초기화
            self.sessions[session_id] = {
                'session_id': session_id,
                'created_at': datetime.now(),
                'last_updated': datetime.now(),
                'current_stage': 1,
                'context': {},
                'stage_results': {},
                'conversation_history': [],
                'analysis_state': {
                    'data_file': None,
                    'user_request': None,
                    'selected_analysis': None,
                    'current_step': None
                },
                'status': 'active'
            }
            
            # 메타데이터 저장
            self.session_metadata[session_id] = {
                'created_at': datetime.now(),
                'last_accessed': datetime.now(),
                'total_interactions': 0,
                'completed_stages': [],
                'persistent': self.persistent
            }
            
            # 영속 저장
            if self.persistent:
                self._save_session(session_id)
            
            logger.info(f"새 세션 생성: {session_id}")
            return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        세션 조회
        
        Args:
            session_id: 세션 ID
            
        Returns:
            Optional[Dict[str, Any]]: 세션 데이터 (없으면 None)
        """
        with self.lock:
            # 메모리에서 먼저 확인
            if session_id in self.sessions:
                self._update_last_accessed(session_id)
                return self.sessions[session_id].copy()
            
            # 영속 저장소에서 로드 시도
            if self.persistent:
                loaded_session = self._load_session(session_id)
                if loaded_session:
                    self.sessions[session_id] = loaded_session
                    self._update_last_accessed(session_id)
                    return loaded_session.copy()
            
            return None
    
    def update_context(self, session_id: str, context: Dict[str, Any]) -> bool:
        """
        세션 컨텍스트 업데이트
        
        Args:
            session_id: 세션 ID
            context: 업데이트할 컨텍스트
            
        Returns:
            bool: 업데이트 성공 여부
        """
        with self.lock:
            session = self.get_session(session_id)
            if not session:
                logger.error(f"세션을 찾을 수 없습니다: {session_id}")
                return False
            
            # 컨텍스트 업데이트
            session['context'].update(context)
            session['last_updated'] = datetime.now()
            
            # 세션 저장
            self.sessions[session_id] = session
            self._update_last_accessed(session_id)
            
            if self.persistent:
                self._save_session(session_id)
            
            logger.debug(f"세션 컨텍스트 업데이트: {session_id}")
            return True
    
    def update_stage_result(self, session_id: str, stage_num: int, 
                          result: Dict[str, Any]) -> bool:
        """
        단계 결과 업데이트
        
        Args:
            session_id: 세션 ID
            stage_num: 단계 번호
            result: 단계 결과
            
        Returns:
            bool: 업데이트 성공 여부
        """
        with self.lock:
            session = self.get_session(session_id)
            if not session:
                logger.error(f"세션을 찾을 수 없습니다: {session_id}")
                return False
            
            # 단계 결과 저장
            session['stage_results'][stage_num] = {
                'result': result,
                'timestamp': datetime.now(),
                'stage_name': result.get('_meta', {}).get('step_name', f'Stage {stage_num}')
            }
            
            # 현재 단계 업데이트
            session['current_stage'] = stage_num + 1
            session['last_updated'] = datetime.now()
            
            # 완료된 단계 메타데이터 업데이트
            if stage_num not in self.session_metadata[session_id]['completed_stages']:
                self.session_metadata[session_id]['completed_stages'].append(stage_num)
            
            # 세션 저장
            self.sessions[session_id] = session
            self._update_last_accessed(session_id)
            
            if self.persistent:
                self._save_session(session_id)
            
            logger.debug(f"단계 {stage_num} 결과 업데이트: {session_id}")
            return True
    
    def add_conversation_turn(self, session_id: str, user_input: str, 
                            assistant_response: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        대화 턴 추가
        
        Args:
            session_id: 세션 ID
            user_input: 사용자 입력
            assistant_response: 어시스턴트 응답
            context: 추가 컨텍스트
            
        Returns:
            bool: 추가 성공 여부
        """
        with self.lock:
            session = self.get_session(session_id)
            if not session:
                return False
            
            turn = {
                'timestamp': datetime.now(),
                'user_input': user_input,
                'assistant_response': assistant_response,
                'context': context or {},
                'turn_id': len(session['conversation_history']) + 1
            }
            
            session['conversation_history'].append(turn)
            session['last_updated'] = datetime.now()
            
            # 상호작용 수 증가
            self.session_metadata[session_id]['total_interactions'] += 1
            
            # 세션 저장
            self.sessions[session_id] = session
            self._update_last_accessed(session_id)
            
            if self.persistent:
                self._save_session(session_id)
            
            return True
    
    def update_analysis_state(self, session_id: str, analysis_state: Dict[str, Any]) -> bool:
        """
        분석 상태 업데이트
        
        Args:
            session_id: 세션 ID
            analysis_state: 분석 상태
            
        Returns:
            bool: 업데이트 성공 여부
        """
        with self.lock:
            session = self.get_session(session_id)
            if not session:
                return False
            
            session['analysis_state'].update(analysis_state)
            session['last_updated'] = datetime.now()
            
            self.sessions[session_id] = session
            self._update_last_accessed(session_id)
            
            if self.persistent:
                self._save_session(session_id)
            
            return True
    
    def complete_session(self, session_id: str) -> bool:
        """
        세션 완료 처리
        
        Args:
            session_id: 세션 ID
            
        Returns:
            bool: 완료 처리 성공 여부
        """
        with self.lock:
            session = self.get_session(session_id)
            if not session:
                return False
            
            session['status'] = 'completed'
            session['completed_at'] = datetime.now()
            session['last_updated'] = datetime.now()
            
            self.sessions[session_id] = session
            
            if self.persistent:
                self._save_session(session_id)
            
            logger.info(f"세션 완료: {session_id}")
            return True
    
    def load_session(self, session_id: str) -> bool:
        """
        저장된 세션 로드
        
        Args:
            session_id: 세션 ID
            
        Returns:
            bool: 로드 성공 여부
        """
        if not self.persistent:
            return False
        
        with self.lock:
            loaded_session = self._load_session(session_id)
            if loaded_session:
                self.sessions[session_id] = loaded_session
                self._update_last_accessed(session_id)
                logger.info(f"세션 로드 완료: {session_id}")
                return True
            return False
    
    def list_sessions(self, include_completed: bool = False) -> List[Dict[str, Any]]:
        """
        세션 목록 조회
        
        Args:
            include_completed: 완료된 세션 포함 여부
            
        Returns:
            List[Dict[str, Any]]: 세션 목록
        """
        with self.lock:
            session_list = []
            
            for session_id, metadata in self.session_metadata.items():
                session = self.sessions.get(session_id)
                if session:
                    status = session.get('status', 'unknown')
                    if not include_completed and status == 'completed':
                        continue
                    
                    session_info = {
                        'session_id': session_id,
                        'created_at': metadata['created_at'],
                        'last_accessed': metadata['last_accessed'],
                        'status': status,
                        'current_stage': session.get('current_stage', 1),
                        'total_interactions': metadata['total_interactions'],
                        'completed_stages': metadata['completed_stages']
                    }
                    session_list.append(session_info)
            
            # 마지막 접근 시간 기준 정렬
            session_list.sort(key=lambda x: x['last_accessed'], reverse=True)
            return session_list
    
    def cleanup_expired_sessions(self, max_age_hours: int = 24) -> int:
        """
        만료된 세션 정리
        
        Args:
            max_age_hours: 최대 보관 시간 (시간)
            
        Returns:
            int: 정리된 세션 수
        """
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            expired_sessions = []
            
            for session_id, metadata in self.session_metadata.items():
                if metadata['last_accessed'] < cutoff_time:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                self._remove_session(session_id)
            
            logger.info(f"{len(expired_sessions)}개의 만료된 세션을 정리했습니다.")
            return len(expired_sessions)
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        세션 요약 정보 조회
        
        Args:
            session_id: 세션 ID
            
        Returns:
            Optional[Dict[str, Any]]: 세션 요약 (없으면 None)
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        metadata = self.session_metadata.get(session_id, {})
        
        return {
            'session_id': session_id,
            'created_at': session['created_at'],
            'last_updated': session['last_updated'],
            'status': session.get('status', 'active'),
            'current_stage': session.get('current_stage', 1),
            'total_stages': 8,
            'completed_stages': metadata.get('completed_stages', []),
            'total_interactions': metadata.get('total_interactions', 0),
            'conversation_turns': len(session.get('conversation_history', [])),
            'has_analysis_results': bool(session.get('analysis_state', {}).get('selected_analysis')),
            'data_file': session.get('analysis_state', {}).get('data_file')
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        상태 관리자 전체 상태 조회
        
        Returns:
            Dict[str, Any]: 상태 정보
        """
        with self.lock:
            active_sessions = len([s for s in self.sessions.values() if s.get('status') != 'completed'])
            completed_sessions = len([s for s in self.sessions.values() if s.get('status') == 'completed'])
            
            return {
                'total_sessions': len(self.sessions),
                'active_sessions': active_sessions,
                'completed_sessions': completed_sessions,
                'persistent_storage': self.persistent,
                'session_dir': str(self.session_dir) if self.session_dir else None
            }
    
    def _save_session(self, session_id: str):
        """세션을 파일에 저장"""
        if not self.persistent:
            return
        
        try:
            session_file = self.session_dir / f"{session_id}.json"
            session_data = {
                'session': self.sessions[session_id],
                'metadata': self.session_metadata.get(session_id, {})
            }
            
            # datetime 객체를 JSON 직렬화 가능한 형태로 변환
            session_json = self._serialize_session(session_data)
            
            with open(session_file, 'w', encoding='utf-8') as f:
                f.write(safe_json_dumps(session_json))
            
        except Exception as e:
            logger.error(f"세션 저장 실패 ({session_id}): {str(e)}")
    
    def _load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """파일에서 세션 로드"""
        if not self.persistent:
            return None
        
        try:
            session_file = self.session_dir / f"{session_id}.json"
            if not session_file.exists():
                return None
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = safe_json_loads(f.read())
            
            if session_data:
                # JSON에서 datetime 객체로 복원
                deserialized = self._deserialize_session(session_data)
                
                # 메타데이터도 함께 복원
                if 'metadata' in deserialized:
                    self.session_metadata[session_id] = deserialized['metadata']
                
                return deserialized.get('session')
            
        except Exception as e:
            logger.error(f"세션 로드 실패 ({session_id}): {str(e)}")
        
        return None
    
    def _remove_session(self, session_id: str):
        """세션 제거"""
        # 메모리에서 제거
        self.sessions.pop(session_id, None)
        self.session_metadata.pop(session_id, None)
        
        # 파일에서 제거
        if self.persistent:
            try:
                session_file = self.session_dir / f"{session_id}.json"
                if session_file.exists():
                    session_file.unlink()
            except Exception as e:
                logger.error(f"세션 파일 삭제 실패 ({session_id}): {str(e)}")
    
    def _update_last_accessed(self, session_id: str):
        """마지막 접근 시간 업데이트"""
        if session_id in self.session_metadata:
            self.session_metadata[session_id]['last_accessed'] = datetime.now()
    
    def _serialize_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """세션 데이터를 JSON 직렬화 가능한 형태로 변환"""
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return {'__datetime__': obj.isoformat()}
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            else:
                return obj
        
        return convert_datetime(session_data)
    
    def _deserialize_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """JSON에서 세션 데이터로 복원"""
        def convert_datetime(obj):
            if isinstance(obj, dict):
                if '__datetime__' in obj:
                    return datetime.fromisoformat(obj['__datetime__'])
                else:
                    return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            else:
                return obj
        
        return convert_datetime(session_data) 