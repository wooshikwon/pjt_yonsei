"""
RAG Cache Manager

RAG 시스템의 검색 결과를 캐싱하여 성능을 향상시키는 모듈
- 검색 결과 캐싱
- 캐시 만료 관리
- 메모리 사용량 제어
"""

import hashlib
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import OrderedDict
import threading

from utils.error_handler import ErrorHandler, RAGException
from utils.helpers import safe_json_dumps, safe_json_loads

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """캐시 엔트리"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: int
    size_bytes: int

class RAGCacheManager:
    """RAG 검색 결과 캐시 관리자"""
    
    def __init__(self, 
                 max_size_mb: int = 100,
                 default_ttl_seconds: int = 3600,
                 cleanup_interval_seconds: int = 300):
        """
        RAG 캐시 매니저 초기화
        
        Args:
            max_size_mb: 최대 캐시 크기 (MB)
            default_ttl_seconds: 기본 TTL (초)
            cleanup_interval_seconds: 정리 작업 간격 (초)
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl_seconds = default_ttl_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        
        # 캐시 저장소 (LRU 순서 유지)
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size_bytes = 0
        
        # 스레드 안전성을 위한 락
        self.lock = threading.RLock()
        
        # 통계
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'cleanup_runs': 0,
            'total_queries': 0
        }
        
        # 마지막 정리 시간
        self.last_cleanup = datetime.now()
        
        # 오류 처리
        self.error_handler = ErrorHandler()
        
        logger.info(f"RAG 캐시 매니저 초기화 완료 - 최대 크기: {max_size_mb}MB")
    
    def get(self, 
            query: str, 
            context: Optional[Dict[str, Any]] = None,
            k: int = 5) -> Optional[List[Dict[str, Any]]]:
        """
        캐시에서 검색 결과 조회
        
        Args:
            query: 검색 쿼리
            context: 컨텍스트 정보
            k: 결과 개수
            
        Returns:
            Optional[List[Dict[str, Any]]]: 캐시된 검색 결과 또는 None
        """
        try:
            with self.lock:
                self.stats['total_queries'] += 1
                
                # 캐시 키 생성
                cache_key = self._generate_cache_key(query, context, k)
                
                # 정리 작업 확인
                self._check_cleanup()
                
                # 캐시 조회
                if cache_key in self.cache:
                    entry = self.cache[cache_key]
                    
                    # TTL 확인
                    if self._is_expired(entry):
                        del self.cache[cache_key]
                        self.current_size_bytes -= entry.size_bytes
                        self.stats['misses'] += 1
                        return None
                    
                    # 액세스 정보 업데이트
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1
                    
                    # LRU 순서 업데이트 (맨 끝으로 이동)
                    self.cache.move_to_end(cache_key)
                    
                    self.stats['hits'] += 1
                    logger.debug(f"캐시 히트: {cache_key[:16]}...")
                    return entry.value
                
                self.stats['misses'] += 1
                return None
                
        except Exception as e:
            logger.error(f"캐시 조회 오류: {str(e)}")
            return None
    
    def put(self, 
            query: str, 
            results: List[Dict[str, Any]],
            context: Optional[Dict[str, Any]] = None,
            k: int = 5,
            ttl_seconds: Optional[int] = None) -> bool:
        """
        검색 결과를 캐시에 저장
        
        Args:
            query: 검색 쿼리
            results: 검색 결과
            context: 컨텍스트 정보
            k: 결과 개수
            ttl_seconds: TTL (None이면 기본값 사용)
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            with self.lock:
                # 캐시 키 생성
                cache_key = self._generate_cache_key(query, context, k)
                
                # 데이터 크기 계산
                data_size = self._calculate_size(results)
                
                # 크기가 너무 큰 경우 저장하지 않음
                if data_size > self.max_size_bytes * 0.1:  # 전체 캐시 크기의 10% 초과
                    logger.warning(f"캐시 엔트리가 너무 큼: {data_size} bytes")
                    return False
                
                # 공간 확보
                self._ensure_space(data_size)
                
                # 캐시 엔트리 생성
                now = datetime.now()
                entry = CacheEntry(
                    key=cache_key,
                    value=results,
                    created_at=now,
                    last_accessed=now,
                    access_count=0,
                    ttl_seconds=ttl_seconds or self.default_ttl_seconds,
                    size_bytes=data_size
                )
                
                # 기존 엔트리가 있으면 크기 차감
                if cache_key in self.cache:
                    old_entry = self.cache[cache_key]
                    self.current_size_bytes -= old_entry.size_bytes
                
                # 캐시에 저장
                self.cache[cache_key] = entry
                self.current_size_bytes += data_size
                
                logger.debug(f"캐시 저장: {cache_key[:16]}... ({data_size} bytes)")
                return True
                
        except Exception as e:
            logger.error(f"캐시 저장 오류: {str(e)}")
            return False
    
    def invalidate(self, 
                  query_pattern: Optional[str] = None,
                  context_pattern: Optional[Dict[str, Any]] = None) -> int:
        """
        캐시 무효화
        
        Args:
            query_pattern: 쿼리 패턴 (None이면 전체)
            context_pattern: 컨텍스트 패턴
            
        Returns:
            int: 무효화된 엔트리 수
        """
        try:
            with self.lock:
                if query_pattern is None and context_pattern is None:
                    # 전체 캐시 클리어
                    count = len(self.cache)
                    self.cache.clear()
                    self.current_size_bytes = 0
                    logger.info(f"전체 캐시 무효화: {count}개 엔트리")
                    return count
                
                # 패턴 매칭으로 선택적 무효화
                keys_to_remove = []
                for key, entry in self.cache.items():
                    should_remove = False
                    
                    if query_pattern and query_pattern in key:
                        should_remove = True
                    
                    # 컨텍스트 패턴 매칭 (단순 구현)
                    if context_pattern and not should_remove:
                        # 실제 구현에서는 더 정교한 매칭 로직 필요
                        if str(context_pattern) in key:
                            should_remove = True
                    
                    if should_remove:
                        keys_to_remove.append(key)
                
                # 선택된 키들 제거
                for key in keys_to_remove:
                    entry = self.cache[key]
                    self.current_size_bytes -= entry.size_bytes
                    del self.cache[key]
                
                logger.info(f"선택적 캐시 무효화: {len(keys_to_remove)}개 엔트리")
                return len(keys_to_remove)
                
        except Exception as e:
            logger.error(f"캐시 무효화 오류: {str(e)}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        with self.lock:
            hit_rate = 0.0
            if self.stats['total_queries'] > 0:
                hit_rate = self.stats['hits'] / self.stats['total_queries']
            
            return {
                'total_entries': len(self.cache),
                'current_size_bytes': self.current_size_bytes,
                'current_size_mb': round(self.current_size_bytes / (1024 * 1024), 2),
                'max_size_mb': self.max_size_bytes // (1024 * 1024),
                'hit_rate': round(hit_rate, 3),
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'cleanup_runs': self.stats['cleanup_runs'],
                'total_queries': self.stats['total_queries']
            }
    
    def _generate_cache_key(self, 
                           query: str, 
                           context: Optional[Dict[str, Any]] = None,
                           k: int = 5) -> str:
        """캐시 키 생성"""
        # 정규화된 데이터로 키 생성
        key_data = {
            'query': query.strip().lower(),
            'context': context or {},
            'k': k
        }
        
        # JSON 직렬화 후 해시
        key_json = safe_json_dumps(key_data)
        hash_object = hashlib.md5(key_json.encode())
        return hash_object.hexdigest()
    
    def _calculate_size(self, data: Any) -> int:
        """데이터 크기 계산 (바이트)"""
        try:
            json_str = safe_json_dumps(data)
            return len(json_str.encode('utf-8'))
        except Exception:
            # 대략적인 크기 추정
            return len(str(data)) * 2
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """엔트리 만료 여부 확인"""
        expire_time = entry.created_at + timedelta(seconds=entry.ttl_seconds)
        return datetime.now() > expire_time
    
    def _ensure_space(self, required_bytes: int):
        """필요한 공간 확보"""
        # 현재 사용량 + 필요 공간이 최대 크기를 초과하면 공간 확보
        while (self.current_size_bytes + required_bytes > self.max_size_bytes 
               and len(self.cache) > 0):
            
            # LRU 정책으로 가장 오래된 엔트리 제거
            oldest_key, oldest_entry = next(iter(self.cache.items()))
            self.current_size_bytes -= oldest_entry.size_bytes
            del self.cache[oldest_key]
            self.stats['evictions'] += 1
            
            logger.debug(f"LRU 제거: {oldest_key[:16]}...")
    
    def _check_cleanup(self):
        """정리 작업 필요 여부 확인 및 실행"""
        now = datetime.now()
        if (now - self.last_cleanup).total_seconds() >= self.cleanup_interval_seconds:
            self._cleanup_expired()
            self.last_cleanup = now
    
    def _cleanup_expired(self):
        """만료된 엔트리 정리"""
        try:
            expired_keys = []
            
            for key, entry in self.cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)
            
            # 만료된 엔트리 제거
            for key in expired_keys:
                entry = self.cache[key]
                self.current_size_bytes -= entry.size_bytes
                del self.cache[key]
            
            if expired_keys:
                logger.debug(f"만료된 엔트리 정리: {len(expired_keys)}개")
            
            self.stats['cleanup_runs'] += 1
            
        except Exception as e:
            logger.error(f"만료 엔트리 정리 오류: {str(e)}")
    
    def warm_up(self, common_queries: List[Dict[str, Any]]):
        """
        캐시 워밍업 (자주 사용되는 쿼리들을 미리 캐시에 로드)
        
        Args:
            common_queries: 일반적인 쿼리들
                [{'query': str, 'context': dict, 'k': int, 'results': list}, ...]
        """
        try:
            logger.info(f"캐시 워밍업 시작: {len(common_queries)}개 쿼리")
            
            for query_info in common_queries:
                if 'results' in query_info:
                    self.put(
                        query=query_info['query'],
                        results=query_info['results'],
                        context=query_info.get('context'),
                        k=query_info.get('k', 5),
                        ttl_seconds=query_info.get('ttl_seconds')
                    )
            
            logger.info("캐시 워밍업 완료")
            
        except Exception as e:
            logger.error(f"캐시 워밍업 오류: {str(e)}")
    
    def export_cache(self) -> Dict[str, Any]:
        """캐시 내용 내보내기 (백업/디버깅용)"""
        try:
            with self.lock:
                exported_data = {
                    'metadata': {
                        'export_time': datetime.now().isoformat(),
                        'total_entries': len(self.cache),
                        'current_size_bytes': self.current_size_bytes,
                        'stats': self.stats
                    },
                    'entries': []
                }
                
                for key, entry in self.cache.items():
                    exported_entry = {
                        'key': key,
                        'value': entry.value,
                        'created_at': entry.created_at.isoformat(),
                        'last_accessed': entry.last_accessed.isoformat(),
                        'access_count': entry.access_count,
                        'ttl_seconds': entry.ttl_seconds,
                        'size_bytes': entry.size_bytes
                    }
                    exported_data['entries'].append(exported_entry)
                
                return exported_data
                
        except Exception as e:
            logger.error(f"캐시 내보내기 오류: {str(e)}")
            return {} 