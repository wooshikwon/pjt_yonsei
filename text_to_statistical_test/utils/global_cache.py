"""
Global Cache

전역 캐싱 시스템 (메모리 기반)
- 성능 모니터링 및 최적화 기능 포함
- TTL 기반 자동 만료
- 메모리 사용량 제한
- 캐시 히트율 추적
"""

import json
import pickle
import hashlib
import time
import threading
import logging
from typing import Any, Dict, Optional, Union, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps
import psutil
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """캐시 메트릭"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    memory_usage_mb: float = 0.0
    total_keys: int = 0
    expired_keys: int = 0
    
    @property
    def hit_rate(self) -> float:
        """캐시 히트율"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """캐시 미스율"""
        return 100.0 - self.hit_rate


@dataclass
class CacheEntry:
    """캐시 엔트리"""
    value: Any
    created_at: float
    ttl_seconds: Optional[int] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """만료 여부 확인"""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    def touch(self):
        """접근 시간 업데이트"""
        self.last_accessed = time.time()
        self.access_count += 1


class GlobalCache:
    """전역 캐시 클래스 (성능 모니터링 포함)"""
    
    def __init__(self, cache_dir: Optional[str] = None, max_size_mb: int = 500, cleanup_interval: int = 300):
        """
        GlobalCache 초기화
        
        Args:
            cache_dir: 캐시 파일 저장 디렉토리 (None이면 메모리 캐시만 사용)
            max_size_mb: 최대 메모리 사용량 (MB)
            cleanup_interval: 정리 작업 간격 (초)
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
        
        # 메트릭 추적
        self._metrics = CacheMetrics()
        self._key_patterns: Dict[str, int] = {}  # 키 패턴별 사용 빈도
        
        # 성능 최적화 설정
        self._auto_optimize = True
        self._optimization_history: List[Dict[str, Any]] = []
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.persistent_cache = True
        else:
            self.cache_dir = None
            self.persistent_cache = False
            
        logger.info(f"GlobalCache 초기화 완료 (최대 크기: {max_size_mb}MB)")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """
        캐시 키 생성
        """
        # 인수들을 문자열로 변환하여 해시 생성
        key_string = json.dumps({
            'args': args,
            'kwargs': sorted(kwargs.items())
        }, sort_keys=True, default=str)
        
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        with self._lock:
            if key not in self._cache:
                self._metrics.misses += 1
                return None
            
            entry = self._cache[key]
            
            # 만료 확인
            if entry.is_expired():
                del self._cache[key]
                self._metrics.misses += 1
                self._metrics.expired_keys += 1
                return None
            
            # 접근 정보 업데이트
            entry.touch()
            self._metrics.hits += 1
            
            # 키 패턴 추적
            self._track_key_pattern(key)
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """캐시에 값 저장"""
        with self._lock:
            try:
                # TTL은 현재 구현에서는 무시 (향후 Redis 등으로 확장시 활용)
                if ttl is not None:
                    # TTL 지원을 위한 확장 가능한 구조
                    expiry_time = time.time() + ttl
                    self._cache[key] = CacheEntry(
                        value=value,
                        created_at=time.time(),
                        ttl_seconds=ttl,
                        size_bytes=self._estimate_size(value)
                    )
                else:
                    self._cache[key] = CacheEntry(
                        value=value,
                        created_at=time.time(),
                        size_bytes=self._estimate_size(value)
                    )
                
                self._metrics.sets += 1
                self._metrics.total_keys = len(self._cache)
                
                # 정기 정리 확인
                self._periodic_cleanup()
                
                return True
                
            except Exception as e:
                logger.error(f"캐시 저장 오류 (키: {key}): {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """캐시에서 키 삭제"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._metrics.deletes += 1
                self._metrics.total_keys = len(self._cache)
                return True
            return False
    
    def clear(self):
        """모든 캐시 삭제"""
        with self._lock:
            cleared_count = len(self._cache)
            self._cache.clear()
            self._metrics.total_keys = 0
            logger.info(f"캐시 전체 삭제 완료 ({cleared_count}개 키)")
    
    def clear_by_prefix(self, prefix: str):
        """접두사로 캐시 삭제"""
        with self._lock:
            keys_to_delete = [key for key in self._cache.keys() if key.startswith(prefix)]
            for key in keys_to_delete:
                del self._cache[key]
                self._metrics.deletes += 1
            
            self._metrics.total_keys = len(self._cache)
            logger.info(f"접두사 '{prefix}' 캐시 삭제 완료 ({len(keys_to_delete)}개 키)")
    
    def get_metrics(self) -> Dict[str, Any]:
        """캐시 메트릭 반환"""
        with self._lock:
            self._update_memory_metrics()
            
            return {
                'hits': self._metrics.hits,
                'misses': self._metrics.misses,
                'hit_rate': self._metrics.hit_rate,
                'miss_rate': self._metrics.miss_rate,
                'sets': self._metrics.sets,
                'deletes': self._metrics.deletes,
                'evictions': self._metrics.evictions,
                'total_keys': self._metrics.total_keys,
                'expired_keys': self._metrics.expired_keys,
                'memory_usage_mb': self._metrics.memory_usage_mb,
                'memory_limit_mb': self._max_size_bytes / (1024 * 1024),
                'memory_usage_percent': (self._metrics.memory_usage_mb / (self._max_size_bytes / (1024 * 1024))) * 100,
                'top_key_patterns': self._get_top_key_patterns(),
                'optimization_suggestions': self._get_optimization_suggestions()
            }
    
    def optimize_ttl(self) -> Dict[str, Any]:
        """TTL 최적화"""
        with self._lock:
            optimization_result = {
                'analyzed_keys': 0,
                'optimized_keys': 0,
                'recommendations': []
            }
            
            # 키별 접근 패턴 분석
            access_patterns = {}
            current_time = time.time()
            
            for key, entry in self._cache.items():
                age = current_time - entry.created_at
                access_frequency = entry.access_count / max(age / 3600, 0.1)  # 시간당 접근 빈도
                
                access_patterns[key] = {
                    'age_hours': age / 3600,
                    'access_count': entry.access_count,
                    'access_frequency': access_frequency,
                    'current_ttl': entry.ttl_seconds,
                    'last_accessed_hours_ago': (current_time - entry.last_accessed) / 3600
                }
                
                optimization_result['analyzed_keys'] += 1
            
            # TTL 최적화 권장사항 생성
            for key, pattern in access_patterns.items():
                if pattern['access_frequency'] > 1.0 and pattern['current_ttl'] and pattern['current_ttl'] < 3600:
                    # 자주 접근되는 데이터는 TTL 연장
                    optimization_result['recommendations'].append({
                        'key': key,
                        'action': 'extend_ttl',
                        'current_ttl': pattern['current_ttl'],
                        'suggested_ttl': min(pattern['current_ttl'] * 2, 7200),
                        'reason': f"높은 접근 빈도 ({pattern['access_frequency']:.1f}/시간)"
                    })
                elif pattern['access_frequency'] < 0.1 and pattern['last_accessed_hours_ago'] > 24:
                    # 거의 접근되지 않는 데이터는 TTL 단축 또는 삭제
                    optimization_result['recommendations'].append({
                        'key': key,
                        'action': 'reduce_ttl_or_delete',
                        'current_ttl': pattern['current_ttl'],
                        'suggested_ttl': 300,  # 5분
                        'reason': f"낮은 접근 빈도 ({pattern['access_frequency']:.1f}/시간), 마지막 접근: {pattern['last_accessed_hours_ago']:.1f}시간 전"
                    })
            
            # 자동 최적화 적용 (설정된 경우)
            if self._auto_optimize:
                for rec in optimization_result['recommendations']:
                    if rec['action'] == 'extend_ttl':
                        entry = self._cache.get(rec['key'])
                        if entry:
                            entry.ttl_seconds = rec['suggested_ttl']
                            optimization_result['optimized_keys'] += 1
            
            # 최적화 이력 저장
            self._optimization_history.append({
                'timestamp': datetime.now().isoformat(),
                'result': optimization_result
            })
            
            # 이력 크기 제한 (최근 10개만 유지)
            if len(self._optimization_history) > 10:
                self._optimization_history = self._optimization_history[-10:]
            
            logger.info(f"TTL 최적화 완료: {optimization_result['optimized_keys']}개 키 최적화")
            return optimization_result
    
    def get_cache_health(self) -> Dict[str, Any]:
        """캐시 건강 상태 반환"""
        metrics = self.get_metrics()
        
        health_score = 100
        issues = []
        
        # 히트율 검사
        if metrics['hit_rate'] < 50:
            health_score -= 20
            issues.append("낮은 캐시 히트율")
        
        # 메모리 사용량 검사
        if metrics['memory_usage_percent'] > 90:
            health_score -= 30
            issues.append("높은 메모리 사용량")
        elif metrics['memory_usage_percent'] > 75:
            health_score -= 15
            issues.append("메모리 사용량 주의")
        
        # 만료된 키 비율 검사
        expired_ratio = (metrics['expired_keys'] / max(metrics['total_keys'], 1)) * 100
        if expired_ratio > 20:
            health_score -= 10
            issues.append("높은 키 만료율")
        
        return {
            'health_score': max(health_score, 0),
            'status': 'healthy' if health_score >= 80 else 'warning' if health_score >= 60 else 'critical',
            'issues': issues,
            'recommendations': self._get_health_recommendations(metrics)
        }
    
    def export_metrics(self, format: str = 'json') -> str:
        """메트릭 내보내기"""
        metrics = self.get_metrics()
        health = self.get_cache_health()
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'health': health,
            'optimization_history': self._optimization_history[-5:]  # 최근 5개
        }
        
        if format.lower() == 'json':
            return json.dumps(export_data, indent=2, ensure_ascii=False)
        else:
            # 간단한 텍스트 형식
            return f"""
캐시 성능 리포트 ({export_data['timestamp']})
==========================================
히트율: {metrics['hit_rate']:.1f}%
메모리 사용량: {metrics['memory_usage_mb']:.1f}MB ({metrics['memory_usage_percent']:.1f}%)
총 키 수: {metrics['total_keys']}
건강 점수: {health['health_score']}/100 ({health['status']})
"""
    
    def _estimate_size(self, value: Any) -> int:
        """객체 크기 추정"""
        try:
            import sys
            return sys.getsizeof(value)
        except:
            # 기본 추정값
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items())
            else:
                return 1024  # 기본값 1KB
    
    def _get_current_size(self) -> int:
        """현재 캐시 크기 계산"""
        return sum(entry.size_bytes for entry in self._cache.values())
    
    def _make_space(self, needed_bytes: int) -> bool:
        """공간 확보 (LRU 기반 제거)"""
        if needed_bytes > self._max_size_bytes:
            return False
        
        # 만료된 키 먼저 제거
        self._cleanup_expired()
        
        if self._get_current_size() + needed_bytes <= self._max_size_bytes:
            return True
        
        # LRU 기반 제거
        entries_by_access = sorted(
            self._cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        removed_size = 0
        for key, entry in entries_by_access:
            del self._cache[key]
            removed_size += entry.size_bytes
            self._metrics.evictions += 1
            
            if self._get_current_size() + needed_bytes <= self._max_size_bytes:
                break
        
        return self._get_current_size() + needed_bytes <= self._max_size_bytes
    
    def _cleanup_expired(self):
        """만료된 키 정리"""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self._cache[key]
            self._metrics.expired_keys += 1
    
    def _periodic_cleanup(self):
        """정기 정리 작업"""
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = current_time
            
            # 자동 최적화 실행
            if self._auto_optimize and len(self._cache) > 100:
                self.optimize_ttl()
    
    def _track_key_pattern(self, key: str):
        """키 패턴 추적"""
        # 키의 접두사 추출 (첫 번째 언더스코어까지)
        pattern = key.split('_')[0] if '_' in key else key
        self._key_patterns[pattern] = self._key_patterns.get(pattern, 0) + 1
    
    def _get_top_key_patterns(self, limit: int = 5) -> List[Dict[str, Any]]:
        """상위 키 패턴 반환"""
        sorted_patterns = sorted(
            self._key_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {'pattern': pattern, 'count': count}
            for pattern, count in sorted_patterns[:limit]
        ]
    
    def _update_memory_metrics(self):
        """메모리 메트릭 업데이트"""
        self._metrics.memory_usage_mb = self._get_current_size() / (1024 * 1024)
        self._metrics.total_keys = len(self._cache)
    
    def _get_optimization_suggestions(self) -> List[str]:
        """최적화 제안 생성"""
        suggestions = []
        metrics = self._metrics
        
        if metrics.hit_rate < 50:
            suggestions.append("캐시 히트율이 낮습니다. TTL 설정을 검토하세요.")
        
        if metrics.memory_usage_mb > (self._max_size_bytes / (1024 * 1024)) * 0.8:
            suggestions.append("메모리 사용량이 높습니다. 캐시 크기를 늘리거나 TTL을 단축하세요.")
        
        if metrics.evictions > metrics.sets * 0.1:
            suggestions.append("제거율이 높습니다. 메모리 한도를 늘리는 것을 고려하세요.")
        
        return suggestions
    
    def _get_health_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """건강 상태 기반 권장사항"""
        recommendations = []
        
        if metrics['hit_rate'] < 70:
            recommendations.append("캐시 전략을 재검토하고 TTL을 조정하세요")
        
        if metrics['memory_usage_percent'] > 85:
            recommendations.append("메모리 한도를 늘리거나 자주 사용되지 않는 데이터를 정리하세요")
        
        if len(self._key_patterns) > 20:
            recommendations.append("키 네이밍 전략을 표준화하여 관리를 개선하세요")
        
        return recommendations


# 전역 캐시 인스턴스
_global_cache = None
_cache_lock = threading.Lock()


def get_global_cache() -> GlobalCache:
    """전역 캐시 인스턴스 반환 (싱글톤)"""
    global _global_cache
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = GlobalCache()
    return _global_cache


def cached(ttl_minutes: int = 60, key_prefix: str = ""):
    """캐시 데코레이터 (성능 모니터링 포함)"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_global_cache()
            
            # 캐시 키 생성
            key_parts = [key_prefix, func.__name__]
            if args:
                key_parts.extend(str(arg) for arg in args)
            if kwargs:
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            
            cache_key = "_".join(filter(None, key_parts))
            
            # 캐시에서 조회
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 함수 실행 및 결과 캐시
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl_minutes * 60)
            
            return result
        return wrapper
    return decorator


# 캐시 성능 모니터링 유틸리티 함수들
def get_cache_metrics() -> Dict[str, Any]:
    """캐시 메트릭 조회"""
    return get_global_cache().get_metrics()


def optimize_cache_ttl() -> Dict[str, Any]:
    """캐시 TTL 최적화"""
    return get_global_cache().optimize_ttl()


def get_cache_health() -> Dict[str, Any]:
    """캐시 건강 상태 조회"""
    return get_global_cache().get_cache_health()


def export_cache_report(format: str = 'json') -> str:
    """캐시 성능 리포트 내보내기"""
    return get_global_cache().export_metrics(format) 