"""
Data Loader

데이터 로딩 및 메타데이터 생성 유틸리티
- 대용량 파일 처리 최적화
- 청크 단위 로딩 지원
- 메모리 사용량 최적화
- 성능 모니터링 포함
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, Iterator, List
import time
from datetime import datetime
import psutil
import os
from dataclasses import dataclass

from utils.global_cache import cached, get_global_cache
from utils.helpers import detect_csv_delimiter, get_file_extension, get_file_size_mb, is_file_readable
from utils.input_validator import InputValidator

logger = logging.getLogger(__name__)


@dataclass
class LoadingMetrics:
    """데이터 로딩 메트릭"""
    file_size_mb: float = 0.0
    loading_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    rows_loaded: int = 0
    columns_loaded: int = 0
    chunk_count: int = 0
    cache_hit: bool = False
    
    @property
    def loading_speed_mbps(self) -> float:
        """로딩 속도 (MB/s)"""
        return self.file_size_mb / max(self.loading_time_seconds, 0.001)
    
    @property
    def rows_per_second(self) -> float:
        """초당 로딩 행 수"""
        return self.rows_loaded / max(self.loading_time_seconds, 0.001)


class DataLoader:
    """
    데이터 로딩 클래스 (성능 최적화 포함)
    
    Features:
    - 다양한 파일 형식 지원 (CSV, Excel, JSON, Parquet)
    - 대용량 파일 청크 단위 처리
    - 메모리 사용량 최적화
    - 자동 데이터 타입 최적화
    - 캐시 통합
    - 성능 모니터링
    """
    
    def __init__(self, chunk_size_mb: int = 100, max_memory_usage_mb: int = 1000):
        """
        DataLoader 초기화
        
        Args:
            chunk_size_mb: 청크 크기 (MB)
            max_memory_usage_mb: 최대 메모리 사용량 (MB)
        """
        self.chunk_size_mb = chunk_size_mb
        self.max_memory_usage_mb = max_memory_usage_mb
        self.validator = InputValidator()
        
        # 성능 메트릭 추적
        self.loading_history: List[Dict[str, Any]] = []
        
        logger.info(f"DataLoader 초기화 완료 (청크 크기: {chunk_size_mb}MB, 최대 메모리: {max_memory_usage_mb}MB)")
    
    @cached(ttl_minutes=30, key_prefix="dataloader")
    def load_file(self, file_path: Union[str, Path], **kwargs) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        파일 로딩 (캐시 지원)
        
        Args:
            file_path: 파일 경로
            **kwargs: 추가 로딩 옵션
            
        Returns:
            (DataFrame, metadata) 튜플
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        # 메트릭 초기화
        metrics = LoadingMetrics()
        metrics.file_size_mb = get_file_size_mb(str(file_path))
        
        try:
            # 파일 검증
            if not is_file_readable(str(file_path)):
                return None, {"error": f"파일을 읽을 수 없습니다: {file_path}"}
            
            # 메모리 사용량 확인
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            if current_memory > self.max_memory_usage_mb * 0.8:
                logger.warning(f"높은 메모리 사용량 감지: {current_memory:.1f}MB")
                # 캐시 정리
                get_global_cache().clear_by_prefix("dataloader")
            
            # 대용량 파일 처리 결정
            use_chunking = metrics.file_size_mb > self.chunk_size_mb
            
            if use_chunking:
                df, metadata = self._load_large_file(file_path, **kwargs)
            else:
                df, metadata = self._load_standard_file(file_path, **kwargs)
            
            # 메트릭 업데이트
            if df is not None:
                metrics.rows_loaded = len(df)
                metrics.columns_loaded = len(df.columns)
                metrics.memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            
            metrics.loading_time_seconds = time.time() - start_time
            
            # 메타데이터에 성능 정보 추가
            metadata.update({
                "loading_metrics": {
                    "file_size_mb": metrics.file_size_mb,
                    "loading_time_seconds": metrics.loading_time_seconds,
                    "loading_speed_mbps": metrics.loading_speed_mbps,
                    "memory_usage_mb": metrics.memory_usage_mb,
                    "rows_loaded": metrics.rows_loaded,
                    "columns_loaded": metrics.columns_loaded,
                    "rows_per_second": metrics.rows_per_second,
                    "used_chunking": use_chunking
                }
            })
            
            # 로딩 이력 저장
            self._record_loading_history(file_path, metrics, metadata)
            
            return df, metadata
            
        except Exception as e:
            metrics.loading_time_seconds = time.time() - start_time
            error_msg = f"파일 로딩 실패: {e}"
            logger.error(error_msg)
            
            return None, {
                "error": error_msg,
                "loading_metrics": {
                    "file_size_mb": metrics.file_size_mb,
                    "loading_time_seconds": metrics.loading_time_seconds,
                    "success": False
                }
            }
    
    def load_file_chunked(self, file_path: Union[str, Path], chunk_size: int = None, **kwargs) -> Iterator[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        파일을 청크 단위로 로딩 (제너레이터)
        
        Args:
            file_path: 파일 경로
            chunk_size: 청크 크기 (행 수)
            **kwargs: 추가 로딩 옵션
            
        Yields:
            (DataFrame 청크, metadata) 튜플
        """
        file_path = Path(file_path)
        file_ext = get_file_extension(str(file_path))
        
        if chunk_size is None:
            # 파일 크기 기반 청크 크기 계산
            file_size_mb = get_file_size_mb(str(file_path))
            chunk_size = max(1000, int(50000 * (self.chunk_size_mb / max(file_size_mb, 1))))
        
        try:
            if file_ext == '.csv':
                delimiter = detect_csv_delimiter(str(file_path))
                
                chunk_reader = pd.read_csv(
                    file_path,
                    delimiter=delimiter,
                    chunksize=chunk_size,
                    **kwargs
                )
                
                chunk_num = 0
                for chunk in chunk_reader:
                    # 데이터 타입 최적화
                    chunk = self._optimize_dtypes(chunk)
                    
                    metadata = {
                        "chunk_number": chunk_num,
                        "chunk_size": len(chunk),
                        "columns": list(chunk.columns),
                        "dtypes": {col: str(dtype) for col, dtype in chunk.dtypes.items()}
                    }
                    
                    yield chunk, metadata
                    chunk_num += 1
                    
            else:
                # 다른 형식은 전체 로딩 후 청크 분할
                df, metadata = self._load_standard_file(file_path, **kwargs)
                if df is not None:
                    for i in range(0, len(df), chunk_size):
                        chunk = df.iloc[i:i+chunk_size].copy()
                        chunk_metadata = metadata.copy()
                        chunk_metadata.update({
                            "chunk_number": i // chunk_size,
                            "chunk_size": len(chunk),
                            "total_chunks": (len(df) + chunk_size - 1) // chunk_size
                        })
                        yield chunk, chunk_metadata
                        
        except Exception as e:
            logger.error(f"청크 로딩 실패: {e}")
            yield None, {"error": str(e)}
    
    def _load_large_file(self, file_path: Path, **kwargs) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """대용량 파일 로딩"""
        logger.info(f"대용량 파일 로딩 시작: {file_path} ({get_file_size_mb(str(file_path)):.1f}MB)")
        
        file_ext = get_file_extension(str(file_path))
        
        if file_ext == '.csv':
            return self._load_large_csv(file_path, **kwargs)
        elif file_ext in ['.xlsx', '.xls']:
            return self._load_large_excel(file_path, **kwargs)
        else:
            # 다른 형식은 표준 로딩 사용
            return self._load_standard_file(file_path, **kwargs)
    
    def _load_large_csv(self, file_path: Path, **kwargs) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """대용량 CSV 파일 로딩"""
        try:
            delimiter = detect_csv_delimiter(str(file_path))
            
            # 샘플링으로 데이터 타입 추정
            sample_df = pd.read_csv(file_path, delimiter=delimiter, nrows=1000)
            dtypes = self._infer_optimal_dtypes(sample_df)
            
            # 청크 단위로 로딩하여 메모리 효율성 확보
            chunk_size = max(10000, int(100000 * (self.chunk_size_mb / get_file_size_mb(str(file_path)))))
            
            chunks = []
            chunk_count = 0
            
            for chunk in pd.read_csv(file_path, delimiter=delimiter, chunksize=chunk_size, dtype=dtypes, **kwargs):
                # 메모리 사용량 모니터링
                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                if current_memory > self.max_memory_usage_mb:
                    logger.warning(f"메모리 한계 도달, 청크 로딩 중단: {current_memory:.1f}MB")
                    break
                
                chunks.append(chunk)
                chunk_count += 1
                
                if chunk_count % 10 == 0:
                    logger.info(f"청크 로딩 진행: {chunk_count}개 청크 완료")
            
            # 청크 결합
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                df = self._optimize_dtypes(df)
                
                metadata = self._generate_metadata(df, file_path)
                metadata["chunk_count"] = chunk_count
                metadata["loading_method"] = "chunked_csv"
                
                return df, metadata
            else:
                return None, {"error": "청크 로딩 실패"}
                
        except Exception as e:
            return None, {"error": f"대용량 CSV 로딩 실패: {e}"}
    
    def _load_large_excel(self, file_path: Path, **kwargs) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """대용량 Excel 파일 로딩"""
        try:
            # Excel 파일은 청크 로딩이 제한적이므로 메모리 모니터링 강화
            initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            df = pd.read_excel(file_path, **kwargs)
            df = self._optimize_dtypes(df)
            
            final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            memory_increase = final_memory - initial_memory
            
            if memory_increase > self.max_memory_usage_mb * 0.5:
                logger.warning(f"Excel 로딩으로 인한 높은 메모리 증가: {memory_increase:.1f}MB")
            
            metadata = self._generate_metadata(df, file_path)
            metadata["loading_method"] = "standard_excel"
            metadata["memory_increase_mb"] = memory_increase
            
            return df, metadata
            
        except Exception as e:
            return None, {"error": f"대용량 Excel 로딩 실패: {e}"}
    
    def _load_standard_file(self, file_path: Path, **kwargs) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """표준 파일 로딩"""
        file_ext = get_file_extension(str(file_path))
        
        try:
            if file_ext == '.csv':
                delimiter = detect_csv_delimiter(str(file_path))
                df = pd.read_csv(file_path, delimiter=delimiter, **kwargs)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, **kwargs)
            elif file_ext == '.json':
                df = pd.read_json(file_path, **kwargs)
            elif file_ext == '.parquet':
                df = pd.read_parquet(file_path, **kwargs)
            else:
                return None, {"error": f"지원하지 않는 파일 형식: {file_ext}"}
            
            # 데이터 타입 최적화
            df = self._optimize_dtypes(df)
            
            metadata = self._generate_metadata(df, file_path)
            metadata["loading_method"] = "standard"
            
            return df, metadata
            
        except Exception as e:
            return None, {"error": f"표준 파일 로딩 실패: {e}"}
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 타입 최적화"""
        try:
            original_memory = df.memory_usage(deep=True).sum()
            
            for col in df.columns:
                col_type = df[col].dtype
                
                if col_type == 'object':
                    # 문자열 컬럼 최적화
                    try:
                        # 숫자로 변환 가능한지 확인
                        pd.to_numeric(df[col], errors='raise')
                        df[col] = pd.to_numeric(df[col], downcast='integer')
                    except:
                        try:
                            # 날짜로 변환 가능한지 확인
                            pd.to_datetime(df[col], errors='raise')
                            df[col] = pd.to_datetime(df[col])
                        except:
                            # 카테고리로 변환 (고유값이 적은 경우)
                            if df[col].nunique() / len(df) < 0.5:
                                df[col] = df[col].astype('category')
                
                elif col_type in ['int64', 'int32']:
                    # 정수 타입 다운캐스팅
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                
                elif col_type in ['float64', 'float32']:
                    # 실수 타입 다운캐스팅
                    df[col] = pd.to_numeric(df[col], downcast='float')
            
            optimized_memory = df.memory_usage(deep=True).sum()
            memory_reduction = (original_memory - optimized_memory) / original_memory * 100
            
            if memory_reduction > 5:
                logger.info(f"데이터 타입 최적화 완료: {memory_reduction:.1f}% 메모리 절약")
            
            return df
            
        except Exception as e:
            logger.warning(f"데이터 타입 최적화 실패: {e}")
            return df
    
    def _infer_optimal_dtypes(self, sample_df: pd.DataFrame) -> Dict[str, str]:
        """샘플 데이터로부터 최적 데이터 타입 추정"""
        dtypes = {}
        
        for col in sample_df.columns:
            col_data = sample_df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            # 정수 타입 확인
            if col_data.dtype == 'object':
                try:
                    numeric_data = pd.to_numeric(col_data, errors='coerce')
                    if not numeric_data.isna().all():
                        if (numeric_data % 1 == 0).all():
                            dtypes[col] = 'int32'
                        else:
                            dtypes[col] = 'float32'
                        continue
                except:
                    pass
                
                # 카테고리 타입 확인
                if col_data.nunique() / len(col_data) < 0.5:
                    dtypes[col] = 'category'
            
            elif col_data.dtype in ['int64']:
                dtypes[col] = 'int32'
            elif col_data.dtype in ['float64']:
                dtypes[col] = 'float32'
        
        return dtypes
    
    def _generate_metadata(self, df: pd.DataFrame, file_path: Path) -> Dict[str, Any]:
        """메타데이터 생성"""
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_extension": file_path.suffix,
            "file_size_mb": get_file_size_mb(str(file_path)),
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "null_counts": df.isnull().sum().to_dict(),
            "loaded_at": datetime.now().isoformat()
        }
    
    def _record_loading_history(self, file_path: Path, metrics: LoadingMetrics, metadata: Dict[str, Any]):
        """로딩 이력 기록"""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "file_path": str(file_path),
            "file_size_mb": metrics.file_size_mb,
            "loading_time_seconds": metrics.loading_time_seconds,
            "loading_speed_mbps": metrics.loading_speed_mbps,
            "memory_usage_mb": metrics.memory_usage_mb,
            "rows_loaded": metrics.rows_loaded,
            "success": "error" not in metadata
        }
        
        self.loading_history.append(history_entry)
        
        # 이력 크기 제한 (최근 50개만 유지)
        if len(self.loading_history) > 50:
            self.loading_history = self.loading_history[-50:]
    
    def get_cached_data(self, file_path: str) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """캐시된 데이터 조회"""
        cache = get_global_cache()
        cache_key = f"dataloader_load_file_{file_path}"
        return cache.get(cache_key)
    
    def clear_cache(self):
        """DataLoader 캐시 정리"""
        cache = get_global_cache()
        cache.clear_by_prefix("dataloader")
        logger.info("DataLoader 캐시 정리 완료")
    
    def get_loading_stats(self) -> Dict[str, Any]:
        """로딩 통계 반환"""
        if not self.loading_history:
            return {"message": "로딩 이력이 없습니다"}
        
        successful_loads = [h for h in self.loading_history if h["success"]]
        
        if not successful_loads:
            return {"message": "성공한 로딩이 없습니다"}
        
        total_files = len(successful_loads)
        total_size_mb = sum(h["file_size_mb"] for h in successful_loads)
        total_time = sum(h["loading_time_seconds"] for h in successful_loads)
        total_rows = sum(h["rows_loaded"] for h in successful_loads)
        
        avg_speed = sum(h["loading_speed_mbps"] for h in successful_loads) / total_files
        avg_memory = sum(h["memory_usage_mb"] for h in successful_loads) / total_files
        
        return {
            "total_files_loaded": total_files,
            "total_size_mb": total_size_mb,
            "total_loading_time_seconds": total_time,
            "total_rows_loaded": total_rows,
            "average_loading_speed_mbps": avg_speed,
            "average_memory_usage_mb": avg_memory,
            "success_rate": len(successful_loads) / len(self.loading_history) * 100,
            "recent_loads": self.loading_history[-5:]  # 최근 5개
        }

# 전역 인스턴스 생성
data_loader = DataLoader() 