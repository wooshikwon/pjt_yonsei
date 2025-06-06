"""
기타 헬퍼 함수들

프로젝트 전반에서 사용되는 작은 유틸리티 함수들
"""

import re
import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import hashlib
import uuid
import time
import csv

logger = logging.getLogger(__name__)

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    안전한 JSON 파싱
    
    Args:
        json_str: JSON 문자열
        default: 파싱 실패 시 반환할 기본값
        
    Returns:
        파싱된 객체 또는 기본값
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"JSON 파싱 실패: {str(e)}")
        return default

def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """
    안전한 JSON 직렬화
    
    Args:
        obj: 직렬화할 객체
        default: 직렬화 실패 시 반환할 기본값
        
    Returns:
        JSON 문자열 또는 기본값
    """
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as e:
        logger.warning(f"JSON 직렬화 실패: {str(e)}")
        return default

def clean_text(text: str) -> str:
    """
    텍스트 정리 (특수문자 제거, 공백 정리 등)
    
    Args:
        text: 정리할 텍스트
        
    Returns:
        정리된 텍스트
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 연속된 공백을 하나로 변경
    text = re.sub(r'\s+', ' ', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    # 특수 문자 정리 (선택적)
    # text = re.sub(r'[^\w\s가-힣]', '', text)
    
    return text

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    텍스트 길이 제한
    
    Args:
        text: 원본 텍스트
        max_length: 최대 길이
        suffix: 생략 표시
        
    Returns:
        제한된 길이의 텍스트
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def format_number(number: Union[int, float], decimal_places: int = 2) -> str:
    """
    숫자를 읽기 좋은 형태로 포맷팅
    
    Args:
        number: 포맷팅할 숫자
        decimal_places: 소수점 자릿수
        
    Returns:
        포맷팅된 숫자 문자열
    """
    if pd.isna(number):
        return "N/A"
    
    try:
        if isinstance(number, float):
            return f"{number:,.{decimal_places}f}"
        else:
            return f"{number:,}"
    except (ValueError, TypeError):
        return str(number)

def format_percentage(value: Union[int, float], decimal_places: int = 1) -> str:
    """
    백분율 포맷팅
    
    Args:
        value: 백분율 값 (0-100)
        decimal_places: 소수점 자릿수
        
    Returns:
        포맷팅된 백분율 문자열
    """
    if pd.isna(value):
        return "N/A"
    
    try:
        return f"{value:.{decimal_places}f}%"
    except (ValueError, TypeError):
        return str(value)

def safe_divide(numerator: Union[int, float], denominator: Union[int, float], default: float = 0.0) -> float:
    """
    안전한 나누기 연산 (0으로 나누기 방지)
    
    Args:
        numerator: 분자
        denominator: 분모
        default: 0으로 나누기 시 반환할 기본값
        
    Returns:
        나누기 결과 또는 기본값
    """
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return default

def generate_unique_id(prefix: str = "") -> str:
    """
    고유 ID 생성
    
    Args:
        prefix: ID 접두사
        
    Returns:
        고유 ID 문자열
    """
    unique_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if prefix:
        return f"{prefix}_{timestamp}_{unique_id}"
    else:
        return f"{timestamp}_{unique_id}"

def create_hash(data: Union[str, Dict, List]) -> str:
    """
    데이터의 해시값 생성
    
    Args:
        data: 해시할 데이터
        
    Returns:
        MD5 해시 문자열
    """
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, sort_keys=True, default=str)
    else:
        data_str = str(data)
    
    return hashlib.md5(data_str.encode()).hexdigest()

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    리스트를 청크 단위로 분할
    
    Args:
        lst: 분할할 리스트
        chunk_size: 청크 크기
        
    Returns:
        청크로 분할된 리스트들의 리스트
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    중첩된 딕셔너리를 평탄화
    
    Args:
        d: 평탄화할 딕셔너리
        parent_key: 부모 키
        sep: 키 구분자
        
    Returns:
        평탄화된 딕셔너리
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    평탄화된 딕셔너리를 중첩 구조로 복원
    
    Args:
        d: 복원할 딕셔너리
        sep: 키 구분자
        
    Returns:
        중첩 구조의 딕셔너리
    """
    result = {}
    for key, value in d.items():
        keys = key.split(sep)
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    return result

def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    여러 딕셔너리를 병합
    
    Args:
        *dicts: 병합할 딕셔너리들
        
    Returns:
        병합된 딕셔너리
    """
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result

def filter_dict(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """
    딕셔너리에서 특정 키들만 필터링
    
    Args:
        d: 원본 딕셔너리
        keys: 유지할 키 목록
        
    Returns:
        필터링된 딕셔너리
    """
    return {k: v for k, v in d.items() if k in keys}

def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    파일 확장자 추출
    
    Args:
        file_path: 파일 경로
        
    Returns:
        소문자 확장자 (점 포함)
    """
    return Path(file_path).suffix.lower()

def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    파일 크기를 MB 단위로 반환
    
    Args:
        file_path: 파일 경로
        
    Returns:
        파일 크기 (MB)
    """
    try:
        size_bytes = Path(file_path).stat().st_size
        return round(size_bytes / (1024 * 1024), 2)
    except (OSError, FileNotFoundError):
        return 0.0

def ensure_directory(dir_path: Union[str, Path]) -> Path:
    """
    디렉토리가 존재하지 않으면 생성
    
    Args:
        dir_path: 디렉토리 경로
        
    Returns:
        Path 객체
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def is_numeric_column(series: pd.Series) -> bool:
    """
    시리즈가 숫자형 데이터인지 확인
    
    Args:
        series: 판다스 시리즈
        
    Returns:
        숫자형 여부
    """
    return pd.api.types.is_numeric_dtype(series)

def is_categorical_column(series: pd.Series, max_unique_ratio: float = 0.1) -> bool:
    """
    시리즈가 범주형 데이터인지 확인
    
    Args:
        series: 판다스 시리즈
        max_unique_ratio: 최대 고유값 비율
        
    Returns:
        범주형 여부
    """
    if pd.api.types.is_categorical_dtype(series) or series.dtype == 'object':
        unique_ratio = series.nunique() / len(series)
        return unique_ratio <= max_unique_ratio
    return False

def detect_column_type(series: pd.Series) -> str:
    """
    컬럼의 데이터 타입 감지
    
    Args:
        series: 판다스 시리즈
        
    Returns:
        데이터 타입 ('numeric', 'categorical', 'datetime', 'text')
    """
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'
    elif pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    elif is_categorical_column(series):
        return 'categorical'
    else:
        return 'text'

def format_duration(seconds: float) -> str:
    """
    초 단위 시간을 읽기 좋은 형태로 포맷팅
    
    Args:
        seconds: 초 단위 시간
        
    Returns:
        포맷팅된 시간 문자열
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}초"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}분"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}시간"

def validate_email(email: str) -> bool:
    """
    이메일 주소 유효성 검증
    
    Args:
        email: 이메일 주소
        
    Returns:
        유효성 여부
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def extract_numbers(text: str) -> List[float]:
    """
    텍스트에서 숫자 추출
    
    Args:
        text: 텍스트
        
    Returns:
        추출된 숫자 리스트
    """
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    return [float(match) for match in matches if match]

def sanitize_filename(filename: str) -> str:
    """
    파일명에서 사용할 수 없는 문자 제거
    
    Args:
        filename: 원본 파일명
        
    Returns:
        정리된 파일명
    """
    # 윈도우에서 사용할 수 없는 문자들
    invalid_chars = r'[<>:"/\\|?*]'
    
    # 특수 문자를 언더스코어로 대체
    sanitized = re.sub(invalid_chars, '_', filename)
    
    # 연속된 언더스코어를 하나로 변경
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # 앞뒤 언더스코어 제거
    sanitized = sanitized.strip('_')
    
    return sanitized or 'unnamed_file'

def get_memory_usage_mb() -> float:
    """
    현재 프로세스의 메모리 사용량 조회 (MB)
    
    Returns:
        메모리 사용량 (MB)
    """
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        return round(memory_mb, 2)
    except ImportError:
        return 0.0

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    두 텍스트 간의 유사도 계산 (Jaccard 유사도 기반)
    
    Args:
        text1: 첫 번째 텍스트
        text2: 두 번째 텍스트
        
    Returns:
        0-1 사이의 유사도 점수
    """
    try:
        # 텍스트 전처리
        words1 = set(clean_text(text1.lower()).split())
        words2 = set(clean_text(text2.lower()).split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard 유사도 계산
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        similarity = len(intersection) / len(union)
        return similarity
        
    except Exception:
        return 0.0

def extract_keywords(text: str, max_words: int = 10) -> List[str]:
    """
    텍스트에서 키워드 추출
    
    Args:
        text: 분석할 텍스트
        max_words: 최대 키워드 수
        
    Returns:
        추출된 키워드 리스트
    """
    try:
        # 기본 불용어 목록 (한국어 + 영어)
        stop_words = {
            '의', '가', '이', '은', '는', '을', '를', '에', '에서', '로', '으로', '와', '과',
            '그', '저', '것', '수', '등', '때', '한', '하는', '있는', '되는', '같은',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        # 텍스트 정리 및 토큰화
        cleaned_text = clean_text(text.lower())
        words = cleaned_text.split()
        
        # 불용어 제거 및 길이 필터링
        keywords = []
        for word in words:
            if (len(word) >= 2 and 
                word not in stop_words and 
                not word.isdigit() and
                word.isalnum()):
                keywords.append(word)
        
        # 빈도 계산
        from collections import Counter
        word_counts = Counter(keywords)
        
        # 빈도 순으로 정렬하여 상위 키워드 반환
        top_keywords = [word for word, count in word_counts.most_common(max_words)]
        
        return top_keywords
        
    except Exception:
        # 오류 발생 시 단순 분할 반환
        return text.split()[:max_words]

def retry_on_exception(max_attempts: int = 3, delay: float = 1.0, 
                      exceptions: Tuple = (Exception,)):
    """
    예외 발생 시 재시도하는 데코레이터
    
    Args:
        max_attempts: 최대 시도 횟수
        delay: 재시도 간격 (초)
        exceptions: 재시도할 예외 타입들
        
    Returns:
        데코레이터 함수
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
                        continue
                    else:
                        raise last_exception
            
        return wrapper
    return decorator

def detect_csv_delimiter(file_path: str) -> str:
    """
    CSV 파일의 구분자를 자동으로 감지합니다.
    
    Args:
        file_path: CSV 파일 경로
        
    Returns:
        감지된 구분자 문자열
        
    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
        ValueError: 구분자를 감지할 수 없는 경우
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            # 첫 몇 줄만 읽어서 구분자 감지
            sample = file.read(8192)  # 8KB 샘플
            
        # csv.Sniffer를 사용하여 구분자 감지
        sniffer = csv.Sniffer()
        try:
            delimiter = sniffer.sniff(sample, delimiters=',;\t|').delimiter
            return delimiter
        except csv.Error:
            # Sniffer가 실패하면 수동으로 감지
            delimiters = [',', ';', '\t', '|']
            delimiter_counts = {}
            
            for delimiter in delimiters:
                count = sample.count(delimiter)
                if count > 0:
                    delimiter_counts[delimiter] = count
            
            if delimiter_counts:
                # 가장 많이 나타나는 구분자 선택
                return max(delimiter_counts, key=delimiter_counts.get)
            else:
                # 기본값으로 쉼표 반환
                return ','
                
    except FileNotFoundError:
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    except Exception as e:
        logger.warning(f"구분자 감지 중 오류 발생: {e}, 기본값 ',' 사용")
        return ','

def is_file_readable(file_path: Union[str, Path]) -> bool:
    """파일이 읽기 가능한지 확인합니다."""
    try:
        path = Path(file_path)
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except (OSError, PermissionError):
        return False 