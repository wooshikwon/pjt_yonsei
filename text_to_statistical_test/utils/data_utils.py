"""
Data Utilities

데이터 파일 검색, 미리보기, 파일 크기 포맷팅 등 데이터 관련 유틸리티 함수들
"""

import glob
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

from config.settings import INPUT_DATA_DEFAULT_DIR


def get_available_data_files(data_dir: str = None) -> List[str]:
    """
    지정된 디렉토리에서 사용 가능한 데이터 파일들을 찾습니다.
    
    Args:
        data_dir: 데이터 파일을 찾을 디렉토리 경로
        
    Returns:
        List[str]: 찾은 데이터 파일들의 경로 리스트
    """
    if data_dir is None:
        data_dir = INPUT_DATA_DEFAULT_DIR
    
    # 지원하는 파일 확장자들
    supported_extensions = ['*.csv', '*.xlsx', '*.xls', '*.json', '*.parquet', '*.tsv']
    
    data_files = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return []
    
    for ext in supported_extensions:
        files = glob.glob(str(data_path / ext))
        data_files.extend(files)
    
    # 파일명으로 정렬
    data_files.sort()
    
    return data_files


def validate_data_file(file_path: str) -> Dict[str, Any]:
    """
    데이터 파일의 유효성을 검사합니다.
    
    Args:
        file_path: 검사할 파일 경로
        
    Returns:
        Dict: 검사 결과 정보
    """
    result = {
        'is_valid': False,
        'error_message': '',
        'file_info': {},
        'preview': None
    }
    
    try:
        if not os.path.exists(file_path):
            result['error_message'] = '파일이 존재하지 않습니다.'
            return result
        
        # 파일 기본 정보
        file_info = get_file_info(file_path)
        result['file_info'] = file_info
        
        # 파일 미리보기
        preview = preview_selected_data(file_path, rows=5)
        if preview is not None:
            result['preview'] = preview
            result['is_valid'] = True
        else:
            result['error_message'] = '지원하지 않는 파일 형식이거나 파일을 읽을 수 없습니다.'
        
    except Exception as e:
        result['error_message'] = f'파일 검사 중 오류: {str(e)}'
    
    return result


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    파일의 기본 정보를 반환합니다.
    
    Args:
        file_path: 파일 경로
        
    Returns:
        Dict: 파일 정보
    """
    try:
        file_stat = os.stat(file_path)
        file_path_obj = Path(file_path)
        
        return {
            'filename': file_path_obj.name,
            'extension': file_path_obj.suffix,
            'size_bytes': file_stat.st_size,
            'size_formatted': format_file_size(file_stat.st_size),
            'modified_time': file_stat.st_mtime,
            'is_readable': os.access(file_path, os.R_OK)
        }
    except Exception as e:
        return {
            'filename': Path(file_path).name,
            'error': str(e)
        }


def format_file_size(size_bytes: int) -> str:
    """
    바이트 크기를 사람이 읽기 쉬운 형식으로 변환합니다.
    
    Args:
        size_bytes: 파일 크기 (바이트)
        
    Returns:
        str: 포맷된 파일 크기
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def preview_selected_data(file_path: str, rows: int = 10) -> Optional[pd.DataFrame]:
    """
    선택된 데이터 파일의 미리보기를 제공합니다.
    
    Args:
        file_path: 데이터 파일 경로
        rows: 미리보기할 행 수
        
    Returns:
        pd.DataFrame or None: 미리보기 데이터 또는 None (실패 시)
    """
    try:
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.csv':
            # CSV 파일 인코딩 자동 감지
            try:
                df = pd.read_csv(file_path, nrows=rows)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, nrows=rows, encoding='cp949')
                
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, nrows=rows)
            
        elif file_extension == '.json':
            # JSON 파일은 전체를 읽고 처음 n행만 선택
            df_full = pd.read_json(file_path)
            df = df_full.head(rows)
            
        elif file_extension == '.parquet':
            df = pd.read_parquet(file_path)
            df = df.head(rows)
            
        elif file_extension == '.tsv':
            try:
                df = pd.read_csv(file_path, sep='\t', nrows=rows)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, sep='\t', nrows=rows, encoding='cp949')
        else:
            return None
            
        return df
        
    except Exception as e:
        print(f"데이터 미리보기 오류: {e}")
        return None


def analyze_data_structure(file_path: str) -> Dict[str, Any]:
    """
    데이터 파일의 구조를 분석합니다.
    
    Args:
        file_path: 데이터 파일 경로
        
    Returns:
        Dict: 데이터 구조 분석 결과
    """
    try:
        # 전체 데이터 로딩 (크기 제한)
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.csv':
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='cp949')
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_extension == '.json':
            df = pd.read_json(file_path)
        elif file_extension == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_extension == '.tsv':
            try:
                df = pd.read_csv(file_path, sep='\t')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, sep='\t', encoding='cp949')
        else:
            return {'error': '지원하지 않는 파일 형식'}
        
        # 데이터 구조 분석
        analysis = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'column_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': list(df.select_dtypes(include=['number']).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'datetime_columns': list(df.select_dtypes(include=['datetime']).columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'has_missing_values': df.isnull().any().any()
        }
        
        # 수치형 컬럼의 기본 통계
        if analysis['numeric_columns']:
            numeric_stats = df[analysis['numeric_columns']].describe().to_dict()
            analysis['numeric_statistics'] = numeric_stats
        
        # 범주형 컬럼의 고유값 개수
        if analysis['categorical_columns']:
            categorical_stats = {}
            for col in analysis['categorical_columns']:
                categorical_stats[col] = {
                    'unique_count': df[col].nunique(),
                    'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None
                }
            analysis['categorical_statistics'] = categorical_stats
        
        return analysis
        
    except Exception as e:
        return {'error': f'데이터 구조 분석 오류: {str(e)}'}


def get_data_summary_for_rag(file_path: str) -> Dict[str, Any]:
    """
    RAG 시스템을 위한 데이터 요약 정보를 생성합니다.
    
    Args:
        file_path: 데이터 파일 경로
        
    Returns:
        Dict: RAG 컨텍스트용 데이터 요약
    """
    analysis = analyze_data_structure(file_path)
    
    if 'error' in analysis:
        return analysis
    
    # RAG 시스템에 최적화된 요약 정보
    rag_summary = {
        'file_info': get_file_info(file_path),
        'dimensions': {
            'rows': analysis['total_rows'],
            'columns': analysis['total_columns']
        },
        'columns': analysis['columns'],
        'data_types': {
            'numeric': analysis['numeric_columns'],
            'categorical': analysis['categorical_columns'],
            'datetime': analysis['datetime_columns']
        },
        'data_quality': {
            'has_missing_values': analysis['has_missing_values'],
            'missing_by_column': analysis['missing_values'],
            'memory_usage_mb': round(analysis['memory_usage'] / 1024 / 1024, 2)
        }
    }
    
    # 통계적 분석을 위한 추가 정보
    if analysis.get('numeric_statistics'):
        rag_summary['statistical_summary'] = {
            'numeric_columns_count': len(analysis['numeric_columns']),
            'potential_target_variables': analysis['numeric_columns'][:5],  # 상위 5개
            'has_sufficient_numeric_data': len(analysis['numeric_columns']) >= 2
        }
    
    if analysis.get('categorical_statistics'):
        categorical_info = analysis['categorical_statistics']
        rag_summary['categorical_summary'] = {
            'categorical_columns_count': len(analysis['categorical_columns']),
            'potential_grouping_variables': [
                col for col, stats in categorical_info.items() 
                if stats['unique_count'] <= 20  # 그룹핑에 적합한 컬럼들
            ],
            'high_cardinality_columns': [
                col for col, stats in categorical_info.items() 
                if stats['unique_count'] > 50  # 고유값이 많은 컬럼들
            ]
        }
    
    return rag_summary 