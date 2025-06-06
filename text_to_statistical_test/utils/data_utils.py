"""
Data Utils

파일 시스템 레벨의 데이터 파일 관련 유틸리티
(Pandas에 의존하지 않는 순수 파일 시스템 유틸리티)
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from utils.input_validator import InputValidator
from utils.helpers import get_file_extension, get_file_size_mb, is_file_readable

logger = logging.getLogger(__name__)

def get_available_data_files(data_dir: str = "input_data/data_files") -> List[str]:
    """
    지정된 디렉토리에서 사용 가능한 데이터 파일 목록을 반환합니다.
    
    Args:
        data_dir: 데이터 파일이 있는 디렉토리 경로
        
    Returns:
        사용 가능한 데이터 파일 경로 목록
    """
    try:
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.warning(f"데이터 디렉토리가 존재하지 않습니다: {data_dir}")
            return []
        
        # 지원하는 파일 확장자
        supported_extensions = {'.csv', '.xlsx', '.xls', '.json', '.parquet', '.tsv'}
        
        data_files = []
        for file_path in data_path.iterdir():
            if (file_path.is_file() and 
                file_path.suffix.lower() in supported_extensions and
                is_file_readable(file_path)):
                data_files.append(str(file_path))
        
        return sorted(data_files)
        
    except Exception as e:
        logger.error(f"데이터 파일 목록 조회 중 오류: {e}")
        return []

def get_file_basic_info(file_path: str) -> Dict[str, Any]:
    """
    파일의 기본 정보를 반환합니다 (파일 시스템 레벨).
    
    Args:
        file_path: 파일 경로
        
    Returns:
        파일 기본 정보 딕셔너리
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return {"error": "파일이 존재하지 않습니다"}
        
        stat = path.stat()
        return {
            "name": path.name,
            "extension": get_file_extension(path),
            "size_mb": get_file_size_mb(path),
            "size_bytes": stat.st_size,
            "created_time": stat.st_ctime,
            "modified_time": stat.st_mtime,
            "is_readable": is_file_readable(path)
        }
        
    except Exception as e:
        logger.error(f"파일 정보 조회 중 오류: {e}")
        return {"error": str(e)}

def validate_file_access(file_path: str) -> Dict[str, Any]:
    """
    파일 접근 가능성을 검증합니다 (InputValidator 사용).
    
    Args:
        file_path: 검증할 파일 경로
        
    Returns:
        검증 결과 딕셔너리
    """
    try:
        validator = InputValidator()
        
        # 기본 파일 경로 검증
        path_validation = validator.validate_file_path(file_path)
        if not path_validation["is_valid"]:
            return {
                "is_valid": False,
                "error": path_validation["error"],
                "file_info": None
            }
        
        # 파일 기본 정보 수집
        file_info = get_file_basic_info(file_path)
        
        return {
            "is_valid": True,
            "error": None,
            "file_info": file_info
        }
        
    except Exception as e:
        logger.error(f"파일 검증 중 오류: {e}")
        return {
            "is_valid": False,
            "error": str(e),
            "file_info": None
        }

def compare_data_files(file_paths: List[str]) -> Dict[str, Any]:
    """
    여러 데이터 파일들의 기본 정보를 비교합니다.
    
    Args:
        file_paths: 비교할 파일 경로 목록
        
    Returns:
        파일 비교 결과
    """
    try:
        comparison = {
            "files": [],
            "summary": {
                "total_files": len(file_paths),
                "valid_files": 0,
                "total_size_mb": 0.0,
                "file_types": {}
            }
        }
        
        for file_path in file_paths:
            file_info = get_file_basic_info(file_path)
            validation = validate_file_access(file_path)
            
            file_data = {
                "path": file_path,
                "info": file_info,
                "is_valid": validation["is_valid"]
            }
            
            comparison["files"].append(file_data)
            
            if validation["is_valid"] and "error" not in file_info:
                comparison["summary"]["valid_files"] += 1
                comparison["summary"]["total_size_mb"] += file_info.get("size_mb", 0)
                
                ext = file_info.get("extension", "unknown")
                comparison["summary"]["file_types"][ext] = comparison["summary"]["file_types"].get(ext, 0) + 1
        
        return comparison
        
    except Exception as e:
        logger.error(f"파일 비교 중 오류: {e}")
        return {"error": str(e)}

def create_data_directory_structure(base_dir: str = "input_data") -> bool:
    """
    필요한 데이터 디렉토리 구조를 생성합니다.
    
    Args:
        base_dir: 기본 디렉토리 경로
        
    Returns:
        성공 여부
    """
    try:
        directories = [
            f"{base_dir}/data_files",
            f"{base_dir}/metadata/database_schemas",
            f"{base_dir}/metadata/data_dictionaries"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        logger.info(f"데이터 디렉토리 구조 생성 완료: {base_dir}")
        return True
        
    except Exception as e:
        logger.error(f"디렉토리 구조 생성 중 오류: {e}")
        return False 