# 파일명: services/utils/helpers.py

import os
from pathlib import Path


def get_file_extension(file_path: str) -> str:
    """
    파일 경로에서 확장자를 소문자 문자열로 반환합니다.
    :param file_path: 파일 경로
    :return: '.csv', '.xlsx' 등
    """
    return Path(file_path).suffix.lower()


def get_file_size_mb(file_path: str) -> float:
    """
    파일 크기를 메가바이트 단위로 반환합니다.
    :param file_path: 파일 경로
    :return: 파일 크기 (단위: MB)
    :raises ValueError: 파일이 존재하지 않는 경우
    """
    path_obj = Path(file_path)
    if not path_obj.exists() or not path_obj.is_file():
        raise ValueError(f"파일이 존재하지 않거나 파일이 아닙니다: {file_path}")

    size_bytes = os.path.getsize(path_obj)
    size_mb = size_bytes / (1024 * 1024)
    return round(size_mb, 4)


def is_file_readable(file_path: str) -> bool:
    """
    파일이 존재하고 읽기 권한이 있는지 여부를 반환합니다.
    :param file_path: 파일 경로
    :return: 읽기 가능하면 True, 아니면 False
    """
    path_obj = Path(file_path)
    return path_obj.exists() and path_obj.is_file() and os.access(path_obj, os.R_OK)
