# 파일명: services/utils/data_utils.py

from pathlib import Path
from typing import List


def get_available_data_files(data_dir: str = "input_data/data_files") -> List[str]:
    """
    지정한 디렉토리 아래의 사용 가능한 데이터 파일 목록을 반환합니다.
    지원 확장자: .csv, .xls, .xlsx, .json, .parquet
    :param data_dir: 데이터 파일이 저장된 디렉토리 경로 (상대 또는 절대)
    :return: 파일 경로 리스트 (절대 경로 문자열)
    """
    allowed_suffixes = {'.csv', '.xls', '.xlsx', '.json', '.parquet'}
    base_path = Path(data_dir)

    if not base_path.exists() or not base_path.is_dir():
        return []

    files = []
    for file_path in base_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in allowed_suffixes:
            files.append(str(file_path.resolve()))
    return sorted(files)
