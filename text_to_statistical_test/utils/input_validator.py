# 파일명: services/utils/input_validator.py

from pathlib import Path


def validate_file_path(file_path: str) -> Path:
    """
    주어진 파일 경로의 유효성을 검증하고, 절대 경로를 반환합니다.
    검증 항목:
      1. 파일이 존재하는지
      2. 읽기 가능한 파일인지
      3. 지원 확장자인지 (.csv, .xls, .xlsx, .json, .parquet)
    :param file_path: 검사할 파일 경로 (상대 또는 절대)
    :return: pathlib.Path 객체 (절대 경로)
    :raises ValueError: 경로가 유효하지 않거나, 읽을 수 없거나, 확장자가 지원되지 않는 경우
    """
    path_obj = Path(file_path)

    if not path_obj.exists():
        raise ValueError(f"파일이 존재하지 않습니다: {file_path}")
    if not path_obj.is_file():
        raise ValueError(f"파일이 아닙니다: {file_path}")
    if not os.access(path_obj, os.R_OK):
        raise ValueError(f"파일을 읽을 권한이 없습니다: {file_path}")

    suffix = path_obj.suffix.lower()
    if suffix not in {'.csv', '.xls', '.xlsx', '.json', '.parquet'}:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")

    return path_obj.resolve()
