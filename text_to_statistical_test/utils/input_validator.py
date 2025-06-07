from pathlib import Path
from typing import Set

def validate_file_path(path: str | Path, supported_extensions: Set[str] = None) -> Path:
    """
    주어진 경로가 유효한 파일인지 검증하고 Path 객체로 반환합니다.

    Args:
        path: 검증할 파일 경로.
        supported_extensions: 허용되는 파일 확장자 집합 (예: {'.csv', '.xlsx'}).

    Returns:
        절대 경로를 가리키는 Path 객체.

    Raises:
        TypeError: 경로가 제공되지 않았거나 문자열 또는 Path가 아닌 경우.
        FileNotFoundError: 파일이 존재하지 않거나 디렉토리인 경우.
        ValueError: 지원되지 않는 파일 확장자인 경우.
    """
    if not path:
        raise TypeError("파일 경로가 제공되지 않았습니다.")
    
    if not isinstance(path, (str, Path)):
        raise TypeError(f"경로는 문자열 또는 Path 객체여야 합니다: {type(path)}")

    file_path = Path(path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    if not file_path.is_file():
        raise FileNotFoundError(f"경로가 파일이 아닌 디렉토리입니다: {file_path}")

    if supported_extensions and file_path.suffix.lower() not in supported_extensions:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {file_path.suffix}. 지원되는 형식: {supported_extensions}")

    return file_path 