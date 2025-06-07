# 파일명: utils/data_loader.py

import pandas as pd
from typing import Tuple
from pathlib import Path

from .input_validator import validate_file_path

class DataLoader:
    """데이터 로딩과 관련된 기능을 제공하는 클래스"""

    @staticmethod
    def load_file(file_path: str) -> pd.DataFrame:
        """
        주어진 파일 경로의 파일을 읽어 DataFrame으로 반환합니다.
        지원 포맷: CSV, Excel(.xls, .xlsx), JSON, Parquet.
        :param file_path: 로드할 파일의 절대 또는 상대 경로
        :return: pandas.DataFrame
        :raises ValueError: 파일 경로가 유효하지 않거나, 지원하지 않는 포맷인 경우
        """
        # 1. 파일 경로 유효성 검증
        abs_path = validate_file_path(file_path)

        # 2. 확장자 기반 분기 처리
        suffix = abs_path.suffix.lower()
        if suffix == '.csv':
            df = pd.read_csv(abs_path)
        elif suffix in ('.xls', '.xlsx'):
            df = pd.read_excel(abs_path)
        elif suffix == '.json':
            df = pd.read_json(abs_path)
        elif suffix == '.parquet':
            df = pd.read_parquet(abs_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")

        return df

    @staticmethod
    def get_dataframe_shape(df: pd.DataFrame) -> Tuple[int, int]:
        """
        DataFrame의 (행 개수, 열 개수)를 반환합니다.
        :param df: pandas.DataFrame
        :return: (row_count, column_count)
        """
        return df.shape
