# 파일명: services/utils/ui_helpers.py

import sys
from typing import List

from services.utils.data_utils import get_available_data_files


def display_file_selection_menu(data_dir: str = "input_data/data_files") -> str:
    """
    사용자에게 입력 데이터 디렉토리 내의 파일 목록을 보여주고,
    번호를 선택받아 해당 파일 경로를 반환합니다.
    :param data_dir: 데이터 파일이 저장된 디렉토리 경로
    :return: 사용자가 선택한 파일의 절대 경로 (문자열)
    :raises ValueError: 올바르지 않은 선택 번호를 입력한 경우
    """
    files = get_available_data_files(data_dir)
    if not files:
        print(f"디렉토리 '{data_dir}'에 유효한 데이터 파일이 없습니다.")
        sys.exit(1)

    print("===== 사용 가능한 데이터 파일 목록 =====")
    for idx, file_path in enumerate(files, start=1):
        print(f"[{idx}] {file_path}")
    print("======================================")

    try:
        choice = input("분석할 파일 번호를 입력하세요: ").strip()
        choice_num = int(choice)
        if choice_num < 1 or choice_num > len(files):
            raise ValueError
        selected_file = files[choice_num - 1]
        print(f"선택된 파일: {selected_file}")
        return selected_file
    except (ValueError, IndexError):
        raise ValueError("올바른 번호를 입력해야 합니다.")
