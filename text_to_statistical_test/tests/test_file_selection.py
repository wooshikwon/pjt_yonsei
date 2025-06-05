#!/usr/bin/env python3
"""
데이터 파일 선택 기능 테스트 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가 (상위 디렉토리)
sys.path.insert(0, str(Path(__file__).parent.parent))

from main_runner import find_available_data_files, display_file_selection_menu, preview_selected_data, format_file_size


def test_file_discovery():
    """파일 탐색 기능 테스트"""
    print("🔍 데이터 파일 탐색 테스트")
    print("=" * 40)
    
    files = find_available_data_files()
    print(f"발견된 파일 수: {len(files)}")
    
    for file_path in files:
        filename = Path(file_path).name
        size = Path(file_path).stat().st_size
        size_str = format_file_size(size)
        print(f"  • {filename} ({size_str})")
    
    return files


def test_preview_functionality():
    """데이터 미리보기 기능 테스트"""
    print("\n📊 데이터 미리보기 테스트")
    print("=" * 40)
    
    files = find_available_data_files()
    if files:
        # 첫 번째 파일로 미리보기 테스트
        test_file = files[0]
        print(f"테스트 파일: {Path(test_file).name}")
        
        success = preview_selected_data(test_file)
        print(f"미리보기 성공: {success}")
    else:
        print("❌ 테스트할 파일이 없습니다.")


def mock_file_selection_demo():
    """파일 선택 UI 데모 (자동 선택)"""
    print("\n🎯 파일 선택 UI 데모")
    print("=" * 40)
    
    files = find_available_data_files()
    if not files:
        print("❌ 사용 가능한 데이터 파일이 없습니다.")
        return None
    
    print("\n📊 사용 가능한 데이터 파일들:")
    print("=" * 50)
    
    for i, file_path in enumerate(files, 1):
        filename = Path(file_path).name
        file_size = Path(file_path).stat().st_size
        size_str = format_file_size(file_size)
        print(f"  {i}. {filename} ({size_str})")
    
    print("=" * 50)
    print("✅ 파일 선택 메뉴 표시 완료!")
    
    # 자동으로 첫 번째 파일 선택 (데모용)
    selected_file = files[0]
    filename = Path(selected_file).name
    print(f"🎯 데모: 자동으로 '{filename}' 선택됨")
    
    return selected_file


def main():
    """메인 테스트 함수"""
    print("🧪 CLI Multi-turn 데이터 파일 선택 시스템 테스트")
    print("=" * 60)
    
    try:
        # 1. 파일 탐색 테스트
        files = test_file_discovery()
        
        if not files:
            print("\n❌ input_data/ 디렉토리에 데이터 파일을 추가해주세요.")
            print("📋 지원 형식: CSV, Excel (xlsx/xls), JSON, Parquet, TSV")
            return
        
        # 2. 미리보기 기능 테스트
        test_preview_functionality()
        
        # 3. 파일 선택 UI 데모
        selected_file = mock_file_selection_demo()
        
        if selected_file:
            print(f"\n✅ 테스트 완료! 선택된 파일: {Path(selected_file).name}")
            print("🚀 실제 대화형 모드 실행 준비 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 