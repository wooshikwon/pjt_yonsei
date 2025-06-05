#!/usr/bin/env python3
"""
데이터 로딩 기능 테스트

다양한 형식의 데이터 파일 로딩 및 프로파일링 기능을 테스트합니다.
"""

import pandas as pd
import json
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가 (상위 디렉토리)
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing.data_loader import DataLoader
from config.settings import INPUT_DATA_DEFAULT_DIR


def test_csv_loading():
    """CSV 파일 로딩 테스트"""
    print("=== CSV 파일 로딩 테스트 ===")
    
    try:
        loader = DataLoader()
        df = loader.load_data("input_data/sample_survey_data.csv")
        
        print(f"✅ CSV 로딩 성공: {df.shape[0]}행 {df.shape[1]}열")
        print(f"컬럼: {list(df.columns)}")
        print(f"첫 3행:\n{df.head(3)}")
        
        # 데이터 프로파일링 테스트
        profile = loader.get_data_profile(df)
        print(f"✅ 데이터 프로파일링 성공")
        print(f"기본 정보: {profile['basic_info']}")
        
        return True
        
    except Exception as e:
        print(f"❌ CSV 로딩 실패: {e}")
        return False


def test_excel_creation_and_loading():
    """Excel 파일 생성 및 로딩 테스트"""
    print("\n=== Excel 파일 생성 및 로딩 테스트 ===")
    
    try:
        # 먼저 CSV를 Excel로 변환
        loader = DataLoader()
        df_csv = loader.load_data("input_data/sample_survey_data.csv")
        
        excel_path = "input_data/sample_survey_data.xlsx"
        df_csv.to_excel(excel_path, index=False)
        print(f"✅ Excel 파일 생성: {excel_path}")
        
        # Excel 파일 로딩 테스트
        df_excel = loader.load_data(excel_path)
        print(f"✅ Excel 로딩 성공: {df_excel.shape[0]}행 {df_excel.shape[1]}열")
        
        # 데이터 일치성 확인
        if df_csv.shape == df_excel.shape:
            print("✅ CSV와 Excel 데이터 형태 일치")
        else:
            print("⚠️ CSV와 Excel 데이터 형태 불일치")
            
        return True
        
    except Exception as e:
        print(f"❌ Excel 테스트 실패: {e}")
        return False


def test_parquet_creation_and_loading():
    """Parquet 파일 생성 및 로딩 테스트"""
    print("\n=== Parquet 파일 생성 및 로딩 테스트 ===")
    
    try:
        # 먼저 CSV를 Parquet으로 변환
        loader = DataLoader()
        df_csv = loader.load_data("input_data/sample_survey_data.csv")
        
        parquet_path = "input_data/sample_survey_data.parquet"
        df_csv.to_parquet(parquet_path, index=False)
        print(f"✅ Parquet 파일 생성: {parquet_path}")
        
        # Parquet 파일 로딩 테스트
        df_parquet = loader.load_data(parquet_path)
        print(f"✅ Parquet 로딩 성공: {df_parquet.shape[0]}행 {df_parquet.shape[1]}열")
        
        # 데이터 일치성 확인
        if df_csv.shape == df_parquet.shape:
            print("✅ CSV와 Parquet 데이터 형태 일치")
        else:
            print("⚠️ CSV와 Parquet 데이터 형태 불일치")
            
        return True
        
    except Exception as e:
        print(f"❌ Parquet 테스트 실패: {e}")
        print("   pyarrow가 설치되지 않은 것 같습니다. 다음 명령어로 설치하세요:")
        print("   pip install pyarrow")
        return False


def test_json_creation_and_loading():
    """JSON 파일 생성 및 로딩 테스트"""
    print("\n=== JSON 파일 생성 및 로딩 테스트 ===")
    
    try:
        # 먼저 CSV를 JSON으로 변환
        loader = DataLoader()
        df_csv = loader.load_data("input_data/sample_survey_data.csv")
        
        json_path = "input_data/sample_survey_data.json"
        df_csv.to_json(json_path, orient='records', force_ascii=False, indent=2)
        print(f"✅ JSON 파일 생성: {json_path}")
        
        # JSON 파일 로딩 테스트
        df_json = loader.load_data(json_path)
        print(f"✅ JSON 로딩 성공: {df_json.shape[0]}행 {df_json.shape[1]}열")
        
        # 데이터 일치성 확인
        if df_csv.shape == df_json.shape:
            print("✅ CSV와 JSON 데이터 형태 일치")
        else:
            print("⚠️ CSV와 JSON 데이터 형태 불일치")
            
        return True
        
    except Exception as e:
        print(f"❌ JSON 테스트 실패: {e}")
        return False


def test_data_profile_analysis():
    """데이터 프로파일링 상세 분석 테스트"""
    print("\n=== 데이터 프로파일링 상세 분석 ===")
    
    try:
        loader = DataLoader()
        df = loader.load_data("input_data/sample_survey_data.csv")
        
        profile = loader.get_data_profile(df)
        
        print("📊 컬럼별 분석 결과:")
        for col_name, col_info in profile['columns'].items():
            print(f"  {col_name}:")
            print(f"    - 타입: {col_info['inferred_type']}")
            print(f"    - 고유값 수: {col_info['unique_count']}")
            print(f"    - 결측치: {col_info['null_count']}개 ({col_info['null_percentage']:.1f}%)")
            
        print(f"\n📈 전체 요약:")
        print(f"  - 연속형 변수: {len([c for c, info in profile['columns'].items() if info['inferred_type'] == '연속형'])}")
        print(f"  - 범주형 변수: {len([c for c, info in profile['columns'].items() if info['inferred_type'] == '범주형'])}")
        print(f"  - 식별자 변수: {len([c for c, info in profile['columns'].items() if info['inferred_type'] == '식별자'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 데이터 프로파일링 실패: {e}")
        return False


def main():
    """모든 테스트 실행"""
    print("🚀 데이터 로딩 기능 종합 테스트 시작\n")
    
    tests = [
        test_csv_loading,
        test_excel_creation_and_loading,
        test_parquet_creation_and_loading,
        test_json_creation_and_loading,
        test_data_profile_analysis
    ]
    
    results = []
    for test_func in tests:
        results.append(test_func())
    
    # 결과 요약
    print("\n" + "="*50)
    print("🎯 테스트 결과 요약")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ 통과: {passed}/{total}")
    print(f"❌ 실패: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 모든 테스트 통과! 데이터 로딩 기능이 정상적으로 작동합니다.")
    else:
        print(f"\n⚠️ {total - passed}개 테스트가 실패했습니다. 위의 오류 메시지를 확인하세요.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 