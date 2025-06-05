#!/usr/bin/env python3
"""
🔬 Statistical Analysis Assistant
비즈니스 컨텍스트 인식 AI 통계 분석 시스템

LLM Agent + Enhanced RAG 시스템 기반으로 사용자의 자연어 분석 요청을 받아 
비즈니스 도메인 지식과 DB 스키마 구조를 활용하여 최적의 통계 분석 방법을 추천하고 
자동화된 전제조건 검증과 함께 통계 분석을 수행합니다.
"""

import argparse
import sys
import logging
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    setup_dependencies, validate_settings, 
    run_interactive_mode, print_welcome_message
)


def create_argument_parser():
    """명령행 인수 파서 생성 - 비즈니스 컨텍스트 인식 자연어 요청 기반 분석"""
    parser = argparse.ArgumentParser(
        description='🔬 Statistical Analysis Assistant - 비즈니스 컨텍스트 인식 AI 통계 분석 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🏢 Enhanced RAG 기반 워크플로우:
  1. 📁 input_data/data_files 폴더의 데이터 파일 목록 표시 → 사용자 선택
  2. 📊 데이터 로딩 완료 및 기본 정보 확인
  3. 🗣️ 사용자가 자연어로 분석 요청 입력
  4. 🔍 RAG 시스템 활성화:
     • 비즈니스 도메인 지식 검색 (business_dictionary, domain_knowledge)
     • DB 스키마 구조 검색 (schema_definitions, relationship_maps)
  5. 🧠 AI가 비즈니스 컨텍스트를 고려하여 최적 통계 기법 1~3개 추천
  6. 👤 사용자가 추천 방법 중 선택
  7. 🤖 자동화된 전제조건 검증 및 통계 분석 수행 (정규성, 등분산성 등)
  8. 📄 비즈니스 인사이트 포함 결과 보고서 생성

사용 예시:
  poetry run python main.py                    # 기본 대화형 모드
  poetry run python main.py --data sales.csv   # 특정 데이터로 시작
  poetry run python main.py --verbose          # 상세 로깅 포함  
  poetry run python main.py --no-welcome       # 환영 메시지 생략

Docker 실행:
  docker build -t statistical-assistant .
  docker run -it statistical-assistant

🗣️ 자연어 분석 요청 예시 (비즈니스 도메인별):

┌─────────────────────────────────────────────────────────────────┐
│ 📊 그룹 비교 분석                                               │
├─────────────────────────────────────────────────────────────────┤
│ • "지역별 매출 차이가 통계적으로 유의한지 확인하고 싶어요"       │
│ • "부서별 직원 만족도에 차이가 있나요?"                         │
│ • "브랜드별 고객 충성도 점수를 비교해주세요"                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 🔗 관계 및 상관관계 분석                                        │
├─────────────────────────────────────────────────────────────────┤
│ • "광고비와 매출 간의 상관관계를 분석해주세요"                  │
│ • "고객 만족도와 재구매율 사이의 관계를 알고 싶습니다"          │
│ • "근무시간과 생산성 지표 간 관련성을 확인해주세요"             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 📈 예측 및 회귀 분석                                            │
├─────────────────────────────────────────────────────────────────┤
│ • "여러 마케팅 요인들이 매출에 미치는 영향을 분석해주세요"      │
│ • "고객 특성을 바탕으로 구매 확률을 예측하고 싶어요"            │
│ • "제품 특징들이 가격에 미치는 영향도를 알아보세요"             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 🏢 비즈니스 의사결정 분석                                       │
├─────────────────────────────────────────────────────────────────┤
│ • "새로운 마케팅 전략의 효과를 검증하고 싶습니다"               │
│ • "A/B 테스트 결과가 통계적으로 유의한지 확인해주세요"          │
│ • "고객 세그먼트별 구매 패턴 차이를 분석해주세요"               │
└─────────────────────────────────────────────────────────────────┘

🏗️ 비즈니스 컨텍스트 메타데이터 활용:
  • input_data/metadata/business_dictionary.json - 업계 용어사전
  • input_data/metadata/domain_knowledge.md - 도메인 전문 지식
  • input_data/metadata/database_schemas/ - DB 스키마 구조 정보
        """
    )
    
    # 데이터 파일 옵션
    parser.add_argument(
        '--data', '--input-data',
        type=str,
        help='분석할 데이터 파일 경로 (input_data/data_files 폴더 내)'
    )
    
    # 시스템 옵션
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='상세한 로깅 출력 (디버깅용)'
    )
    
    parser.add_argument(
        '--no-welcome',
        action='store_true',
        help='환영 메시지 및 워크플로우 가이드 생략'
    )
    
    return parser


def print_startup_info():
    """시작 시 시스템 정보 출력 - Enhanced RAG 시스템 강조"""
    print("\n🔬========================================================🔬")
    print("   📊 Statistical Analysis Assistant 📊")
    print("    비즈니스 컨텍스트 인식 AI 통계 분석 시스템")
    print("🔬========================================================🔬")
    print()
    print("🧠 Enhanced RAG 시스템 특징:")
    print("  🏢 비즈니스 도메인 지식 검색")
    print("  🗄️ DB 스키마 구조 정보 활용")
    print("  🌏 BCEmbedding 기반 한중 이중언어 지원")
    print()
    print("📋 워크플로우:")
    print("  1. 📁 데이터 파일 선택 (input_data/data_files)")
    print("  2. 🗣️ 자연어 분석 요청 입력")
    print("  3. 🔍 RAG 시스템 활성화 (비즈니스 지식 + DB 스키마)")
    print("  4. 🧠 AI 컨텍스트 인식 분석 방법 추천 (1~3개)")
    print("  5. 🤖 자동화된 통계 분석 수행 (가정 검정 포함)")
    print("  6. 📄 비즈니스 인사이트 포함 결과 보고서 생성")
    print()


def validate_business_context_setup():
    """비즈니스 컨텍스트 및 RAG 시스템 환경 검증"""
    issues = []
    warnings = []
    
    # 기본 input_data 구조 확인
    input_data_path = Path("input_data")
    if not input_data_path.exists():
        issues.append("input_data 폴더가 존재하지 않습니다")
    
    # 데이터 파일 폴더 확인
    data_files_path = input_data_path / "data_files"
    if not data_files_path.exists():
        warnings.append("input_data/data_files 폴더가 없습니다 (분석할 데이터 파일 위치)")
    
    # 메타데이터 폴더 구조 확인
    metadata_path = input_data_path / "metadata"
    if not metadata_path.exists():
        warnings.append("input_data/metadata 폴더가 없습니다 (비즈니스 컨텍스트 정보)")
    else:
        # 비즈니스 지식 파일들 확인
        business_files = {
            "business_dictionary.json": "업계 용어사전",
            "domain_knowledge.md": "도메인 전문 지식",
            "analysis_guidelines.md": "분석 가이드라인"
        }
        
        for file_name, description in business_files.items():
            if not (metadata_path / file_name).exists():
                warnings.append(f"{file_name} 파일이 없습니다 ({description})")
        
        # DB 스키마 폴더 확인
        schema_path = metadata_path / "database_schemas"
        if not schema_path.exists():
            warnings.append("database_schemas 폴더가 없습니다 (DB 스키마 구조 정보)")
        else:
            schema_files = {
                "schema_definitions.json": "테이블 구조 정의",
                "relationship_maps.json": "테이블 관계 매핑", 
                "column_descriptions.json": "컬럼 상세 설명"
            }
            
            for file_name, description in schema_files.items():
                if not (schema_path / file_name).exists():
                    warnings.append(f"database_schemas/{file_name} 파일이 없습니다 ({description})")
    
    # RAG 인덱스 폴더 확인
    rag_index_path = Path("resources/rag_index")
    if not rag_index_path.exists():
        warnings.append("RAG 인덱스 폴더가 없습니다 (최초 실행 시 자동 생성됩니다)")
    
    return issues, warnings


def main():
    """메인 실행 함수 - 비즈니스 컨텍스트 인식 자연어 요청 기반 AI 추천 분석"""
    try:
        # 명령행 인수 파싱
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # 로깅 레벨 설정
        if args.verbose:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            logging.basicConfig(
                level=logging.WARNING,
                format='%(levelname)s - %(message)s'
            )
        
        # 환영 메시지 및 시작 정보 출력
        if not args.no_welcome:
            print_startup_info()
            print_welcome_message()
        
        # OpenAI API 환경 설정 검증
        print("🔧 OpenAI API 환경 설정 검증 중...")
        try:
            validate_settings()
            print("✅ OpenAI API 설정 완료!")
        except Exception as e:
            print(f"❌ 환경 설정 오류: {e}")
            print("📋 해결 방법:")
            print("1. .env 파일이 존재하는지 확인")
            print("2. 올바른 OPENAI_API_KEY가 설정되었는지 확인") 
            print("3. OPENAI_MODEL=gpt-4o로 설정되었는지 확인")
            print("4. poetry run python setup_project.py를 실행하여 초기 설정 수행")
            return 1
        
        # 비즈니스 컨텍스트 및 RAG 시스템 환경 검증
        print("🏢 비즈니스 컨텍스트 및 RAG 시스템 환경 검증 중...")
        issues, warnings = validate_business_context_setup()
        
        if issues:
            print("❌ 치명적인 문제 발견:")
            for issue in issues:
                print(f"   • {issue}")
            print("📋 poetry run python setup_project.py를 실행하여 프로젝트를 초기화해주세요.")
            return 1
        
        if warnings:
            print("⚠️ 주의사항 (선택적 기능):")
            for warning in warnings:
                print(f"   • {warning}")
            print("💡 완전한 비즈니스 컨텍스트 기능을 위해서는 setup_project.py 실행을 권장합니다.")
            print()
        else:
            print("✅ 비즈니스 컨텍스트 환경 설정 완료!")
        
        # 시스템 의존성 및 RAG 시스템 초기화
        print("🔨 시스템 컴포넌트 및 Enhanced RAG 시스템 초기화 중...")
        print("   📥 BCEmbedding 모델 로딩...")
        print("   🗄️ 비즈니스 지식베이스 인덱싱...")
        print("   🔗 DB 스키마 정보 매핑...")
        try:
            dependencies = setup_dependencies()
            print("✅ Enhanced RAG 시스템 초기화 완료!")
        except Exception as e:
            print(f"❌ 시스템 초기화 실패: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            print("💡 임베딩 모델 다운로드 중일 수 있습니다. 잠시 후 다시 시도해주세요.")
            return 1
        
        # 비즈니스 컨텍스트 인식 자연어 요청 기반 대화형 모드 실행
        print("🚀 비즈니스 컨텍스트 인식 분석 모드를 시작합니다...")
        print("🏢 업계 지식 및 DB 스키마 정보를 활용한 지능형 분석이 가능합니다.")
        print("=" * 70)
        run_interactive_mode(dependencies, args.data)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 사용자가 프로그램을 중단했습니다.")
        print("분석 세션이 안전하게 종료되었습니다.")
        return 0
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        if args.verbose if 'args' in locals() else False:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 