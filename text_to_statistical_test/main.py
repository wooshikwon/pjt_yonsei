#!/usr/bin/env python3
"""
Text-to-Statistical-Test Main Entry Point
`Orchestrator-Engine` 모델에 따라 워크플로우를 실행합니다.
"""

import sys
import asyncio
import argparse
from pathlib import Path
import logging

# .env 파일 로드 (다른 모듈보다 먼저)
from dotenv import load_dotenv
load_dotenv()

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

# --- 서비스 및 모듈 임포트 ---
try:
    from config.logging_config import setup_logging
    from config.settings import get_settings
    from core.workflow.orchestrator import Orchestrator
    # 서비스 모듈을 임포트하여 모든 서비스가 초기화되도록 합니다.
    import services
except ImportError as e:
    print(f"CRITICAL: Essential module failed to import: {e}", file=sys.stderr)
    print("Please ensure all dependencies are installed correctly (`poetry install`).", file=sys.stderr)
    sys.exit(1)

# 로깅 설정 초기화
setup_logging()
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Text-to-Statistical-Test: RAG 기반 Agentic AI 통계 분석 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  poetry run python main.py --file "input_data/data_files/sample_customers.csv" --request "고객 지원 사용 여부에 따라 월 지출액에 차이가 있는지 t-검정으로 분석해줘."
"""
    )
    parser.add_argument(
        '--file', '-f', type=str, required=True,
        help='분석할 데이터 파일 경로'
    )
    parser.add_argument(
        '--request', '-r', type=str, required=True,
        help='사용자의 분석 요청'
    )
    return parser.parse_args()


async def main():
    """메인 실행 함수"""
    args = parse_arguments()
    settings = get_settings()

    # 필수 환경 변수 확인 (예: API 키)
    if not settings.llm.openai_api_key:
        logger.critical("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.", file=sys.stderr)
        sys.exit(1)

    file_path = Path(args.file)
    user_request = args.request

    if not file_path.exists():
        logger.critical(f"데이터 파일을 찾을 수 없습니다: {file_path}")
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {file_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("      🔬 Text-to-Statistical-Test System 🔬")
    print(f"      - 데이터: {file_path.name}")
    print("=" * 70)
    logger.info(f"분석 시작: file='{file_path}', request='{user_request}'")
    
    try:
        orchestrator = Orchestrator()
        
        final_context = await orchestrator.run(
            file_path=str(file_path),
            user_request=user_request
        )

        final_report_content = final_context.get("final_report_content")
        final_report_path = final_context.get("final_report_path")

        if final_report_content and final_report_path:
            logger.info(f"✅ 워크플로우 실행 완료. 최종 보고서: {final_report_path}")
            
            print("\n" + "="*70)
            print(" " * 25 + "📊 분석 결과 📊")
            print("="*70)
            print(final_report_content)
            print("="*70)
            print(f"\n📂 이 결과는 다음 파일에도 저장되었습니다:\n  {final_report_path}")
            print("="*70)
            sys.exit(0)
        else:
            logger.error("❌ 워크플로우 실행 실패. 최종 보고서 콘텐츠가 생성되지 않았습니다.")
            print("\n" + "="*60)
            print("❌ 분석에 실패했습니다. 상세 내용은 로그 파일을 참고해주세요.")
            print("="*60)
            sys.exit(1)

    except Exception as e:
        logger.critical(f"워크플로우 실행 중 예상치 못한 예외 발생: {e}", exc_info=True)
        print(f"\n❌ 시스템 오류가 발생했습니다. 자세한 내용은 로그를 확인해주세요.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 