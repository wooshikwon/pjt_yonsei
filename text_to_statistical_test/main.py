#!/usr/bin/env python3
"""
Text-to-Statistical-Test Main Entry Point

RAG 기반 Agentic AI 통계 분석 시스템의 메인 진입점
CLI 인터페이스를 통해 8단계 워크플로우를 실행합니다.
"""

import sys
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

# 중앙 집중식 설정 로드 (여기서 .env 파일이 자동으로 로드됨)
from config.settings import get_settings

def parse_arguments() -> argparse.Namespace:
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Text-to-Statistical-Test: RAG 기반 Agentic AI 통계 분석 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py                           # 대화형 모드로 전체 워크플로우 실행
  python main.py --file data.csv          # 특정 파일로 시작
  python main.py --stage 3                # 3단계부터 시작
  python main.py --debug                  # 디버그 모드
  python main.py --non-interactive        # 비대화형 모드
        """
    )
    
    # 기본 옵션
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='분석할 데이터 파일 경로'
    )
    
    parser.add_argument(
        '--stage', '-s',
        type=int,
        choices=range(1, 9),
        default=1,
        help='시작할 워크플로우 단계 (1-8)'
    )
    
    parser.add_argument(
        '--non-interactive', '-n',
        action='store_true',
        help='비대화형 모드 (사용자 입력 최소화)'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='디버그 모드 활성화'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='결과 출력 디렉토리 경로'
    )
    
    parser.add_argument(
        '--skip-stages',
        type=str,
        help='건너뛸 단계 번호 (쉼표로 구분, 예: 2,4)'
    )
    
    parser.add_argument(
        '--export-format',
        choices=['html', 'pdf', 'markdown', 'json'],
        default='html',
        help='보고서 출력 형식'
    )
    
    return parser.parse_args()

def setup_environment(args: argparse.Namespace) -> None:
    """환경 설정 및 초기화"""
    
    # 필수 디렉토리 생성
    directories = [
        'input_data/data_files',
        'input_data/metadata',
        'output_data/reports',
        'output_data/visualizations',
        'output_data/analysis_cache',
        'logs'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # 중앙화된 로깅 설정 사용
    from config.logging_config import setup_logging
    
    # 환경 변수 설정
    if args.debug or os.getenv('DEBUG', 'false').lower() == 'true':
        os.environ['DEBUG'] = 'true'
        os.environ['LOG_LEVEL'] = 'DEBUG'
    
    # 비대화형 모드 설정
    if args.non_interactive:
        os.environ['NON_INTERACTIVE'] = 'true'
    
    # 로깅 레벨 결정
    log_level = 'DEBUG' if args.debug else os.getenv('LOG_LEVEL', 'INFO')
    
    # 로깅 설정 적용
    setup_logging(
        log_level=log_level,
        console_output=True,
        structured_logging=True
    )

async def run_workflow(args: argparse.Namespace) -> bool:
    """워크플로우 실행"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # 필요한 모듈들을 지연 import (의존성 오류 방지)
        from core.workflow.orchestrator import Orchestrator
        from core.workflow.state_manager import StateManager
        
        # State Manager 초기화
        state_manager = StateManager()
        
        # Orchestrator 초기화
        orchestrator = Orchestrator(state_manager=state_manager)
        
        # 초기 컨텍스트 설정
        initial_context = {
            'interactive': not args.non_interactive,
            'debug': args.debug,
            'export_format': args.export_format,
            'start_stage': args.stage
        }
        
        # 파일이 지정된 경우
        if args.file:
            if not os.path.exists(args.file):
                logger.error(f"파일을 찾을 수 없습니다: {args.file}")
                return False
            initial_context['file_path'] = args.file
        
        # 건너뛸 단계 설정
        if args.skip_stages:
            skip_list = [int(s.strip()) for s in args.skip_stages.split(',')]
            initial_context['skip_stages'] = skip_list
        
        # 워크플로우 실행
        logger.info(f"워크플로우 시작: {args.stage}단계부터")
        result = await orchestrator.execute_pipeline(
            start_stage=args.stage,
            initial_context=initial_context
        )
        
        if result.get('success', False):
            logger.info("✅ 워크플로우 실행 완료")
            
            # 결과 요약 출력
            print("\n" + "="*60)
            print("📊 분석 완료!")
            print("="*60)
            
            if 'comprehensive_report' in result:
                report = result['comprehensive_report']
                print(f"\n📋 보고서 제목: {report.get('report_metadata', {}).get('title', 'N/A')}")
                print(f"🎯 분석 방법: {report.get('report_metadata', {}).get('analysis_method', 'N/A')}")
            
            # 출력 파일 정보
            if 'save_result' in result and result['save_result'].get('success'):
                print(f"\n📁 결과 파일:")
                for file_path in result['save_result'].get('files_generated', []):
                    print(f"  - {file_path}")
            
            return True
        else:
            logger.error("❌ 워크플로우 실행 실패")
            if 'error' in result:
                print(f"오류: {result['error']}")
            return False
            
    except KeyboardInterrupt:
        logger.info("사용자에 의해 실행이 중단되었습니다.")
        return False
    except ImportError as e:
        logger.error(f"모듈 import 오류: {e}")
        print(f"❌ 의존성 오류: {e}")
        print("다음 명령으로 의존성을 설치하세요: poetry install")
        return False
    except Exception as e:
        logger.error(f"워크플로우 실행 중 예외 발생: {e}", exc_info=True)
        print(f"❌ 시스템 오류: {e}")
        return False

def print_welcome_message():
    """환영 메시지 출력"""
    print("\n" + "="*70)
    print("🤖 Text-to-Statistical-Test(TTST)")
    print("   RAG 기반 Agentic AI 통계 분석 시스템")
    print("="*70)
    print()

def check_prerequisites() -> bool:
    """필수 조건 확인"""
    import logging
    logger = logging.getLogger(__name__)
    
    # Python 버전 확인
    if sys.version_info < (3, 11):
        logger.error("Python 3.11 이상이 필요합니다.")
        return False
    
    # 설정 로드 상태 확인
    try:
        settings = get_settings()
        llm_settings = settings['llm']
        
        # .env 파일 로드 상태 메시지
        env_file = Path(__file__).parent / '.env'
        if env_file.exists():
            print(f"✅ 환경변수 파일 로드됨: {env_file}")
        else:
            print("⚠️  .env 파일을 찾을 수 없습니다. env.example을 참고하여 .env 파일을 생성하세요.")
        
    except Exception as e:
        logger.error(f"설정 로드 실패: {e}")
        return False
    
    # 필수 API 키 확인
    missing_vars = []
    
    if not llm_settings.openai_api_key:
        missing_vars.append('OPENAI_API_KEY')
    
    if missing_vars:
        logger.error(f"❌ 다음 필수 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
        logger.error("해결 방법:")
        logger.error("  1. env.example을 .env로 복사: cp env.example .env")
        logger.error("  2. .env 파일에서 API 키를 실제 값으로 변경")
        logger.error("  3. 또는 환경변수로 직접 설정: export OPENAI_API_KEY=your_key")
        return False
    
    # 선택적 API 키 확인
    if not llm_settings.anthropic_api_key:
        logger.info("선택적 환경 변수 미설정: ANTHROPIC_API_KEY")
    
    logger.info("✅ 환경변수 확인 완료")
    return True

async def main():
    """메인 함수"""
    # 인자 파싱
    args = parse_arguments()
    
    # 환경 설정
    setup_environment(args)
    
    # 환영 메시지
    if not args.non_interactive:
        print_welcome_message()
    
    # 필수 조건 확인
    if not check_prerequisites():
        sys.exit(1)
    
    # 워크플로우 실행
    success = await run_workflow(args)
    
    # 종료 코드 설정
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류가 발생했습니다: {e}")
        sys.exit(1) 