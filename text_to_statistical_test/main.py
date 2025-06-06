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

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings, ensure_directories
from config.logging_config import init_default_logging, get_logger
from core.workflow.orchestrator import Orchestrator
from core.workflow.state_manager import StateManager

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
        '--config',
        type=str,
        help='설정 파일 경로 (JSON)'
    )
    
    # 고급 옵션
    parser.add_argument(
        '--skip-stages',
        type=str,
        help='건너뛸 단계 번호 (쉼표로 구분, 예: 2,4)'
    )
    
    parser.add_argument(
        '--resume-session',
        type=str,
        help='이전 세션 ID로 재시작'
    )
    
    parser.add_argument(
        '--export-format',
        choices=['html', 'pdf', 'markdown', 'json'],
        default='html',
        help='보고서 출력 형식'
    )
    
    return parser.parse_args()

def setup_environment(args: argparse.Namespace) -> Dict[str, Any]:
    """환경 설정 및 초기화"""
    
    # 디렉토리 생성 확인
    ensure_directories()
    
    # 로깅 초기화
    if args.debug:
        import os
        os.environ['DEBUG'] = 'true'
        os.environ['LOG_LEVEL'] = 'DEBUG'
    
    init_default_logging()
    logger = get_logger(__name__)
    
    # 설정 로드
    settings = get_settings()
    
    if args.output_dir:
        settings['paths'].output_data_dir = Path(args.output_dir)
        ensure_directories()
    
    logger.info("Text-to-Statistical-Test 시스템 시작")
    logger.info(f"설정 로드 완료: {settings['application']}")
    
    return settings

async def run_workflow(args: argparse.Namespace, settings: Dict[str, Any]) -> bool:
    """워크플로우 실행"""
    logger = get_logger(__name__)
    
    try:
        # State Manager 초기화
        state_manager = StateManager()
        
        # 이전 세션 복원 (옵션)
        if args.resume_session:
            if not state_manager.load_session(args.resume_session):
                logger.warning(f"세션 {args.resume_session}을 찾을 수 없습니다. 새 세션을 시작합니다.")
        
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
            initial_context['file_path'] = args.file
        
        # 건너뛸 단계 설정
        if args.skip_stages:
            skip_list = [int(s.strip()) for s in args.skip_stages.split(',')]
            initial_context['skip_stages'] = skip_list
        
        # 워크플로우 실행
        logger.info(f"{args.stage}단계부터 워크플로우 시작")
        result = await orchestrator.execute_pipeline(
            start_stage=args.stage,
            initial_context=initial_context
        )
        
        if result.get('success', False):
            logger.info("✅ 워크플로우 실행 완료")
            
            # 결과 요약 출력
            if 'final_report' in result:
                print("\n" + "="*60)
                print("📊 분석 결과 요약")
                print("="*60)
                print(result['final_report'].get('summary', '요약 정보 없음'))
                
            # 출력 파일 정보
            if 'output_files' in result:
                print(f"\n📁 결과 파일들:")
                for file_path in result['output_files']:
                    print(f"  - {file_path}")
            
            return True
        else:
            logger.error("❌ 워크플로우 실행 실패")
            if 'error_message' in result:
                print(f"오류: {result['error_message']}")
            return False
            
    except KeyboardInterrupt:
        logger.info("사용자에 의해 실행이 중단되었습니다.")
        return False
    except Exception as e:
        logger.error(f"워크플로우 실행 중 예외 발생: {e}", exc_info=True)
        print(f"❌ 시스템 오류: {e}")
        return False

def print_welcome_message():
    """환영 메시지 출력"""
    print("\n" + "="*70)
    print("🤖 Text-to-Statistical-Test")
    print("="*70)
    print("RAG 기반 Agentic AI 통계 분석 시스템에 오신 것을 환영합니다!")
    print("")
    print("🎯 이 시스템은 다음 8단계로 진행됩니다:")
    print("  1️⃣  데이터 파일 선택 및 초기 이해")
    print("  2️⃣  사용자 자연어 요청 및 목표 정의")
    print("  3️⃣  데이터 심층 분석 및 요약")
    print("  4️⃣  Agentic LLM의 분석 전략 제안")
    print("  5️⃣  사용자 피드백 기반 분석 방식 구체화")
    print("  6️⃣  RAG 기반 Agentic LLM의 데이터 분석 계획 수립")
    print("  7️⃣  Agentic LLM의 자율적 통계 검정")
    print("  8️⃣  Agentic LLM의 보고서 생성 및 해석")
    print("")
    print("💡 도움이 필요하시면 언제든 'help' 또는 '도움말'을 입력하세요.")
    print("="*70 + "\n")

async def main():
    """메인 함수"""
    try:
        # 명령행 인자 파싱
        args = parse_arguments()
        
        # 대화형 모드인 경우 환영 메시지 출력
        if not args.non_interactive:
            print_welcome_message()
        
        # 환경 설정
        settings = setup_environment(args)
        
        # 워크플로우 실행
        success = await run_workflow(args, settings)
        
        # 종료 코드 반환
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ 시스템 초기화 오류: {e}")
        return 1

if __name__ == "__main__":
    # Python 3.7+ 호환성
    if sys.version_info >= (3, 7):
        exit_code = asyncio.run(main())
    else:
        loop = asyncio.get_event_loop()
        exit_code = loop.run_until_complete(main())
        loop.close()
    
    sys.exit(exit_code) 