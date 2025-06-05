"""
Enhanced RAG 기반 Workflow Utilities

Multi-turn 대화형 통계 분석 워크플로우를 지원하는 유틸리티 함수들
Enhanced RAG 시스템과 AI 추천 엔진을 활용한 비즈니스 컨텍스트 인식 워크플로우
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from utils.ui_helpers import (
    print_welcome, print_usage_examples, print_analysis_guide,
    display_file_selection_menu, print_session_status
)
from utils.data_utils import get_available_data_files


def start_enhanced_rag_workflow(agent) -> Dict:
    """
    Enhanced RAG 기반 Multi-turn 분석 워크플로우 시작
    
    Args:
        agent: LLMAgent 인스턴스 (Enhanced RAG 시스템 포함)
        
    Returns:
        Dict: 워크플로우 시작 결과
    """
    logging.info("Enhanced RAG 기반 Multi-turn 워크플로우 시작")
    
    # 환영 메시지 및 시스템 소개
    print_welcome()
    print_enhanced_rag_features()
    print_usage_examples()
    
    # 세션 시작
    result = agent.start_session()
    
    if not result.get('session_started'):
        logging.error("세션 시작 실패")
        return {'error': '세션 시작에 실패했습니다.'}
    
    logging.info(f"워크플로우 시작 완료 - 세션 ID: {result.get('session_id')}")
    return result


def handle_data_selection_workflow(agent) -> Dict:
    """
    Enhanced RAG 워크플로우의 데이터 선택 단계 처리
    
    Args:
        agent: LLMAgent 인스턴스
        
    Returns:
        Dict: 데이터 선택 결과
    """
    try:
        # 사용 가능한 데이터 파일 검색
        data_files = get_available_data_files()
        
        if not data_files:
            print("\n❌ 분석할 데이터 파일이 없습니다.")
            print("📁 input_data/data_files/ 디렉토리에 데이터 파일을 추가해주세요.")
            print("📋 지원 형식: CSV, Excel, JSON, Parquet, TSV")
            return {'error': 'no_data_files'}
        
        # 파일 선택 메뉴 표시
        selected_file = display_file_selection_menu(data_files)
        
        if not selected_file:
            return {'cancelled': True}
        
        # Agent에 데이터 선택 전달
        result = agent.process_user_input(f"데이터 파일 선택: {selected_file}")
        
        return result
        
    except Exception as e:
        logging.error(f"데이터 선택 워크플로우 오류: {e}")
        return {'error': f'데이터 선택 중 오류: {str(e)}'}


def handle_natural_language_request_workflow(agent) -> Dict:
    """
    자연어 분석 요청 단계 처리
    
    Args:
        agent: LLMAgent 인스턴스
        
    Returns:
        Dict: 자연어 요청 처리 결과
    """
    try:
        print_analysis_guide()
        
        print("\n💬 분석하고 싶은 내용을 자연어로 말씀해주세요:")
        print("   예: '그룹별 평균 차이를 분석해주세요', '상관관계를 알고 싶어요'")
        
        user_request = input("\n📝 분석 요청: ").strip()
        
        if not user_request:
            return {'error': 'empty_request'}
        
        # 특수 명령어 처리
        if user_request.lower() in ['quit', 'exit', '종료']:
            return {'action': 'quit'}
        elif user_request.lower() in ['new', '새파일']:
            return {'action': 'new_file'}
        elif user_request.lower() in ['status', '상태']:
            return {'action': 'show_status'}
        
        # Agent에 자연어 요청 전달
        result = agent.process_user_input(user_request)
        
        return result
        
    except KeyboardInterrupt:
        print("\n\n👋 분석 요청을 취소합니다.")
        return {'cancelled': True}
    except Exception as e:
        logging.error(f"자연어 요청 워크플로우 오류: {e}")
        return {'error': f'자연어 요청 처리 중 오류: {str(e)}'}


def handle_rag_activation_workflow(agent) -> Dict:
    """
    Enhanced RAG 시스템 활성화 워크플로우 처리
    
    Args:
        agent: LLMAgent 인스턴스
        
    Returns:
        Dict: RAG 활성화 결과
    """
    try:
        print("\n🔍 Enhanced RAG 시스템을 활성화하고 있습니다...")
        print("   📊 비즈니스 도메인 지식 검색 중...")
        print("   🗄️ DB 스키마 구조 분석 중...")
        
        # Agent의 RAG 활성화 처리 (자동 진행)
        result = agent.process_user_input("rag_system_activate")
        
        if result.get('business_context') or result.get('schema_context'):
            print("   ✅ Enhanced RAG 시스템 활성화 완료!")
            
            # RAG 검색 결과 요약 표시
            _display_rag_search_summary(result)
        else:
            print("   ⚠️ RAG 시스템 활성화에 일부 문제가 있었습니다.")
        
        return result
        
    except Exception as e:
        logging.error(f"RAG 활성화 워크플로우 오류: {e}")
        return {'error': f'RAG 시스템 활성화 중 오류: {str(e)}'}


def handle_ai_recommendation_workflow(agent) -> Dict:
    """
    AI 추천 생성 및 사용자 선택 워크플로우 처리
    
    Args:
        agent: LLMAgent 인스턴스
        
    Returns:
        Dict: AI 추천 선택 결과
    """
    try:
        print("\n🤖 AI가 분석 방법을 추천하고 있습니다...")
        print("   🔍 비즈니스 컨텍스트 분석 중...")
        print("   📊 통계적 적합성 검토 중...")
        
        # Agent의 AI 추천 생성 (자동 진행)
        result = agent.process_user_input("generate_ai_recommendations")
        
        if result.get('recommendations'):
            print("   ✅ AI 추천 생성 완료!")
            
            # 추천 결과 표시는 Agent 내부에서 처리됨
            return result
        else:
            print("   ❌ AI 추천 생성에 실패했습니다.")
            return {'error': 'ai_recommendation_failed'}
        
    except Exception as e:
        logging.error(f"AI 추천 워크플로우 오류: {e}")
        return {'error': f'AI 추천 생성 중 오류: {str(e)}'}


def handle_method_confirmation_workflow(agent, user_choice: str) -> Dict:
    """
    분석 방법 확인 및 실행 워크플로우 처리
    
    Args:
        agent: LLMAgent 인스턴스
        user_choice: 사용자 선택
        
    Returns:
        Dict: 방법 확인 결과
    """
    try:
        # Agent에 사용자 선택 전달
        result = agent.process_user_input(user_choice)
        
        if result.get('analysis_started'):
            print("\n⚡ 선택된 분석 방법을 실행하고 있습니다...")
            print("   📊 데이터 전처리 중...")
            print("   🔬 통계 분석 수행 중...")
            print("   📋 결과 정리 중...")
        
        return result
        
    except Exception as e:
        logging.error(f"방법 확인 워크플로우 오류: {e}")
        return {'error': f'분석 방법 확인 중 오류: {str(e)}'}


def handle_session_continuation_workflow(agent) -> Dict:
    """
    세션 지속 여부 확인 워크플로우 처리
    
    Args:
        agent: LLMAgent 인스턴스
        
    Returns:
        Dict: 세션 지속 결과
    """
    try:
        print("\n🔄 추가 분석을 수행하시겠습니까?")
        print("   1️⃣ 새로운 분석 요청")
        print("   2️⃣ 다른 데이터 파일 선택")
        print("   3️⃣ 현재 세션 상태 확인")
        print("   0️⃣ 분석 종료")
        
        choice = input("\n📝 선택하세요 (0-3): ").strip()
        
        if choice == '0':
            return {'action': 'quit'}
        elif choice == '1':
            return {'action': 'new_analysis'}
        elif choice == '2':
            return {'action': 'new_file'}
        elif choice == '3':
            return {'action': 'show_status'}
        else:
            print("❌ 잘못된 선택입니다. 다시 시도해주세요.")
            return handle_session_continuation_workflow(agent)
        
    except KeyboardInterrupt:
        print("\n\n👋 세션을 종료합니다.")
        return {'action': 'quit'}
    except Exception as e:
        logging.error(f"세션 지속 워크플로우 오류: {e}")
        return {'error': f'세션 지속 확인 중 오류: {str(e)}'}


def print_enhanced_rag_features():
    """Enhanced RAG 시스템 특징 소개"""
    print("🔍 Enhanced RAG 시스템 특징:")
    print("   🏢 비즈니스 도메인 지식 활용")
    print("   🗄️ DB 스키마 구조 인식")
    print("   🤖 컨텍스트 기반 AI 추천")
    print("   📊 실무 중심 통계 분석")
    print()


def _display_rag_search_summary(result: Dict):
    """RAG 검색 결과 요약 표시"""
    print("\n📋 RAG 검색 결과 요약:")
    
    business_context = result.get('business_context', {})
    schema_context = result.get('schema_context', {})
    
    if business_context:
        print(f"   🏢 비즈니스 컨텍스트: {len(business_context)}개 항목")
        if 'domain_knowledge' in business_context:
            print("      • 도메인 전문 지식 ✅")
        if 'terminology' in business_context:
            print("      • 업계 용어사전 ✅")
        if 'analysis_guidelines' in business_context:
            print("      • 분석 가이드라인 ✅")
    
    if schema_context:
        print(f"   🗄️ 스키마 컨텍스트: {len(schema_context)}개 항목")
        if 'table_definitions' in schema_context:
            print("      • 테이블 구조 정의 ✅")
        if 'relationships' in schema_context:
            print("      • 테이블 관계 매핑 ✅")
        if 'constraints' in schema_context:
            print("      • 제약조건 정보 ✅")
    
    print()


def run_enhanced_multiturn_workflow(agent) -> None:
    """
    Enhanced RAG 기반 Multi-turn 워크플로우 전체 실행
    
    Args:
        agent: LLMAgent 인스턴스 (Enhanced RAG 시스템 포함)
    """
    try:
        # 1. 워크플로우 시작
        start_result = start_enhanced_rag_workflow(agent)
        if 'error' in start_result:
            print(f"❌ 워크플로우 시작 실패: {start_result['error']}")
            return
        
        # 2. 메인 루프 - Multi-turn 대화
        session_count = 0
        
        while True:
            session_count += 1
            current_node = agent.current_node_id
            
            # 현재 노드에 따른 워크플로우 처리
            if current_node == 'data_selection':
                result = handle_data_selection_workflow(agent)
            elif current_node == 'natural_language_request':
                result = handle_natural_language_request_workflow(agent)
            elif current_node == 'rag_system_activation':
                result = handle_rag_activation_workflow(agent)
            elif current_node == 'ai_recommendation_generation':
                result = handle_ai_recommendation_workflow(agent)
            elif current_node == 'method_confirmation':
                user_choice = input("\n📝 선택하신 분석 방법을 확인해주세요: ").strip()
                result = handle_method_confirmation_workflow(agent, user_choice)
            elif current_node == 'session_continuation':
                result = handle_session_continuation_workflow(agent)
            else:
                # 기본 사용자 입력 처리
                user_input = input("\n📝 입력하세요: ").strip()
                result = agent.process_user_input(user_input)
            
            # 결과 처리
            if 'error' in result:
                print(f"❌ 오류: {result['error']}")
                continue
            elif result.get('action') == 'quit':
                break
            elif result.get('action') == 'new_file':
                agent.current_node_id = 'data_selection'
                continue
            elif result.get('action') == 'show_status':
                _show_session_status(agent, session_count)
                continue
            
            # 자동 진행 확인
            if result.get('auto_proceed'):
                continue
            
            # 워크플로우 완료 확인
            if result.get('workflow_completed'):
                if not _ask_continue_analysis():
                    break
                else:
                    agent.current_node_id = 'natural_language_request'
    
    except KeyboardInterrupt:
        print("\n\n👋 사용자가 워크플로우를 중단했습니다.")
    except Exception as e:
        logging.error(f"Enhanced Multi-turn 워크플로우 오류: {e}")
        print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")
    finally:
        # 세션 정리
        try:
            session_summary = agent.context_manager.get_summary()
            print(f"\n📊 총 {session_count}번의 분석을 수행했습니다.")
            print("👋 Enhanced RAG 기반 통계 분석 어시스턴트를 이용해주셔서 감사합니다!")
        except:
            print("\n👋 세션이 종료되었습니다.")


def _show_session_status(agent, session_count: int):
    """현재 세션 상태 표시"""
    try:
        data_path = getattr(agent, 'current_data_path', 'N/A')
        context_items = len(agent.context_manager._interaction_history)
        
        print_session_status(data_path, session_count, context_items)
        
        # RAG 컨텍스트 상태 표시
        rag_summary = agent.context_manager.get_rag_context_summary()
        print(f"  • 비즈니스 컨텍스트: {len(rag_summary.get('business_context_keys', []))}개")
        print(f"  • 스키마 컨텍스트: {len(rag_summary.get('schema_context_keys', []))}개")
        print(f"  • RAG 검색 횟수: {rag_summary.get('rag_searches_count', 0)}회")
        
    except Exception as e:
        print(f"❌ 세션 상태 표시 오류: {e}")


def _ask_continue_analysis() -> bool:
    """추가 분석 진행 여부 확인"""
    while True:
        try:
            choice = input("\n🔄 추가 분석을 진행하시겠습니까? (y/n): ").strip().lower()
            if choice in ['y', 'yes', '예', 'ㅇ']:
                return True
            elif choice in ['n', 'no', '아니오', 'ㄴ']:
                return False
            else:
                print("❌ 'y' 또는 'n'으로 답해주세요.")
        except KeyboardInterrupt:
            print("\n")
            return False


# main.py 호환성을 위한 wrapper 함수
def run_interactive_mode(dependencies: Dict, initial_data_file: Optional[str] = None) -> None:
    """
    main.py에서 호출하는 대화형 모드 실행 함수 (호환성 래퍼)
    
    Args:
        dependencies: setup_dependencies()에서 반환된 의존성 딕셔너리
        initial_data_file: 초기 로딩할 데이터 파일 경로 (선택사항)
    """
    from utils.system_setup import create_agent_instance
    
    try:
        # Agent 인스턴스 생성
        agent = create_agent_instance(dependencies)
        
        # 초기 데이터 파일이 지정된 경우 미리 설정
        if initial_data_file:
            print(f"📁 지정된 데이터 파일로 시작: {initial_data_file}")
            agent.current_data_path = initial_data_file
            # 데이터 선택 노드를 건너뛰고 바로 자연어 요청으로 진행
            agent.current_node_id = 'natural_language_request'
        else:
            # 기본값: 데이터 선택부터 시작
            agent.current_node_id = 'data_selection'
        
        # Enhanced Multi-turn 워크플로우 실행
        run_enhanced_multiturn_workflow(agent)
        
    except Exception as e:
        logging.error(f"Interactive 모드 실행 오류: {e}")
        print(f"❌ 대화형 모드 실행 중 오류가 발생했습니다: {e}") 