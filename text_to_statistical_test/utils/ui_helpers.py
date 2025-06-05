"""
UI Helper Functions

사용자 인터페이스 관련 헬퍼 함수들
- 환영 메시지, 사용 예시, 파일 선택 메뉴 등
"""

from typing import List, Dict, Any, Optional
from pathlib import Path


def print_welcome_message():
    """메인 진입점을 위한 환영 메시지"""
    print()
    print("🎯 Enhanced RAG 기반 비즈니스 컨텍스트 인식 AI 통계 분석 시스템")
    print("   자연어 요청 → AI 추천 → 자동화된 통계 분석 → 비즈니스 인사이트")
    print()


def print_welcome():
    """워크플로우 시작 시 상세 환영 메시지"""
    print("🔬" + "="*70 + "🔬")
    print("   📊 Enhanced RAG 기반 Statistical Analysis Assistant 📊")
    print("     비즈니스 컨텍스트 인식 AI 통계 분석 시스템")
    print("🔬" + "="*70 + "🔬")
    print()
    
    print("🧠 Enhanced RAG 시스템 특징:")
    print("   🏢 비즈니스 도메인 지식 검색 (업계 용어사전, 분석 가이드라인)")
    print("   🗄️ DB 스키마 구조 인식 (테이블 관계, 제약조건 분석)")
    print("   🌏 BCEmbedding 기반 한중 이중언어 지원")
    print("   🤖 컨텍스트 기반 AI 분석 방법 추천")
    print()


def print_enhanced_rag_features():
    """Enhanced RAG 시스템 특징 소개"""
    print("🔍 Enhanced RAG 시스템 특징:")
    print("   🏢 비즈니스 도메인 지식 활용")
    print("   🗄️ DB 스키마 구조 인식")
    print("   🤖 컨텍스트 기반 AI 추천")
    print("   📊 실무 중심 통계 분석")
    print()


def print_usage_examples():
    """자연어 분석 요청 예시 표시"""
    print("🗣️ 자연어 분석 요청 예시:")
    print()
    
    examples = [
        {
            'category': '📊 그룹 비교 분석',
            'examples': [
                "지역별 매출 차이가 통계적으로 유의한지 확인하고 싶어요",
                "부서별 직원 만족도에 차이가 있나요?",
                "브랜드별 고객 충성도 점수를 비교해주세요"
            ]
        },
        {
            'category': '🔗 관계 및 상관관계 분석',
            'examples': [
                "광고비와 매출 간의 상관관계를 분석해주세요",
                "고객 만족도와 재구매율 사이의 관계를 알고 싶습니다",
                "근무시간과 생산성 지표 간 관련성을 확인해주세요"
            ]
        },
        {
            'category': '📈 예측 및 회귀 분석',
            'examples': [
                "여러 마케팅 요인들이 매출에 미치는 영향을 분석해주세요",
                "고객 특성을 바탕으로 구매 확률을 예측하고 싶어요",
                "제품 특징들이 가격에 미치는 영향도를 알아보세요"
            ]
        },
        {
            'category': '🏢 비즈니스 의사결정 분석',
            'examples': [
                "새로운 마케팅 전략의 효과를 검증하고 싶습니다",
                "A/B 테스트 결과가 통계적으로 유의한지 확인해주세요",
                "고객 세그먼트별 구매 패턴 차이를 분석해주세요"
            ]
        }
    ]
    
    for example_group in examples:
        print(f"   {example_group['category']}")
        for example in example_group['examples']:
            print(f"     • {example}")
        print()


def print_analysis_guide():
    """분석 요청 가이드 출력"""
    print("\n💡 분석 요청 작성 가이드:")
    print("   ✅ 구체적인 변수명 언급: '지역별', '부서별', '시간대별' 등")
    print("   ✅ 분석 목적 명시: '차이 확인', '관계 분석', '예측' 등")
    print("   ✅ 비즈니스 맥락 포함: '매출 증대', '고객 만족도', '효율성' 등")
    print("   ✅ 자연스러운 한국어 사용")
    print()
    print("   🚫 피해야 할 표현:")
    print("     • 너무 짧은 요청: '분석해주세요', '확인하고 싶어요'")
    print("     • 기술적 용어만 사용: 't-test', 'ANOVA' 등")
    print("     • 모호한 표현: '이것저것', '그런거' 등")


def display_file_selection_menu(data_files: List[str]) -> Optional[str]:
    """
    데이터 파일 선택 메뉴 표시
    
    Args:
        data_files: 선택 가능한 데이터 파일 리스트
        
    Returns:
        str or None: 선택된 파일 경로 또는 None (취소 시)
    """
    if not data_files:
        print("\n❌ 분석할 데이터 파일이 없습니다.")
        return None
    
    print("\n📁 사용 가능한 데이터 파일:")
    print("=" * 50)
    
    for i, file_path in enumerate(data_files, 1):
        file_name = Path(file_path).name
        file_size = _get_file_size_info(file_path)
        print(f"   {i:2d}. {file_name} {file_size}")
    
    print("=" * 50)
    print("   0. 취소")
    print()
    
    while True:
        try:
            choice = input("📝 분석할 파일 번호를 선택하세요: ").strip()
            
            if choice == '0':
                return None
            
            file_index = int(choice) - 1
            if 0 <= file_index < len(data_files):
                selected_file = data_files[file_index]
                print(f"✅ 선택된 파일: {Path(selected_file).name}")
                return selected_file
            else:
                print("❌ 잘못된 번호입니다. 다시 선택해주세요.")
                
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
        except KeyboardInterrupt:
            print("\n\n👋 파일 선택을 취소합니다.")
            return None


def print_session_status(data_path: str, session_count: int, context_items: int):
    """현재 세션 상태 표시"""
    print(f"\n📊 현재 세션 상태:")
    print(f"  • 현재 데이터: {Path(data_path).name if data_path != 'N/A' else 'N/A'}")
    print(f"  • 분석 세션 수: {session_count}")
    print(f"  • 컨텍스트 항목: {context_items}개")


def display_rag_search_results(business_context: Dict, schema_context: Dict):
    """RAG 검색 결과 표시"""
    print("\n🔍 Enhanced RAG 검색 결과:")
    print("=" * 50)
    
    # 비즈니스 컨텍스트 표시
    if business_context:
        detected_domain = business_context.get('detected_domain')
        if detected_domain:
            print(f"🏢 감지된 비즈니스 도메인: {detected_domain}")
        
        terminology = business_context.get('terminology', [])
        if terminology:
            print(f"📚 관련 비즈니스 용어: {len(terminology)}개")
            for term in terminology[:3]:  # 상위 3개만 표시
                term_name = term.get('term', 'Unknown')
                relevance = term.get('relevance_score', 0)
                print(f"   • {term_name} (관련도: {relevance:.2f})")
        
        key_insights = business_context.get('key_insights', [])
        if key_insights:
            print(f"💡 핵심 인사이트:")
            for insight in key_insights[:2]:  # 상위 2개만 표시
                print(f"   • {insight}")
    
    # 스키마 컨텍스트 표시
    if schema_context:
        matched_tables = schema_context.get('matched_tables', {})
        if matched_tables:
            print(f"🗄️ 매칭된 테이블: {len(matched_tables)}개")
            for table_name, columns in matched_tables.items():
                print(f"   • {table_name}: {', '.join(columns)}")
        
        suggestions = schema_context.get('suggestions', [])
        if suggestions:
            print(f"📋 스키마 기반 제안:")
            for suggestion in suggestions[:2]:  # 상위 2개만 표시
                print(f"   • {suggestion}")
    
    print("=" * 50)


def display_ai_recommendations(recommendations: List[Dict]):
    """AI 추천 결과 표시"""
    if not recommendations:
        print("\n❌ AI 추천을 생성할 수 없습니다.")
        return
    
    print("\n🤖 AI 분석 방법 추천:")
    print("=" * 60)
    
    for i, rec in enumerate(recommendations, 1):
        method_name = rec.get('method_name', f'방법 {i}')
        confidence = rec.get('confidence', 0)
        reasoning = rec.get('reasoning', '추천 근거 없음')
        
        print(f"{i}. {method_name} (추천도: {confidence:.0f}%)")
        print(f"   📋 추천 근거: {reasoning}")
        
        # 비즈니스 컨텍스트 고려사항
        business_considerations = rec.get('business_considerations', [])
        if business_considerations:
            print(f"   🏢 비즈니스 고려사항:")
            for consideration in business_considerations[:2]:
                print(f"      • {consideration}")
        
        # 스키마 고려사항
        schema_considerations = rec.get('schema_considerations', [])
        if schema_considerations:
            print(f"   🗄️ 스키마 고려사항:")
            for consideration in schema_considerations[:2]:
                print(f"      • {consideration}")
        
        print()
    
    print("=" * 60)


def display_analysis_progress(stage: str, details: str = ""):
    """분석 진행 상황 표시"""
    stage_icons = {
        'data_loading': '📥',
        'preprocessing': '🔧',
        'assumption_testing': '🔬',
        'analysis': '📊',
        'interpretation': '💡',
        'reporting': '📋'
    }
    
    icon = stage_icons.get(stage, '⚡')
    print(f"{icon} {details}")


def print_workflow_completion_message():
    """워크플로우 완료 메시지"""
    print("\n🎉 분석이 완료되었습니다!")
    print("📊 결과 보고서가 생성되었습니다.")
    print("💡 비즈니스 인사이트를 확인해보세요.")


def _get_file_size_info(file_path: str) -> str:
    """파일 크기 정보 반환"""
    try:
        from utils.data_utils import get_file_info
        file_info = get_file_info(file_path)
        return f"({file_info.get('size_formatted', 'N/A')})"
    except:
        return "(크기 정보 없음)"


def print_error_message(error_type: str, details: str = ""):
    """에러 메시지 출력"""
    error_messages = {
        'no_data': "❌ 분석할 데이터가 없습니다.",
        'invalid_file': "❌ 올바르지 않은 파일 형식입니다.",
        'rag_error': "❌ RAG 시스템 오류가 발생했습니다.",
        'ai_error': "❌ AI 추천 생성 중 오류가 발생했습니다.",
        'analysis_error': "❌ 통계 분석 중 오류가 발생했습니다."
    }
    
    base_message = error_messages.get(error_type, "❌ 알 수 없는 오류가 발생했습니다.")
    
    if details:
        print(f"{base_message}\n💡 상세 정보: {details}")
    else:
        print(base_message)


def print_help_message():
    """도움말 메시지 출력"""
    print("\n📋 사용 가능한 명령어:")
    print("   • 'quit' 또는 'exit': 프로그램 종료")
    print("   • 'new' 또는 '새파일': 새로운 데이터 파일 선택") 
    print("   • 'status' 또는 '상태': 현재 세션 상태 확인")
    print("   • 'help' 또는 '도움말': 이 메시지 표시")
    print()


def ask_user_confirmation(message: str) -> bool:
    """사용자 확인 요청"""
    while True:
        try:
            response = input(f"{message} (y/n): ").strip().lower()
            if response in ['y', 'yes', '예', 'ㅇ']:
                return True
            elif response in ['n', 'no', '아니오', 'ㄴ']:
                return False
            else:
                print("❌ 'y' 또는 'n'으로 답해주세요.")
        except KeyboardInterrupt:
            print("\n")
            return False 