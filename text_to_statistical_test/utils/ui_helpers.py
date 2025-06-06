"""
UI 헬퍼 함수들

사용자 인터페이스 관련 헬퍼 함수들
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class UIHelpers:
    """UI 헬퍼 클래스 - 사용자 인터페이스 관련 기능들을 제공"""
    
    def __init__(self):
        """UIHelpers 초기화"""
        self.logger = logging.getLogger(__name__)
    
    def display_file_selection_menu(self, data_files: List[str]) -> Optional[str]:
        """파일 선택 메뉴 표시"""
        return display_file_selection_menu(data_files)
    
    def print_analysis_guide(self):
        """분석 가이드 출력"""
        return print_analysis_guide()
    
    def print_step_header(self, step_number: int, step_name: str, description: str = ""):
        """파이프라인 단계 헤더 출력"""
        return print_step_header(step_number, step_name, description)
    
    def print_data_preview(self, data_preview: Dict[str, Any], title: str = "데이터 미리보기"):
        """데이터 미리보기 출력"""
        return print_data_preview(data_preview, title)
    
    def print_file_info(self, file_info: Dict[str, Any]):
        """파일 정보 출력"""
        return print_file_info(file_info)
    
    def print_analysis_options(self, options: List[Dict[str, Any]]):
        """분석 옵션 출력"""
        return print_analysis_options(options)
    
    def get_user_input(self, prompt: str, input_type: str = "text", 
                      valid_options: Optional[List[str]] = None) -> Optional[str]:
        """사용자 입력 받기"""
        return get_user_input(prompt, input_type, valid_options)
    
    def print_progress_bar(self, current: int, total: int, prefix: str = "진행률", 
                          suffix: str = "완료", length: int = 40):
        """진행률 바 출력"""
        return print_progress_bar(current, total, prefix, suffix, length)
    
    def print_error_message(self, error_msg: str, error_type: str = "일반 오류"):
        """오류 메시지 출력"""
        return print_error_message(error_msg, error_type)
    
    def print_success_message(self, success_msg: str, details: Optional[str] = None):
        """성공 메시지 출력"""
        return print_success_message(success_msg, details)
    
    def clear_screen(self):
        """화면 지우기"""
        return clear_screen()
    
    def confirm_action(self, message: str) -> bool:
        """사용자 확인"""
        return confirm_action(message)
    
    def display_table(self, data: List[Dict[str, Any]], headers: Optional[List[str]] = None, 
                     max_rows: int = 10):
        """테이블 형태로 데이터 출력"""
        return display_table(data, headers, max_rows)
    
    def wait_for_key(self, message: str = "계속하려면 Enter를 누르세요..."):
        """키 입력 대기"""
        return wait_for_key(message)

def display_file_selection_menu(data_files: List[str]) -> Optional[str]:
    """
    파일 선택 메뉴 표시
    
    Args:
        data_files: 선택 가능한 파일 목록
        
    Returns:
        Optional[str]: 선택된 파일 경로 (취소시 None)
    """
    if not data_files:
        print("선택 가능한 파일이 없습니다.")
        return None
    
    # 비대화형 모드에서는 첫 번째 파일 자동 선택
    if os.getenv('NON_INTERACTIVE') == 'true':
        selected_file = data_files[0]
        print(f"✅ 자동 선택된 파일: {Path(selected_file).name}")
        return selected_file
    
    print("\n📁 사용 가능한 데이터 파일:")
    print("=" * 60)
    
    for i, file_path in enumerate(data_files, 1):
        file_name = Path(file_path).name
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        file_ext = Path(file_path).suffix.upper()
        
        print(f" {i}. {file_name}")
        print(f"     📄 형식: {file_ext}  📊 크기: {file_size_mb:.1f}MB")
        print(f"     📂 경로: {file_path}")
        print()
    
    print("=" * 60)
    
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        try:
            selection = input(f"파일을 선택하세요 (1-{len(data_files)}, 0=취소): ").strip()
            
            if selection == '0':
                print("파일 선택이 취소되었습니다.")
                return None
            
            try:
                file_index = int(selection) - 1
                if 0 <= file_index < len(data_files):
                    selected_file = data_files[file_index]
                    print(f"✅ 선택된 파일: {Path(selected_file).name}")
                    return selected_file
                else:
                    print(f"❌ 잘못된 선택입니다. 1-{len(data_files)} 사이의 숫자를 입력해주세요.")
                    
            except ValueError:
                print("❌ 숫자를 입력해주세요.")
                
        except EOFError:
            print("\n파일 선택 메뉴 오류: 입력이 종료되었습니다.")
            # 비대화형 환경에서는 첫 번째 파일 자동 선택
            if len(data_files) > 0:
                selected_file = data_files[0]
                print(f"✅ 자동 선택된 파일: {Path(selected_file).name}")
                return selected_file
            return None
        except KeyboardInterrupt:
            print("\n파일 선택이 취소되었습니다.")
            return None
        except Exception as e:
            print(f"파일 선택 메뉴 오류: {e}")
            
        attempt += 1
    
    # 최대 시도 횟수 초과 시 첫 번째 파일 자동 선택
    print(f"❌ {max_attempts}번 시도했지만 선택에 실패했습니다.")
    if len(data_files) > 0:
        selected_file = data_files[0]
        print(f"✅ 자동으로 첫 번째 파일을 선택합니다: {Path(selected_file).name}")
        return selected_file
    
    return None

def print_analysis_guide():
    """
    분석 가이드 출력
    """
    guide_text = """
🔍 통계 분석 가이드
═══════════════════════════════════════════════════════════════

📊 지원하는 분석 유형:

1️⃣ 그룹 비교 분석
   • 독립표본 t-검정: 두 독립 그룹 간 평균 비교
   • 대응표본 t-검정: 단일 그룹의 처치 전후 비교
   • 일원분산분석 (ANOVA): 3개 이상 그룹 간 평균 비교
   • 이원분산분석: 두 독립변수의 영향 및 상호작용 분석

2️⃣ 관계 분석
   • 피어슨 상관분석: 연속형 변수 간 선형 관계
   • 스피어만 상관분석: 순위 기반 관계 분석
   • 단순/다중 선형회귀: 예측 모델링
   • 로지스틱 회귀: 범주형 결과 예측

3️⃣ 범주형 데이터 분석
   • 카이제곱 검정: 범주형 변수 간 연관성
   • Fisher 정확검정: 소표본 범주형 분석
   • McNemar 검정: 대응 범주형 변수 비교

💡 분석 요청 예시:
   • "성별에 따른 키의 차이를 알고 싶어요"
   • "교육 방법이 시험 점수에 미치는 영향은?"
   • "나이와 소득 간의 관계를 분석해주세요"
   • "브랜드 선호도와 연령대의 연관성은?"

🎯 분석 목적을 구체적으로 설명해주시면 더 정확한 분석을 제공할 수 있습니다!
═══════════════════════════════════════════════════════════════
"""
    print(guide_text)

def print_step_header(step_number: int, step_name: str, description: str = ""):
    """
    파이프라인 단계 헤더 출력
    
    Args:
        step_number: 단계 번호
        step_name: 단계 이름
        description: 단계 설명
    """
    print("\n" + "=" * 80)
    print(f"🔄 {step_number}단계: {step_name}")
    if description:
        print(f"📝 {description}")
    print("=" * 80)

def print_data_preview(data_preview: Dict[str, Any], title: str = "데이터 미리보기"):
    """
    데이터 미리보기 출력
    
    Args:
        data_preview: 데이터 미리보기 딕셔너리
        title: 출력 제목
    """
    print(f"\n📊 {title}")
    print("-" * 50)
    
    if 'head' in data_preview and data_preview['head']:
        print("📈 처음 몇 행:")
        for i, row in enumerate(data_preview['head'][:3]):
            print(f"   {i+1}: {row}")
        print()
    
    if 'sample' in data_preview and data_preview['sample']:
        print("🎲 무작위 샘플:")
        for i, row in enumerate(data_preview['sample'][:2]):
            print(f"   {i+1}: {row}")

def print_file_info(file_info: Dict[str, Any]):
    """
    파일 정보 출력
    
    Args:
        file_info: 파일 정보 딕셔너리
    """
    print("\n📄 파일 정보")
    print("-" * 30)
    print(f"📁 파일명: {file_info.get('file_name', 'N/A')}")
    print(f"📏 크기: {file_info.get('file_size', 0) / (1024*1024):.1f}MB")
    print(f"📊 행 수: {file_info.get('row_count', 0):,}")
    print(f"🏷️ 열 수: {file_info.get('column_count', 0)}")
    print(f"🎯 형식: {file_info.get('file_extension', 'N/A')}")
    
    if 'columns' in file_info and file_info['columns']:
        print(f"📋 컬럼: {', '.join(file_info['columns'][:5])}")
        if len(file_info['columns']) > 5:
            print(f"     ... 외 {len(file_info['columns']) - 5}개")

def print_analysis_options(options: List[Dict[str, Any]]):
    """
    분석 옵션 출력
    
    Args:
        options: 분석 옵션 리스트
    """
    print("\n🎯 추천 분석 방법:")
    print("=" * 50)
    
    for i, option in enumerate(options, 1):
        print(f"{i}. {option.get('name', 'Unknown')}")
        print(f"   📝 {option.get('description', '')}")
        if 'pros' in option:
            print(f"   ✅ 장점: {', '.join(option['pros'])}")
        if 'requirements' in option:
            print(f"   📋 요구사항: {', '.join(option['requirements'])}")
        print()

def get_user_input(prompt: str, input_type: str = "text", 
                   valid_options: Optional[List[str]] = None) -> Optional[str]:
    """
    사용자 입력 받기
    
    Args:
        prompt: 입력 프롬프트
        input_type: 입력 유형 ('text', 'number', 'yes_no')
        valid_options: 유효한 옵션 목록
        
    Returns:
        Optional[str]: 사용자 입력 (취소시 None)
    """
    # 비대화형 모드에서는 기본값 반환
    if os.getenv('NON_INTERACTIVE') == 'true':
        if input_type == "number":
            return "1"  # 기본값으로 1 반환
        elif input_type == "yes_no":
            return "yes"  # 기본값으로 yes 반환
        elif valid_options:
            return valid_options[0]  # 첫 번째 옵션 반환
        else:
            return "성별에 따른 만족도 평균 차이를 분석해줘"  # 기본 분석 요청
    
    try:
        while True:
            user_input = input(f"{prompt}: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                return None
            
            if input_type == "number":
                try:
                    int(user_input)
                    return user_input
                except ValueError:
                    print("❌ 숫자를 입력해주세요.")
                    continue
            
            elif input_type == "yes_no":
                if user_input.lower() in ['y', 'yes', '예', '네']:
                    return 'yes'
                elif user_input.lower() in ['n', 'no', '아니오', '아님']:
                    return 'no'
                else:
                    print("❌ 'y' 또는 'n'을 입력해주세요.")
                    continue
            
            elif valid_options:
                if user_input in valid_options:
                    return user_input
                else:
                    print(f"❌ 다음 중 하나를 선택해주세요: {', '.join(valid_options)}")
                    continue
            
            return user_input
            
    except KeyboardInterrupt:
        print("\n입력이 취소되었습니다.")
        return None
    except Exception as e:
        logger.error(f"사용자 입력 오류: {e}")
        return None

def print_progress_bar(current: int, total: int, prefix: str = "진행률", 
                      suffix: str = "완료", length: int = 40):
    """
    진행률 표시줄 출력
    
    Args:
        current: 현재 진행 상황
        total: 전체 작업량
        prefix: 접두사
        suffix: 접미사
        length: 표시줄 길이
    """
    percent = current / total * 100
    filled_length = int(length * current // total)
    bar = "█" * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent:.1f}% {suffix}", end="")
    
    if current == total:
        print()  # 완료 시 새 줄

def print_error_message(error_msg: str, error_type: str = "일반 오류"):
    """
    오류 메시지 출력
    
    Args:
        error_msg: 오류 메시지
        error_type: 오류 유형
    """
    print(f"\n❌ {error_type}")
    print("=" * 50)
    print(f"📝 {error_msg}")
    print("=" * 50)

def print_success_message(success_msg: str, details: Optional[str] = None):
    """
    성공 메시지 출력
    
    Args:
        success_msg: 성공 메시지
        details: 상세 정보
    """
    print(f"\n✅ {success_msg}")
    if details:
        print(f"📝 {details}")

def clear_screen():
    """
    화면 지우기
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def confirm_action(message: str) -> bool:
    """
    사용자에게 확인 요청
    
    Args:
        message: 확인 메시지
        
    Returns:
        bool: 확인 결과
    """
    response = get_user_input(f"{message} (y/n)", input_type="yes_no")
    return response == 'yes'

def display_table(data: List[Dict[str, Any]], headers: Optional[List[str]] = None, 
                 max_rows: int = 10):
    """
    테이블 형식으로 데이터 출력
    
    Args:
        data: 출력할 데이터
        headers: 테이블 헤더
        max_rows: 최대 출력 행 수
    """
    if not data:
        print("표시할 데이터가 없습니다.")
        return
    
    # 헤더 설정
    if not headers and data:
        headers = list(data[0].keys())
    
    if not headers:
        return
    
    # 컬럼 너비 계산
    col_widths = {}
    for header in headers:
        col_widths[header] = max(
            len(str(header)),
            max(len(str(row.get(header, ''))) for row in data[:max_rows])
        )
    
    # 헤더 출력
    header_line = " | ".join(f"{header:<{col_widths[header]}}" for header in headers)
    print(header_line)
    print("-" * len(header_line))
    
    # 데이터 출력
    for i, row in enumerate(data[:max_rows]):
        row_line = " | ".join(f"{str(row.get(header, '')):<{col_widths[header]}}" 
                             for header in headers)
        print(row_line)
    
    if len(data) > max_rows:
        print(f"... 외 {len(data) - max_rows}개 행")

def wait_for_key(message: str = "계속하려면 Enter를 누르세요..."):
    """
    키 입력 대기
    
    Args:
        message: 대기 메시지
    """
    try:
        input(f"\n{message}")
    except KeyboardInterrupt:
        print("\n취소되었습니다.") 