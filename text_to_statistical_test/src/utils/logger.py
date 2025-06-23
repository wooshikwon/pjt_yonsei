import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

class StatisticalAnalysisLogger:
    """
    통계 분석 시스템 전용 로거.
    터미널 출력을 간소화하고 상세 로그는 날짜별 파일로 저장합니다.
    """
    
    def __init__(self, base_path: str = "logs"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # 날짜별 로그 파일명 생성
        today = datetime.now().strftime("%Y%m%d")
        self.log_file = self.base_path / f"analysis_{today}.log"
        
        # 파일 로거 설정 (상세 로그)
        self.file_logger = logging.getLogger("file_logger")
        self.file_logger.setLevel(logging.DEBUG)
        
        # 기존 핸들러 제거 (중복 방지)
        self.file_logger.handlers.clear()
        
        # 파일 핸들러 추가
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.file_logger.addHandler(file_handler)
        
        # 콘솔 로거 설정 (간소한 출력)
        self.console_logger = logging.getLogger("console_logger")
        self.console_logger.setLevel(logging.INFO)
        self.console_logger.handlers.clear()
        
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        self.console_logger.addHandler(console_handler)

    def log_system_info(self, message: str):
        """시스템 정보를 파일과 터미널 모두에 기록"""
        self.file_logger.info(message)
        self.console_logger.info(f"🔧 {message}")

    def log_step_start(self, step_num: int, step_description: str):
        """단계 시작을 기록"""
        self.file_logger.info(f"=== Step {step_num} Started: {step_description} ===")
        self.console_logger.info(f"📋 Step {step_num}: {step_description}")

    def log_step_success(self, step_num: int, brief_result: str = "완료"):
        """단계 성공을 기록"""
        self.file_logger.info(f"Step {step_num} completed successfully: {brief_result}")
        self.console_logger.info(f"✅ Step {step_num}: {brief_result}")

    def log_step_failure(self, step_num: int, error_message: str):
        """단계 실패를 기록"""
        self.file_logger.error(f"Step {step_num} failed: {error_message}")
        self.console_logger.info(f"❌ Step {step_num}: 실패")

    def log_detailed(self, message: str, level: str = "INFO"):
        """상세 정보를 파일에만 기록 (터미널 출력 안함)"""
        if level.upper() == "DEBUG":
            self.file_logger.debug(message)
        elif level.upper() == "ERROR":
            self.file_logger.error(message)
        else:
            self.file_logger.info(message)

    def log_rag_context(self, context: str):
        """RAG 컨텍스트를 파일에만 상세 기록"""
        self.file_logger.info("=== RAG Context Retrieved ===")
        self.file_logger.info(context)
        self.file_logger.info("=== End RAG Context ===")

    def log_generated_code(self, step_num: int, code: str, rationale: str, is_corrected: bool = False):
        """
        LLM이 생성한 Rationale과 Code를 로깅합니다.
        
        Args:
            step_num (int): 현재 단계 번호.
            code (str): 생성된 Python 코드.
            rationale (str): 코드 생성의 근거가 된 Rationale.
            is_corrected (bool, optional): 자가 수정된 코드인지 여부. Defaults to False.
        """
        log_header = f"--- Generated Code for Step {step_num} {'(Corrected)' if is_corrected else ''} ---"
        
        log_message = (
            f"{log_header}\n"
            f"Rationale: {rationale}\n\n"  # Rationale과 Code 사이에 한 줄 띄움
            f"Code:\n{code}\n"
            f"=== End Generated Code ==="
        )
        self.file_logger.info(log_message)

    def log_execution_result(self, step_num: int, result: str, success: bool):
        """코드 실행 결과를 파일에만 기록"""
        status = "SUCCESS" if success else "FAILED"
        self.file_logger.info(f"=== Execution Result for Step {step_num}: {status} ===")
        self.file_logger.info(result)
        self.file_logger.info("=== End Execution Result ===")

    def print_final_report(self, report: str):
        """최종 보고서를 터미널에 깔끔하게 출력"""
        self.file_logger.info("=== FINAL REPORT ===")
        self.file_logger.info(report)
        self.file_logger.info("=== END FINAL REPORT ===")
        
        # 터미널에는 깔끔하게 출력
        print("\n" + "="*80)
        print("📊 통계 분석 결과 보고서")
        print("="*80)
        print(report)
        print("="*80)

    def log_report_saved(self, file_path: str):
        """최종 보고서가 저장되었음을 콘솔에 알립니다."""
        self.console_logger.info(f"💾 보고서가 저장되었습니다: {file_path}")

    def log_final_data_saved(self, file_path: str):
        """최종 데이터가 저장되었음을 콘솔에 알립니다."""
        self.console_logger.info(f"💾 최종 데이터가 저장되었습니다: {file_path}")

    def log_step_separator(self):
        """단계 구분을 위한 수평선을 파일 로그에 출력합니다."""
        self.file_logger.info("\n" + "="*80 + "\n")

    def log_data_summary(self, summary: str):
        """데이터 요약 정보를 파일 로그에 기록합니다."""
        self.file_logger.info("--- Data Summary ---")
        self.file_logger.info(summary)
        self.file_logger.info("--------------------")

# 전역 로거 인스턴스
analysis_logger: Optional[StatisticalAnalysisLogger] = None

def get_logger() -> StatisticalAnalysisLogger:
    """전역 로거 인스턴스를 반환합니다."""
    global analysis_logger
    if analysis_logger is None:
        analysis_logger = StatisticalAnalysisLogger()
    return analysis_logger 