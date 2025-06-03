# src/main.py

import argparse
import logging
import os
import time

# config_loader.py에서 APP_SETTINGS와 load_app_config를 import
from .config_loader import APP_SETTINGS, load_app_config

from .data_ingestion import load_and_preprocess_input_data
from .external_api_handler import ExternalAPIHandler
from .risk_assessment_engine import RiskAssessmentEngine
from .solution_advisor import SolutionAdvisor
from .report_generator import ReportGenerator

# 애플리케이션 설정 및 로깅 초기화 (가장 먼저 실행)
try:
    load_app_config()
    logger = logging.getLogger(__name__) # 설정이 로드된 후 로거 가져오기
    logger.info("Application configuration and logging initialized for main execution.")
except Exception as e:
    # 로깅 설정 실패 시 print로 에러 출력 후 종료
    print(f"CRITICAL: Failed to initialize application configuration in main: {e}")
    # traceback.print_exc() # 상세 에러 출력
    exit(1)


def run_pipeline(test_case_id: str) -> None:
    """
    지정된 테스트 케이스 ID에 대해 전체 PoC 파이프라인을 실행합니다.
    """
    logger.info(f"Starting pipeline for test case ID: {test_case_id}")
    start_time = time.time()

    try:
        # 1. 입력 데이터 로드 및 전처리 (PDF에서 구조화된 JSON으로 변환된 데이터 로드)
        logger.info("Step 1: Loading and preprocessing input data...")
        processed_input_data = load_and_preprocess_input_data(test_case_id, APP_SETTINGS)
        if not processed_input_data:
            logger.error(f"Failed to load or preprocess input data for '{test_case_id}'. Aborting.")
            return
        logger.info("Input data loaded and preprocessed successfully.")

        # 2. 외부 API 데이터 조회 (실제 API 호출)
        logger.info("Step 2: Fetching data from external APIs...")
        api_handler = ExternalAPIHandler(APP_SETTINGS)
        # 주소 정보는 processed_input_data 내에 포함되어 있다고 가정
        # 예: property_address = processed_input_data.get('contract_info', {}).get('property_address')
        property_address = processed_input_data.get('property_details', {}).get('address_full', '주소 정보 없음') # 예시 경로
        
        external_data = {}
        if property_address != '주소 정보 없음':
            external_data['building_ledger'] = api_handler.fetch_building_ledger_info(property_address)
            external_data['transaction_price'] = api_handler.fetch_real_estate_transaction_price(property_address, processed_input_data.get('property_details', {}))
        else:
            logger.warning("Property address not found in input data. Skipping external API calls.")
            external_data['building_ledger'] = {"error": "주소 정보 없음"}
            external_data['transaction_price'] = {"error": "주소 정보 없음"}
        
        logger.info("External API data fetched.")
        logger.debug(f"External data: {external_data}")

        # 3. 위험 평가 수행
        logger.info("Step 3: Assessing risks...")
        risk_engine = RiskAssessmentEngine(APP_SETTINGS)
        risk_assessment_result = risk_engine.assess(processed_input_data, external_data)
        logger.info("Risk assessment completed.")

        # 4. 해결책 및 조언 생성
        logger.info("Step 4: Generating solutions and advice...")
        solution_engine = SolutionAdvisor(APP_SETTINGS)
        solution_advice_result = solution_engine.advise(processed_input_data, external_data, risk_assessment_result)
        logger.info("Solutions and advice generated.")

        # 5. 최종 보고서 생성
        logger.info("Step 5: Generating final report...")
        report_gen = ReportGenerator(APP_SETTINGS)
        report_path = report_gen.create(
            test_case_id,
            processed_input_data,
            external_data,
            risk_assessment_result,
            solution_advice_result
        )
        logger.info(f"Final report generated successfully at: {report_path}")

    except FileNotFoundError as e:
        logger.error(f"A required file was not found: {e}", exc_info=True)
    except KeyError as e:
        logger.error(f"Missing expected key in data or configuration: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred in the pipeline: {e}", exc_info=True)
    finally:
        end_time = time.time()
        logger.info(f"Pipeline for '{test_case_id}' finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Jeonse Fraud Prevention LLM PoC Pipeline.")
    parser.add_argument(
        "--test_case_id",
        type=str,
        required=True, # 실제 운영 시에는 필수로 받도록 변경
        # default=APP_SETTINGS.get('default_test_case_id', 'test_case_001'), # APP_SETTINGS 로드 후 사용 가능
        help="The ID of the test case to process (e.g., test_case_001)."
    )
    args = parser.parse_args()

    # default_test_case_id는 APP_SETTINGS가 로드된 후에 접근 가능
    test_case_to_run = args.test_case_id if args.test_case_id else APP_SETTINGS.get('default_test_case_id')
    
    if not test_case_to_run:
        logger.critical("No test case ID provided and no default set in settings.yaml. Exiting.")
        exit(1)
        
    run_pipeline(test_case_to_run)