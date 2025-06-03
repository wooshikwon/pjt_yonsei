# src/data_ingestion.py

import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def _load_json_file(file_path: str, file_description: str) -> Optional[Dict[str, Any]]:
    """Helper to load a JSON file with error handling."""
    if not os.path.exists(file_path):
        logger.error(f"{file_description} file not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Successfully loaded {file_description} from: {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_description} file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading {file_description} file {file_path}: {e}")
        return None

def _validate_property_register_data(data: Dict[str, Any]) -> bool:
    """등기부등본 필수 정보 유효성 검사 (예시)"""
    # 갑구, 을구 정보가 있는지, 필수 키가 있는지 등 검사
    if not data.get("갑구") or not data.get("을구"):
        logger.error("Property register data is missing '갑구' or '을구' sections.")
        return False
    # 추가적인 상세 검증 로직 필요
    logger.debug("Property register data basic validation passed.")
    return True

def _validate_contract_info_data(data: Dict[str, Any]) -> bool:
    """계약 정보 필수 정보 유효성 검사 (예시)"""
    if not data.get("property_address") or not data.get("jeonse_deposit_amount"):
        logger.error("Contract info data is missing 'property_address' or 'jeonse_deposit_amount'.")
        return False
    logger.debug("Contract info data basic validation passed.")
    return True

def load_and_preprocess_input_data(test_case_id: str, settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    지정된 테스트 케이스 ID에 대한 입력 파일들을 로드하고 기본적인 전처리/검증을 수행합니다.
    등기부등본 PDF를 직접 처리하는 대신, 사용자가 PDF를 분석하여 생성한
    구조화된 JSON 파일 (property_register_input.json)을 로드합니다.
    """
    input_base_path = settings.get('paths', {}).get('input_data_base', 'data/input/')
    case_path = os.path.join(input_base_path, test_case_id)

    if not os.path.isdir(case_path):
        logger.error(f"Test case directory not found: {case_path}")
        return None

    logger.info(f"Loading input data from directory: {case_path}")

    # 필수 입력 파일 정의 (사용자가 PDF를 보고 수동으로 또는 별도 도구로 생성한 JSON)
    property_register_file = os.path.join(case_path, "property_register_input.json")
    contract_info_file = os.path.join(case_path, "contract_info_input.json")
    
    # 모의 API 데이터 파일 (실제 API 호출 시에는 사용되지 않을 수 있음)
    # 이 파일들은 external_api_handler.py 에서 모의 사용 여부에 따라 참조됨
    # building_ledger_mock_file = os.path.join(case_path, "building_ledger_api_mock.json")
    # transaction_price_mock_file = os.path.join(case_path, "transaction_price_api_mock.json")

    processed_data: Dict[str, Any] = {}

    # 1. 등기부등본 정보 로드 (사용자 생성 JSON)
    property_register_data = _load_json_file(property_register_file, "Property Register (JSON)")
    if not property_register_data or not _validate_property_register_data(property_register_data):
        logger.error("Failed to load or validate mandatory property register data.")
        return None
    processed_data['property_register'] = property_register_data
    
    # property_register_data 에서 주소 추출 (API 호출 시 사용)
    # 예시: 등기부 표제부 주소. 실제 JSON 구조에 따라 키 경로 수정 필요
    # property_address_full = property_register_data.get("표제부", {}).get("소재지번", "주소 정보 없음")
    # processed_data['property_details'] = {"address_full": property_address_full}
    # 더 정확한 주소는 계약 정보에서 가져올 수도 있음

    # 2. 계약 정보 로드
    contract_info_data = _load_json_file(contract_info_file, "Contract Information")
    if not contract_info_data or not _validate_contract_info_data(contract_info_data):
        logger.error("Failed to load or validate mandatory contract information.")
        return None
    processed_data['contract_info'] = contract_info_data
    
    # 계약 정보에서 주소 및 기타 정보를 property_details로 통합
    property_details = processed_data.get('property_details', {})
    property_details['address_full'] = contract_info_data.get('property_address', property_details.get('address_full', '주소 정보 없음'))
    property_details['property_type'] = contract_info_data.get('property_type') # 예: 아파트, 빌라, 단독주택 (API 호출 시 필요)
    processed_data['property_details'] = property_details


    # 3. (선택) 임대인 관련 정보 처리
    # contract_info_data 내에 'lessor_info' 키 등으로 선택적 정보 포함 가능
    # 예: lessor_tax_info_provided = 'lessor_tax_certificate' in contract_info_data.get('lessor_info', {})
    # 이 정보는 위험 평가 시 고려될 수 있음

    # 4. 모의 API 데이터 파일 경로 저장 (실제 로드는 external_api_handler에서)
    # 모의 API 데이터는 raw_input_data에 포함되어 external_api_handler로 전달될 수 있도록 구성
    # 여기서는 파일 경로만 전달하고, 실제 로드는 핸들러가 수행하도록 하거나, 여기서 로드할 수도 있음
    processed_data['mock_data_paths'] = {
        "building_ledger": os.path.join(case_path, "building_ledger_api_mock.json"),
        "transaction_price": os.path.join(case_path, "transaction_price_api_mock.json")
    }


    # TODO: 추가적인 데이터 전처리 및 정제 로직
    # 예: 날짜 형식 통일, 금액 단위 변환, 텍스트 정규화 등

    logger.info(f"Input data for '{test_case_id}' loaded and preprocessed.")
    return processed_data