# src/external_api_handler.py

import os
import logging
import requests # 실제 API 호출을 위해 필요 (requirements.txt에 추가)
import json # 모의 데이터 로딩용 (필요 시)
from typing import Dict, Any, Optional
from urllib.parse import urlencode # API 파라미터 인코딩용
import re

logger = logging.getLogger(__name__)

class ExternalAPIHandler:
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.api_settings = settings.get('external_apis', {})
        logger.info("ExternalAPIHandler initialized.")

    def _make_api_request(self, base_url: str, endpoint: str, params: Dict[str, Any], service_name: str) -> Optional[Dict[str, Any]]:
        """공통 API 요청 함수"""
        full_url = f"{base_url}{endpoint}"
        # GET 요청의 경우 params를 URL에 인코딩
        # encoded_params = urlencode(params) # 일부 API는 이 방식이 필요
        # request_url = f"{full_url}?{encoded_params}"
        # POST 요청이나, requests 라이브러리가 params를 잘 처리하므로 아래와 같이 사용 가능
        
        logger.debug(f"Requesting {service_name} API: URL={full_url}, Params={params}")
        try:
            # 대부분의 공공 API는 GET 요청이지만, 필요시 method 변경
            response = requests.get(full_url, params=params, timeout=self.settings.get('llm', {}).get('timeout', 120)/2) # LLM 타임아웃의 절반 정도
            response.raise_for_status() # 200 OK가 아니면 HTTPError 발생
            
            # 응답 형식에 따라 .json() 또는 .text 후 XML 파싱 등 필요
            # 공공데이터포털 API는 XML을 반환하는 경우가 많음. PoC에서는 JSON 반환 가정.
            # 실제로는 XML 파서 (예: xml.etree.ElementTree) 사용 필요
            # 여기서는 응답이 JSON이라고 가정하거나, XML을 JSON으로 변환하는 로직이 추가되어야 함.
            content_type = response.headers.get('Content-Type', '')
            if 'json' in content_type:
                data = response.json()
            elif 'xml' in content_type:
                logger.warning(f"{service_name} API returned XML. XML parsing not fully implemented in this PoC. Attempting basic text.")
                # TODO: XML 파싱 로직 추가 (xml.etree.ElementTree 등 사용)
                # data = self._parse_xml_response(response.text) # 예시
                data = {"raw_xml_content": response.text, "error": "XML response requires specific parser"}
            else:
                logger.warning(f"Unexpected content type from {service_name} API: {content_type}")
                data = {"raw_content": response.text, "error": "Unexpected content type"}

            logger.info(f"{service_name} API request successful.")
            logger.debug(f"{service_name} API Response (first 500 chars): {str(data)[:500]}")
            return data
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error occurred while calling {service_name} API: {e.response.status_code} - {e.response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error occurred while calling {service_name} API: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during {service_name} API call: {e}", exc_info=True)
        return None

    def fetch_building_ledger_info(self, property_address: str) -> Optional[Dict[str, Any]]:
        """
        건축물대장 정보를 실제 API를 통해 조회합니다.
        주소 정보를 바탕으로 표제부, 기본개요 등을 조회하여 위반건축물 여부 등을 파악합니다.
        """
        config = self.api_settings.get('building_ledger', {})
        api_key_env_var = config.get('service_key_env_var')
        api_key = os.getenv(api_key_env_var) if api_key_env_var else None
        base_url = config.get('base_url')
        
        if not all([api_key, base_url]):
            logger.error("Building Ledger API key or base_url not configured.")
            return {"error": "API configuration missing"}

        # 예시: 표제부 정보 조회 (위반건축물 여부 등)
        title_endpoint = config.get('endpoints', {}).get('title_info')
        if not title_endpoint:
            logger.error("Building Ledger title_info endpoint not configured.")
            return {"error": "API endpoint configuration missing"}

        # API 요청 파라미터는 해당 API 명세에 따라 구성
        # 주소를 시군구코드, 법정동코드, 번지 등으로 분리해야 할 수 있음
        # 여기서는 property_address를 단순 파라미터로 전달하는 예시 (실제로는 더 복잡)
        params = {
            "ServiceKey": api_key,
            **self._parse_address_for_building_api(property_address),
            "numOfRows": "10",
            "pageNo": "1",
            "_type": "json"
        }
        # TODO: property_address를 파싱하여 sigunguCd, bjdongCd, bun, ji 등으로 변환하는 로직 필요
        logger.warning(f"Building Ledger API requires address parsing to specific codes (sigunguCd, bjdongCd, etc.). Current address: {property_address}. Using placeholder params.")
        # 실제 파라미터 채우기 (예시, 실제 API 명세 확인 필수)
        # params.update(self._parse_address_for_building_api(property_address))


        # PoC에서는 모의 데이터 사용 여부 확인 로직 제거 (실제 API 호출 가정)
        # if self.settings.get('api_simulation', {}).get('use_mock_building_ledger_api'):
        #     # 모의 데이터 로드 로직 (data_ingestion에서 전달받거나 여기서 로드)
        #     logger.info("Using mock data for Building Ledger API (as per settings).")
        #     return {"mock_data": "Building ledger mock data for " + property_address}

        return self._make_api_request(base_url, title_endpoint, params, "Building Ledger (Title)")
        # 필요시 기본개요 등 다른 정보도 추가 조회

    def fetch_real_estate_transaction_price(self, property_address: str, property_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        부동산 실거래가 정보를 실제 API를 통해 조회합니다.
        주소 및 부동산 유형(아파트, 빌라 등)에 따라 적절한 엔드포인트 사용.
        """
        config = self.api_settings.get('real_estate_transaction', {})
        api_key_env_var = config.get('service_key_env_var')
        api_key = os.getenv(api_key_env_var) if api_key_env_var else None
        base_url = config.get('base_url')

        if not all([api_key, base_url]):
            logger.error("Real Estate Transaction API key or base_url not configured.")
            return {"error": "API configuration missing"}

        # 부동산 유형에 따라 다른 엔드포인트 사용 (settings.yaml에 더 많은 엔드포인트 정의 필요)
        property_type = property_details.get('property_type', '아파트') # 기본값 아파트
        deal_ymd = datetime.datetime.now().strftime("%Y%m") # 조회 기준년월 (예: 현재년월)
        
        endpoint_map = config.get('endpoints', {})
        if property_type == "아파트":
            endpoint = endpoint_map.get('apartment_trade') # 또는 전월세 엔드포인트
        # elif property_type == "빌라": # 연립다세대
        #     endpoint = endpoint_map.get('row_house_trade')
        else:
            logger.warning(f"Unsupported property type for Real Estate API: {property_type}. Defaulting to Apartment Trade.")
            endpoint = endpoint_map.get('apartment_trade')
            
        if not endpoint:
            logger.error(f"Real Estate Transaction endpoint for {property_type} not configured.")
            return {"error": "API endpoint configuration missing"}

        params = {
            "serviceKey": api_key,
            "LAWD_CD": self._get_lawd_code_from_address(property_address),
            "DEAL_YMD": deal_ymd, # 계약년월
            "numOfRows": "50", # 최근 50건 정도 조회
            "_type": "json"
        }
        # TODO: property_address를 파싱하여 LAWD_CD (법정동코드 앞 5자리) 추출 로직 필요
        logger.warning(f"Real Estate API requires LAWD_CD from address. Current address: {property_address}. Using placeholder params.")
        # params["LAWD_CD"] = self._get_lawd_code_from_address(property_address)

        return self._make_api_request(base_url, endpoint, params, f"Real Estate Transaction ({property_type})")

    def _parse_address_for_building_api(self, address: str) -> Dict[str, str]:
        """
        주소 문자열을 건축물대장 API가 요구하는 코드로 변환하는 PoC용 간단 함수.
        실제 서비스에서는 공공데이터포털의 법정동코드 데이터셋을 활용해야 함.
        """
        # 예시: 서울특별시 강남구 역삼동 123-45
        # sigunguCd: 11680 (강남구), bjdongCd: 10300 (역삼동), bun: 00123, ji: 00045
        # 실제로는 외부 데이터셋 필요. 여기서는 임의 값 반환
        return {"sigunguCd": "11680", "bjdongCd": "10300", "platGbCd": "0", "bun": "00123", "ji": "00045"}

    def _get_lawd_code_from_address(self, address: str) -> str:
        """
        주소 문자열에서 법정동코드(LAWD_CD) 앞 5자리 추출 PoC용 함수.
        실제 서비스에서는 외부 데이터셋 필요.
        """
        # 예시: 서울특별시 강남구 역삼동 123-45 → 11680
        return "11680"