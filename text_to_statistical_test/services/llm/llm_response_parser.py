"""
LLM Response Parser

LLM 응답을 파싱하고 구조화된 데이터로 변환
- JSON 응답 파싱
- 텍스트에서 구조화된 정보 추출
- 응답 검증 및 정제
- 오류 처리 및 복구
"""

import logging
import json
import re
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from utils.error_handler import ErrorHandler, ParsingException
from utils.helpers import extract_keywords

logger = logging.getLogger(__name__)

class ResponseType(Enum):
    """응답 타입 열거형"""
    JSON = "json"
    TEXT = "text"
    CODE = "code"
    MIXED = "mixed"

@dataclass
class ParsedResponse:
    """파싱된 응답 데이터 클래스"""
    content: Any
    response_type: ResponseType
    confidence: float
    metadata: Dict[str, Any]
    raw_content: str
    parsing_errors: List[str] = None

class LLMResponseParser:
    """LLM 응답 파서 메인 클래스"""
    
    def __init__(self):
        """파서 초기화"""
        self.error_handler = ErrorHandler()
        
        # JSON 패턴
        self.json_patterns = [
            r'```json\s*(.*?)\s*```',  # ```json ... ```
            r'```\s*(.*?)\s*```',      # ``` ... ```
            r'\{.*\}',                 # { ... }
            r'\[.*\]'                  # [ ... ]
        ]
        
        # 코드 패턴
        self.code_patterns = [
            r'```python\s*(.*?)\s*```',  # ```python ... ```
            r'```py\s*(.*?)\s*```',      # ```py ... ```
            r'```\s*(.*?)\s*```'         # ``` ... ```
        ]
        
        logger.info("LLM 응답 파서 초기화 완료")
    
    def parse_response(self, 
                      response_text: str,
                      expected_type: Optional[ResponseType] = None,
                      strict_mode: bool = False) -> ParsedResponse:
        """
        LLM 응답 파싱
        
        Args:
            response_text: 원본 응답 텍스트
            expected_type: 예상 응답 타입
            strict_mode: 엄격 모드 (파싱 실패 시 예외 발생)
            
        Returns:
            ParsedResponse: 파싱된 응답
        """
        try:
            # 응답 타입 자동 감지
            if expected_type is None:
                detected_type = self._detect_response_type(response_text)
            else:
                detected_type = expected_type
            
            # 타입별 파싱
            if detected_type == ResponseType.JSON:
                return self._parse_json_response(response_text, strict_mode)
            elif detected_type == ResponseType.CODE:
                return self._parse_code_response(response_text, strict_mode)
            elif detected_type == ResponseType.MIXED:
                return self._parse_mixed_response(response_text, strict_mode)
            else:
                return self._parse_text_response(response_text)
                
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'response_length': len(response_text)})
            
            if strict_mode:
                raise ParsingException(f"응답 파싱 실패: {error_info['message']}")
            
            # 실패 시 텍스트로 반환
            return ParsedResponse(
                content=response_text,
                response_type=ResponseType.TEXT,
                confidence=0.0,
                metadata={'parsing_failed': True, 'error': str(e)},
                raw_content=response_text,
                parsing_errors=[str(e)]
            )
    
    def _detect_response_type(self, text: str) -> ResponseType:
        """응답 타입 자동 감지"""
        # JSON 패턴 확인
        json_score = 0
        for pattern in self.json_patterns:
            if re.search(pattern, text, re.DOTALL):
                json_score += 1
        
        # 코드 패턴 확인
        code_score = 0
        for pattern in self.code_patterns:
            if re.search(pattern, text, re.DOTALL):
                code_score += 1
        
        # 키워드 기반 감지
        json_keywords = ['json', '{', '}', '[', ']', '":', 'null', 'true', 'false']
        code_keywords = ['import', 'def ', 'class ', 'if ', 'for ', 'while ', 'print(', 'return']
        
        json_keyword_count = sum(1 for keyword in json_keywords if keyword in text.lower())
        code_keyword_count = sum(1 for keyword in code_keywords if keyword in text.lower())
        
        # 점수 계산
        total_json_score = json_score * 2 + json_keyword_count
        total_code_score = code_score * 2 + code_keyword_count
        
        if total_json_score > 0 and total_code_score > 0:
            return ResponseType.MIXED
        elif total_json_score > total_code_score:
            return ResponseType.JSON
        elif total_code_score > 0:
            return ResponseType.CODE
        else:
            return ResponseType.TEXT
    
    def _parse_json_response(self, text: str, strict_mode: bool) -> ParsedResponse:
        """JSON 응답 파싱"""
        parsing_errors = []
        
        # JSON 추출 시도
        json_content = None
        confidence = 0.0
        
        for pattern in self.json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # JSON 파싱 시도
                    json_content = json.loads(match.strip())
                    confidence = 1.0
                    break
                except json.JSONDecodeError as e:
                    parsing_errors.append(f"JSON 파싱 오류: {str(e)}")
                    continue
            
            if json_content is not None:
                break
        
        # JSON을 찾지 못한 경우 전체 텍스트에서 시도
        if json_content is None:
            try:
                json_content = json.loads(text.strip())
                confidence = 1.0
            except json.JSONDecodeError as e:
                parsing_errors.append(f"전체 텍스트 JSON 파싱 오류: {str(e)}")
                
                if strict_mode:
                    raise ParsingException(f"JSON 파싱 실패: {str(e)}")
                
                # 부분적 JSON 복구 시도
                json_content = self._attempt_json_recovery(text)
                confidence = 0.5 if json_content else 0.0
        
        return ParsedResponse(
            content=json_content or text,
            response_type=ResponseType.JSON,
            confidence=confidence,
            metadata={'json_extracted': json_content is not None},
            raw_content=text,
            parsing_errors=parsing_errors if parsing_errors else None
        )
    
    def _parse_code_response(self, text: str, strict_mode: bool) -> ParsedResponse:
        """코드 응답 파싱"""
        parsing_errors = []
        
        # 코드 블록 추출
        code_blocks = []
        
        for pattern in self.code_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            code_blocks.extend([match.strip() for match in matches])
        
        if not code_blocks:
            # 코드 블록이 없으면 전체 텍스트를 코드로 간주
            code_blocks = [text.strip()]
        
        # 코드 검증
        validated_code = []
        for code in code_blocks:
            validation_result = self._validate_python_code(code)
            validated_code.append({
                'code': code,
                'is_valid': validation_result['is_valid'],
                'errors': validation_result['errors']
            })
            
            if not validation_result['is_valid']:
                parsing_errors.extend(validation_result['errors'])
        
        confidence = sum(1 for code in validated_code if code['is_valid']) / len(validated_code)
        
        return ParsedResponse(
            content=validated_code,
            response_type=ResponseType.CODE,
            confidence=confidence,
            metadata={'code_blocks_count': len(code_blocks)},
            raw_content=text,
            parsing_errors=parsing_errors if parsing_errors else None
        )
    
    def _parse_mixed_response(self, text: str, strict_mode: bool) -> ParsedResponse:
        """혼합 응답 파싱"""
        result = {
            'text_parts': [],
            'json_parts': [],
            'code_parts': []
        }
        
        parsing_errors = []
        
        # JSON 부분 추출
        for pattern in self.json_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    json_content = json.loads(match.group(1).strip())
                    result['json_parts'].append({
                        'content': json_content,
                        'start': match.start(),
                        'end': match.end()
                    })
                except json.JSONDecodeError as e:
                    parsing_errors.append(f"JSON 파싱 오류: {str(e)}")
        
        # 코드 부분 추출
        for pattern in self.code_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                code = match.group(1).strip()
                validation = self._validate_python_code(code)
                result['code_parts'].append({
                    'content': code,
                    'is_valid': validation['is_valid'],
                    'errors': validation['errors'],
                    'start': match.start(),
                    'end': match.end()
                })
        
        # 텍스트 부분 추출 (JSON, 코드가 아닌 부분)
        used_ranges = []
        for part in result['json_parts'] + result['code_parts']:
            used_ranges.append((part['start'], part['end']))
        
        used_ranges.sort()
        text_parts = []
        last_end = 0
        
        for start, end in used_ranges:
            if start > last_end:
                text_part = text[last_end:start].strip()
                if text_part:
                    text_parts.append(text_part)
            last_end = end
        
        if last_end < len(text):
            text_part = text[last_end:].strip()
            if text_part:
                text_parts.append(text_part)
        
        result['text_parts'] = text_parts
        
        # 신뢰도 계산
        total_parts = len(result['json_parts']) + len(result['code_parts']) + len(result['text_parts'])
        valid_parts = len(result['json_parts']) + sum(1 for code in result['code_parts'] if code['is_valid']) + len(result['text_parts'])
        confidence = valid_parts / total_parts if total_parts > 0 else 0.0
        
        return ParsedResponse(
            content=result,
            response_type=ResponseType.MIXED,
            confidence=confidence,
            metadata={
                'json_count': len(result['json_parts']),
                'code_count': len(result['code_parts']),
                'text_count': len(result['text_parts'])
            },
            raw_content=text,
            parsing_errors=parsing_errors if parsing_errors else None
        )
    
    def _parse_text_response(self, text: str) -> ParsedResponse:
        """텍스트 응답 파싱"""
        # 구조화된 정보 추출
        structured_info = self._extract_structured_info(text)
        
        return ParsedResponse(
            content={
                'text': text,
                'structured_info': structured_info
            },
            response_type=ResponseType.TEXT,
            confidence=1.0,
            metadata={'text_length': len(text)},
            raw_content=text
        )
    
    def _attempt_json_recovery(self, text: str) -> Optional[Dict[str, Any]]:
        """JSON 복구 시도"""
        try:
            # 일반적인 JSON 오류 수정 시도
            
            # 1. 후행 쉼표 제거
            fixed_text = re.sub(r',\s*}', '}', text)
            fixed_text = re.sub(r',\s*]', ']', fixed_text)
            
            # 2. 누락된 따옴표 추가 (간단한 경우만)
            fixed_text = re.sub(r'(\w+):', r'"\1":', fixed_text)
            
            # 3. 파싱 시도
            return json.loads(fixed_text)
            
        except:
            # 부분적 정보 추출 시도
            try:
                # 키-값 쌍 추출
                key_value_pattern = r'"([^"]+)":\s*"([^"]*)"'
                matches = re.findall(key_value_pattern, text)
                
                if matches:
                    return dict(matches)
                    
            except:
                pass
        
        return None
    
    def _validate_python_code(self, code: str) -> Dict[str, Any]:
        """Python 코드 검증"""
        try:
            import ast
            ast.parse(code)
            return {'is_valid': True, 'errors': []}
        except SyntaxError as e:
            return {'is_valid': False, 'errors': [f"구문 오류: {str(e)}"]}
        except Exception as e:
            return {'is_valid': False, 'errors': [f"검증 오류: {str(e)}"]}
    
    def _extract_structured_info(self, text: str) -> Dict[str, Any]:
        """텍스트에서 구조화된 정보 추출"""
        info = {}
        
        # 키워드 추출
        keywords = extract_keywords(text)
        info['keywords'] = keywords
        
        # 숫자 추출
        numbers = re.findall(r'\b\d+\.?\d*\b', text)
        info['numbers'] = [float(n) if '.' in n else int(n) for n in numbers]
        
        # 목록 항목 추출
        list_items = re.findall(r'^\s*[-*•]\s*(.+)$', text, re.MULTILINE)
        if list_items:
            info['list_items'] = list_items
        
        # 번호 목록 추출
        numbered_items = re.findall(r'^\s*\d+\.\s*(.+)$', text, re.MULTILINE)
        if numbered_items:
            info['numbered_items'] = numbered_items
        
        # 질문 추출
        questions = re.findall(r'[^.!?]*\?', text)
        if questions:
            info['questions'] = [q.strip() for q in questions]
        
        return info
    
    def extract_specific_data(self, 
                            parsed_response: ParsedResponse,
                            data_type: str,
                            **kwargs) -> Any:
        """
        특정 데이터 타입 추출
        
        Args:
            parsed_response: 파싱된 응답
            data_type: 추출할 데이터 타입
            **kwargs: 추가 옵션
            
        Returns:
            추출된 데이터
        """
        if data_type == "analysis_methods":
            return self._extract_analysis_methods(parsed_response)
        elif data_type == "statistical_results":
            return self._extract_statistical_results(parsed_response)
        elif data_type == "code_blocks":
            return self._extract_code_blocks(parsed_response)
        elif data_type == "recommendations":
            return self._extract_recommendations(parsed_response)
        else:
            raise ParsingException(f"지원하지 않는 데이터 타입: {data_type}")
    
    def _extract_analysis_methods(self, parsed_response: ParsedResponse) -> List[Dict[str, Any]]:
        """분석 방법 추출"""
        methods = []
        
        if parsed_response.response_type == ResponseType.JSON and isinstance(parsed_response.content, dict):
            # JSON에서 분석 방법 추출
            if 'methods' in parsed_response.content:
                methods = parsed_response.content['methods']
            elif 'analysis_methods' in parsed_response.content:
                methods = parsed_response.content['analysis_methods']
        
        elif parsed_response.response_type == ResponseType.TEXT:
            # 텍스트에서 분석 방법 추출
            text = parsed_response.content.get('text', '')
            
            # 패턴 기반 추출
            method_patterns = [
                r'(\w+\s*검정)',
                r'(\w+\s*분석)',
                r'(\w+\s*test)',
                r'(\w+\s*analysis)'
            ]
            
            for pattern in method_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                methods.extend(matches)
        
        return methods
    
    def _extract_statistical_results(self, parsed_response: ParsedResponse) -> Dict[str, Any]:
        """통계 결과 추출"""
        results = {}
        
        if parsed_response.response_type == ResponseType.JSON and isinstance(parsed_response.content, dict):
            # JSON에서 통계 결과 추출
            for key in ['results', 'statistics', 'test_results', 'analysis_results']:
                if key in parsed_response.content:
                    results.update(parsed_response.content[key])
        
        elif parsed_response.response_type == ResponseType.TEXT:
            # 텍스트에서 통계 값 추출
            text = parsed_response.content.get('text', '')
            
            # p-value 추출
            p_values = re.findall(r'p[-\s]*value?\s*[=:]\s*([\d.]+)', text, re.IGNORECASE)
            if p_values:
                results['p_value'] = float(p_values[0])
            
            # t-statistic 추출
            t_stats = re.findall(r't[-\s]*stat(?:istic)?\s*[=:]\s*([-\d.]+)', text, re.IGNORECASE)
            if t_stats:
                results['t_statistic'] = float(t_stats[0])
            
            # F-statistic 추출
            f_stats = re.findall(r'f[-\s]*stat(?:istic)?\s*[=:]\s*([\d.]+)', text, re.IGNORECASE)
            if f_stats:
                results['f_statistic'] = float(f_stats[0])
        
        return results
    
    def _extract_code_blocks(self, parsed_response: ParsedResponse) -> List[str]:
        """코드 블록 추출"""
        code_blocks = []
        
        if parsed_response.response_type == ResponseType.CODE:
            if isinstance(parsed_response.content, list):
                code_blocks = [block['code'] for block in parsed_response.content]
            else:
                code_blocks = [parsed_response.content]
        
        elif parsed_response.response_type == ResponseType.MIXED:
            code_parts = parsed_response.content.get('code_parts', [])
            code_blocks = [part['content'] for part in code_parts]
        
        return code_blocks
    
    def _extract_recommendations(self, parsed_response: ParsedResponse) -> List[str]:
        """권장사항 추출"""
        recommendations = []
        
        if parsed_response.response_type == ResponseType.JSON and isinstance(parsed_response.content, dict):
            # JSON에서 권장사항 추출
            for key in ['recommendations', 'suggestions', 'advice']:
                if key in parsed_response.content:
                    recs = parsed_response.content[key]
                    if isinstance(recs, list):
                        recommendations.extend(recs)
                    else:
                        recommendations.append(str(recs))
        
        elif parsed_response.response_type == ResponseType.TEXT:
            # 텍스트에서 권장사항 추출
            text = parsed_response.content.get('text', '')
            
            # 권장사항 섹션 찾기
            recommendation_patterns = [
                r'권장사항[:\s]*\n(.*?)(?=\n\n|\n[A-Z]|\Z)',
                r'추천[:\s]*\n(.*?)(?=\n\n|\n[A-Z]|\Z)',
                r'제안[:\s]*\n(.*?)(?=\n\n|\n[A-Z]|\Z)'
            ]
            
            for pattern in recommendation_patterns:
                matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    # 목록 항목 추출
                    items = re.findall(r'^\s*[-*•]\s*(.+)$', match, re.MULTILINE)
                    recommendations.extend(items)
        
        return recommendations
    
    def validate_parsed_response(self, 
                                parsed_response: ParsedResponse,
                                validation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """파싱된 응답 검증"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # 신뢰도 검사
        min_confidence = validation_rules.get('min_confidence', 0.5)
        if parsed_response.confidence < min_confidence:
            validation_result['warnings'].append(f"신뢰도가 낮습니다: {parsed_response.confidence}")
        
        # 필수 필드 검사
        required_fields = validation_rules.get('required_fields', [])
        if parsed_response.response_type == ResponseType.JSON and isinstance(parsed_response.content, dict):
            for field in required_fields:
                if field not in parsed_response.content:
                    validation_result['errors'].append(f"필수 필드 누락: {field}")
                    validation_result['is_valid'] = False
        
        # 데이터 타입 검사
        expected_types = validation_rules.get('expected_types', {})
        if parsed_response.response_type == ResponseType.JSON and isinstance(parsed_response.content, dict):
            for field, expected_type in expected_types.items():
                if field in parsed_response.content:
                    actual_value = parsed_response.content[field]
                    if not isinstance(actual_value, expected_type):
                        validation_result['errors'].append(
                            f"필드 타입 불일치: {field} (예상: {expected_type.__name__}, 실제: {type(actual_value).__name__})"
                        )
                        validation_result['is_valid'] = False
        
        return validation_result 