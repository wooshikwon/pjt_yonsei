"""
Data Selection Pipeline

1단계: 데이터 파일 선택 및 초기 이해
사용자가 input_data/data_files/ 폴더에서 분석할 데이터 파일을 선택하고
데이터의 기본적인 메타정보를 파악합니다.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import os

from .base_pipeline_step import BasePipelineStep, PipelineStepRegistry
from utils.data_utils import get_available_data_files, validate_file_access
from utils.ui_helpers import display_file_selection_menu
from utils.data_loader import DataLoader


class DataSelectionStep(BasePipelineStep):
    """1단계: 데이터 파일 선택 및 초기 이해"""
    
    def __init__(self):
        """DataSelectionStep 초기화"""
        super().__init__("데이터 파일 선택 및 초기 이해", 1)
        self.data_loader = DataLoader()
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        입력 데이터 유효성 검증
        
        Args:
            input_data: 입력 데이터 (초기 단계이므로 빈 딕셔너리 또는 선택적 파라미터)
            
        Returns:
            bool: 유효성 검증 결과
        """
        # 1단계는 초기 단계이므로 특별한 입력 요구사항 없음
        return isinstance(input_data, dict)
    
    def get_expected_output_schema(self) -> Dict[str, Any]:
        """
        예상 출력 스키마 반환
        
        Returns:
            Dict[str, Any]: 출력 데이터 스키마
        """
        return {
            'selected_file': str,
            'file_info': {
                'file_name': str,
                'file_size': int,
                'file_extension': str,
                'row_count': int,
                'column_count': int,
                'columns': list,
                'data_types': dict
            },
            'data_preview': dict,
            'metadata': {
                'file_path': str,
                'encoding': str,
                'delimiter': str
            },
            'data_object': object  # 실제 pandas DataFrame 객체
        }
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        데이터 파일 선택 파이프라인 실행
        
        Args:
            input_data: 파이프라인 실행 컨텍스트
                - file_path (optional): 직접 지정할 파일 경로
                - file_number (optional): 파일 목록에서 선택할 번호
                - interactive (optional): 대화형 선택 여부 (기본값: True)
            
        Returns:
            Dict: 실행 결과
        """
        self.logger.info("1단계: 데이터 파일 선택 및 초기 이해 시작")
        
        try:
            # 직접 파일 경로가 지정된 경우
            if 'file_path' in input_data and input_data['file_path']:
                return self._execute_with_file_path(input_data['file_path'])
            
            # 파일 번호가 지정된 경우
            if 'file_number' in input_data and input_data['file_number']:
                return self._execute_with_file_number(input_data['file_number'])
            
            # 대화형 선택 (기본값)
            interactive = input_data.get('interactive', True)
            if interactive:
                return self._execute_interactive()
            
            # 아무 조건도 맞지 않으면 오류
            return {
                'error': True,
                'error_message': '파일 선택 방법이 지정되지 않았습니다.',
                'suggestions': [
                    'file_path를 지정하거나',
                    'file_number를 지정하거나',
                    'interactive=True로 설정하세요'
                ]
            }
                
        except Exception as e:
            self.logger.error(f"데이터 선택 파이프라인 오류: {e}")
            return {
                'error': True,
                'error_message': str(e),
                'error_type': 'pipeline_error'
            }
    
    def _execute_interactive(self) -> Dict[str, Any]:
        """대화형 파일 선택 실행"""
        # 사용 가능한 데이터 파일 검색
        data_files = get_available_data_files()
        
        if not data_files:
            return self._handle_no_data_files()
        
        # UI를 통한 파일 선택
        selected_file = self._select_file_interactive(data_files)
        
        if not selected_file:
            return {
                'error': True,
                'error_message': '파일 선택이 취소되었습니다.',
                'cancelled': True
            }
        
        return self._validate_and_analyze_file(selected_file)
    
    def _execute_with_file_path(self, file_path: str) -> Dict[str, Any]:
        """
        직접 파일 경로로 데이터 선택
        
        Args:
            file_path: 선택할 파일 경로
            
        Returns:
            Dict: 실행 결과
        """
        self.logger.info(f"데이터 파일 직접 선택: {file_path}")
        
        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            return {
                'error': True,
                'error_message': f'파일을 찾을 수 없습니다: {file_path}',
                'error_type': 'file_not_found'
            }
        
        return self._validate_and_analyze_file(file_path)
    
    def _execute_with_file_number(self, file_number: int) -> Dict[str, Any]:
        """
        파일 번호로 데이터 선택
        
        Args:
            file_number: 선택할 파일 번호 (1부터 시작)
            
        Returns:
            Dict: 실행 결과
        """
        data_files = get_available_data_files()
        
        if not data_files:
            return self._handle_no_data_files()
        
        if file_number < 1 or file_number > len(data_files):
            return {
                'error': True,
                'error_message': f'1-{len(data_files)} 범위의 번호를 입력해주세요.',
                'error_type': 'invalid_selection',
                'available_range': f'1-{len(data_files)}'
            }
        
        selected_file = data_files[file_number - 1]
        return self._validate_and_analyze_file(selected_file)
    
    def _handle_no_data_files(self) -> Dict[str, Any]:
        """데이터 파일이 없을 때 처리"""
        error_message = [
            "분석할 데이터 파일이 없습니다.",
            "input_data/data_files/ 디렉토리에 데이터 파일을 추가해주세요.",
            "지원 형식: CSV, Excel, JSON, Parquet, TSV"
        ]
        
        return {
            'error': True,
            'error_message': '\n'.join(error_message),
            'error_type': 'no_data_files',
            'supported_formats': ['CSV', 'Excel', 'JSON', 'Parquet', 'TSV'],
            'required_action': 'input_data/data_files/ 디렉토리에 데이터 파일 추가'
        }
    
    def _select_file_interactive(self, data_files: List[str]) -> Optional[str]:
        """대화형 파일 선택"""
        try:
            return display_file_selection_menu(data_files)
        except KeyboardInterrupt:
            self.logger.info("사용자가 파일 선택을 취소했습니다.")
            return None
        except Exception as e:
            self.logger.error(f"대화형 파일 선택 오류: {e}")
            return None
    
    def _validate_and_analyze_file(self, file_path: str) -> Dict[str, Any]:
        """파일 검증 및 기본 분석"""
        try:
            # 파일 유효성 검증
            validation = validate_file_access(file_path)
            if not validation["is_valid"]:
                return {
                    "success": False,
                    "error": f"파일 검증 실패: {validation['error']}",
                    "file_path": file_path
                }

            # 데이터 로딩 시도 (올바른 메서드명 사용)
            data, metadata = self.data_loader.load_file(file_path)
            if data is None:
                error_msg = metadata.get('error', '데이터 로딩 실패')
                return {
                    "success": False,
                    "error": error_msg,
                    "file_path": file_path
                }
            
            # 파일 정보 수집
            file_info = self._collect_file_info(file_path, data)
            
            # 데이터 미리보기 생성
            data_preview = self._generate_data_preview(data)
            
            # 메타데이터 수집 (로더의 메타데이터와 병합)
            file_metadata = self._collect_metadata(file_path, data)
            file_metadata.update(metadata)  # 로더의 메타데이터 추가
            
            self.logger.info(f"데이터 파일 분석 완료: {file_path}")
            
            return {
                'selected_file': file_path,
                'file_info': file_info,
                'data_preview': data_preview,
                'metadata': file_metadata,
                'data_object': data,
                'success_message': f"✅ 파일 선택 및 분석 완료: {Path(file_path).name}"
            }
            
        except Exception as e:
            self.logger.error(f"파일 검증 및 분석 오류: {e}")
            return {
                'error': True,
                'error_message': f'파일 분석 중 오류: {str(e)}',
                'error_type': 'analysis_error'
            }
    
    def _collect_file_info(self, file_path: str, data) -> Dict[str, Any]:
        """파일 기본 정보 수집"""
        file_stat = os.stat(file_path)
        
        return {
            'file_name': Path(file_path).name,
            'file_size': file_stat.st_size,
            'file_extension': Path(file_path).suffix,
            'row_count': len(data),
            'column_count': len(data.columns),
            'columns': list(data.columns),
            'data_types': {col: str(dtype) for col, dtype in data.dtypes.items()}
        }
    
    def _generate_data_preview(self, data, rows: int = 5) -> Dict[str, Any]:
        """데이터 미리보기 생성"""
        return {
            'head': data.head(rows).to_dict('records'),
            'tail': data.tail(rows).to_dict('records'),
            'sample': data.sample(min(rows, len(data))).to_dict('records') if len(data) > 0 else []
        }
    
    def _collect_metadata(self, file_path: str, data) -> Dict[str, Any]:
        """메타데이터 수집"""
        metadata = {
            'file_path': file_path,
            'encoding': 'utf-8',  # 기본값, 실제로는 감지 로직 필요
            'delimiter': ','  # 기본값, 실제로는 파일 형식에 따라 결정
        }
        
        # CSV 파일인 경우 구분자 감지 시도
        if file_path.endswith('.csv'):
            try:
                # 간단한 구분자 감지 로직
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    if '\t' in first_line:
                        metadata['delimiter'] = '\t'
                    elif ';' in first_line:
                        metadata['delimiter'] = ';'
            except:
                pass  # 기본값 유지
        
        return metadata
    
    def get_available_files(self) -> List[str]:
        """사용 가능한 파일 목록 반환"""
        return get_available_data_files()
    
    def get_step_info(self) -> Dict[str, Any]:
        """단계 정보 반환 (부모 클래스 메서드 확장)"""
        base_info = super().get_step_info()
        base_info.update({
            'description': '데이터 파일 선택 및 초기 이해',
            'input_requirements': ['선택적: file_path, file_number, interactive'],
            'output_provides': ['selected_file', 'file_info', 'data_preview', 'metadata', 'data_object']
        })
        return base_info


