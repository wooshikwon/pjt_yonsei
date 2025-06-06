"""
입력 검증 유틸리티

사용자 입력 및 데이터 유효성 검사를 수행하는 유틸리티
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class InputValidator:
    """
    사용자 입력 및 데이터 유효성 검사 클래스
    """
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_file_path(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        파일 경로 유효성 검증
        
        Args:
            file_path: 검증할 파일 경로
            
        Returns:
            Dict: 검증 결과
        """
        result = {
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'file_info': {}
        }
        
        try:
            path = Path(file_path)
            
            # 경로 존재 확인
            if not path.exists():
                result['errors'].append(f"파일이 존재하지 않습니다: {file_path}")
                return result
            
            # 파일인지 확인
            if not path.is_file():
                result['errors'].append(f"경로가 파일이 아닙니다: {file_path}")
                return result
            
            # 읽기 권한 확인
            if not path.stat().st_mode & 0o444:
                result['errors'].append(f"파일 읽기 권한이 없습니다: {file_path}")
                return result
            
            # 파일 크기 확인
            file_size = path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            if file_size == 0:
                result['errors'].append("파일이 비어있습니다")
                return result
            
            if file_size_mb > 100:  # 100MB 제한
                result['warnings'].append(f"파일 크기가 큽니다: {file_size_mb:.1f}MB")
            
            # 파일 확장자 확인
            supported_extensions = ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.tsv', '.txt']
            if path.suffix.lower() not in supported_extensions:
                result['errors'].append(f"지원하지 않는 파일 형식입니다: {path.suffix}")
                return result
            
            result['file_info'] = {
                'name': path.name,
                'size_mb': round(file_size_mb, 2),
                'extension': path.suffix.lower(),
                'absolute_path': str(path.absolute())
            }
            
            result['is_valid'] = True
            logger.info(f"파일 경로 검증 성공: {file_path}")
            
        except Exception as e:
            result['errors'].append(f"파일 경로 검증 중 오류: {str(e)}")
            logger.error(f"파일 경로 검증 오류: {file_path}, {str(e)}")
        
        return result
    
    def validate_dataframe(self, df: pd.DataFrame, min_rows: int = 1, min_cols: int = 1) -> Dict[str, Any]:
        """
        데이터프레임 유효성 검증
        
        Args:
            df: 검증할 데이터프레임
            min_rows: 최소 행 수
            min_cols: 최소 열 수
            
        Returns:
            Dict: 검증 결과
        """
        result = {
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'data_info': {}
        }
        
        try:
            # 기본 검증
            if df is None:
                result['errors'].append("데이터프레임이 None입니다")
                return result
            
            if df.empty:
                result['errors'].append("데이터프레임이 비어있습니다")
                return result
            
            # 크기 검증
            if len(df) < min_rows:
                result['errors'].append(f"데이터 행 수가 부족합니다 (현재: {len(df)}, 최소: {min_rows})")
                return result
            
            if len(df.columns) < min_cols:
                result['errors'].append(f"데이터 열 수가 부족합니다 (현재: {len(df.columns)}, 최소: {min_cols})")
                return result
            
            # 경고 사항 확인
            missing_percentage = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_percentage > 50:
                result['warnings'].append(f"결측값이 많습니다: {missing_percentage:.1f}%")
            
            duplicate_rows = df.duplicated().sum()
            if duplicate_rows > 0:
                result['warnings'].append(f"중복된 행이 있습니다: {duplicate_rows}개")
            
            # 컬럼명 검증
            invalid_columns = []
            for col in df.columns:
                if pd.isna(col) or str(col).strip() == '':
                    invalid_columns.append(col)
            
            if invalid_columns:
                result['warnings'].append(f"유효하지 않은 컬럼명이 있습니다: {invalid_columns}")
            
            result['data_info'] = {
                'shape': df.shape,
                'missing_percentage': round(missing_percentage, 2),
                'duplicate_rows': duplicate_rows,
                'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
                'datetime_columns': list(df.select_dtypes(include=['datetime']).columns)
            }
            
            result['is_valid'] = True
            logger.info(f"데이터프레임 검증 성공: {df.shape}")
            
        except Exception as e:
            result['errors'].append(f"데이터프레임 검증 중 오류: {str(e)}")
            logger.error(f"데이터프레임 검증 오류: {str(e)}")
        
        return result
    
    def validate_column_selection(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """
        컬럼 선택 유효성 검증
        
        Args:
            df: 데이터프레임
            columns: 선택된 컬럼 목록
            
        Returns:
            Dict: 검증 결과
        """
        result = {
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'valid_columns': [],
            'invalid_columns': []
        }
        
        try:
            if not columns:
                result['errors'].append("선택된 컬럼이 없습니다")
                return result
            
            available_columns = list(df.columns)
            
            for col in columns:
                if col in available_columns:
                    result['valid_columns'].append(col)
                else:
                    result['invalid_columns'].append(col)
            
            if result['invalid_columns']:
                result['errors'].append(f"존재하지 않는 컬럼: {result['invalid_columns']}")
                return result
            
            # 선택된 컬럼들의 데이터 품질 확인
            for col in result['valid_columns']:
                missing_pct = (df[col].isna().sum() / len(df)) * 100
                if missing_pct > 80:
                    result['warnings'].append(f"컬럼 '{col}'의 결측값이 많습니다: {missing_pct:.1f}%")
                
                if df[col].dtype == 'object' and df[col].nunique() > len(df) * 0.8:
                    result['warnings'].append(f"컬럼 '{col}'의 고유값이 너무 많습니다 (대부분 고유함)")
            
            result['is_valid'] = True
            logger.info(f"컬럼 선택 검증 성공: {result['valid_columns']}")
            
        except Exception as e:
            result['errors'].append(f"컬럼 선택 검증 중 오류: {str(e)}")
            logger.error(f"컬럼 선택 검증 오류: {str(e)}")
        
        return result
    
    def validate_statistical_test_requirements(
        self, 
        df: pd.DataFrame, 
        test_type: str, 
        columns: List[str]
    ) -> Dict[str, Any]:
        """
        통계 검정 요구사항 검증
        
        Args:
            df: 데이터프레임
            test_type: 검정 유형
            columns: 분석할 컬럼들
            
        Returns:
            Dict: 검증 결과
        """
        result = {
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'requirements_met': {}
        }
        
        try:
            # 기본 요구사항
            min_requirements = {
                't_test': {'min_rows': 15, 'required_columns': 1, 'data_types': ['numeric']},
                'anova': {'min_rows': 20, 'required_columns': 2, 'data_types': ['numeric', 'categorical']},
                'chi_square': {'min_rows': 50, 'required_columns': 2, 'data_types': ['categorical']},
                'correlation': {'min_rows': 30, 'required_columns': 2, 'data_types': ['numeric']},
                'regression': {'min_rows': 50, 'required_columns': 2, 'data_types': ['numeric']}
            }
            
            if test_type not in min_requirements:
                result['errors'].append(f"지원하지 않는 검정 유형: {test_type}")
                return result
            
            requirements = min_requirements[test_type]
            
            # 행 수 확인
            if len(df) < requirements['min_rows']:
                result['errors'].append(
                    f"{test_type}에 필요한 최소 행 수 부족 "
                    f"(현재: {len(df)}, 필요: {requirements['min_rows']})"
                )
            else:
                result['requirements_met']['min_rows'] = True
            
            # 컬럼 수 확인
            if len(columns) < requirements['required_columns']:
                result['errors'].append(
                    f"{test_type}에 필요한 최소 컬럼 수 부족 "
                    f"(현재: {len(columns)}, 필요: {requirements['required_columns']})"
                )
            else:
                result['requirements_met']['required_columns'] = True
            
            # 데이터 타입 확인
            numeric_cols = df[columns].select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df[columns].select_dtypes(include=['object', 'category']).columns.tolist()
            
            required_types = requirements['data_types']
            type_requirements_met = True
            
            if 'numeric' in required_types and not numeric_cols:
                result['errors'].append(f"{test_type}에 필요한 수치형 컬럼이 없습니다")
                type_requirements_met = False
            
            if 'categorical' in required_types and not categorical_cols:
                result['errors'].append(f"{test_type}에 필요한 범주형 컬럼이 없습니다")
                type_requirements_met = False
            
            result['requirements_met']['data_types'] = type_requirements_met
            
            # 추가 검증 (검정별 특수 요구사항)
            if test_type == 'chi_square':
                # 카이제곱 검정의 경우 기댓값 조건 확인
                for col in categorical_cols:
                    value_counts = df[col].value_counts()
                    if any(count < 5 for count in value_counts):
                        result['warnings'].append(
                            f"컬럼 '{col}'에서 빈도가 5 미만인 범주가 있습니다 "
                            "(카이제곱 검정 가정 위배 가능성)"
                        )
            
            # 모든 요구사항이 충족되었는지 확인
            if not result['errors']:
                result['is_valid'] = True
                logger.info(f"통계 검정 요구사항 검증 성공: {test_type}")
            
        except Exception as e:
            result['errors'].append(f"통계 검정 요구사항 검증 중 오류: {str(e)}")
            logger.error(f"통계 검정 요구사항 검증 오류: {str(e)}")
        
        return result
    
    def validate_user_input(self, user_input: str, input_type: str = 'general') -> Dict[str, Any]:
        """
        사용자 입력 유효성 검증
        
        Args:
            user_input: 사용자 입력 텍스트
            input_type: 입력 유형 ('general', 'analysis_goal', 'column_name' 등)
            
        Returns:
            Dict: 검증 결과
        """
        result = {
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'cleaned_input': ''
        }
        
        try:
            if not user_input or not user_input.strip():
                result['errors'].append("입력이 비어있습니다")
                return result
            
            cleaned_input = user_input.strip()
            
            # 길이 검증
            if len(cleaned_input) > 1000:
                result['warnings'].append("입력이 너무 깁니다 (1000자 초과)")
                cleaned_input = cleaned_input[:1000]
            
            # 특수 문자 검증 (입력 유형별)
            if input_type == 'column_name':
                # 컬럼명의 경우 특수 문자 제한
                if re.search(r'[^\w\s가-힣-]', cleaned_input):
                    result['warnings'].append("컬럼명에 특수 문자가 포함되어 있습니다")
            
            elif input_type == 'analysis_goal':
                # 분석 목표의 경우 기본적인 텍스트 검증
                if len(cleaned_input) < 5:
                    result['warnings'].append("분석 목표가 너무 짧습니다")
            
            # SQL 인젝션 패턴 확인
            sql_patterns = ['drop', 'delete', 'insert', 'update', 'select', '--', ';']
            if any(pattern in cleaned_input.lower() for pattern in sql_patterns):
                result['warnings'].append("입력에 의심스러운 패턴이 포함되어 있습니다")
            
            result['cleaned_input'] = cleaned_input
            result['is_valid'] = True
            logger.info(f"사용자 입력 검증 성공: {input_type}")
            
        except Exception as e:
            result['errors'].append(f"사용자 입력 검증 중 오류: {str(e)}")
            logger.error(f"사용자 입력 검증 오류: {str(e)}")
        
        return result

# 전역 인스턴스
input_validator = InputValidator() 