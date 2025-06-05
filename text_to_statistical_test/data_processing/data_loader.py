"""
DataLoader: 데이터 로딩 및 기본 정보 추출

CSV, Excel, Parquet 등 일반적인 형식의 데이터 파일을 로드하고
통계 분석에 필요한 기본 정보를 추출합니다.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import warnings


class DataLoader:
    """
    데이터 로딩 및 기본 정보 추출 클래스
    
    다양한 형식의 데이터 파일을 로드하고, 통계 분석에 필요한
    기본적인 데이터 프로파일링 기능을 제공합니다.
    """
    
    def __init__(self):
        """DataLoader 초기화"""
        self.logger = logging.getLogger(__name__)
        self.supported_formats = {
            '.csv': self._load_csv,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.json': self._load_json,
            '.parquet': self._load_parquet,
            '.pq': self._load_parquet  # parquet 파일의 다른 확장자
        }
    
    def load_data(self, file_path: str, file_type: str = None, **kwargs) -> pd.DataFrame:
        """
        파일 경로와 타입에 따라 적절한 로더를 사용하여 데이터를 로드합니다.
        
        Args:
            file_path: 데이터 파일 경로
            file_type: 파일 타입 (미지정시 확장자로 추론)
            **kwargs: 로더별 추가 옵션
            
        Returns:
            pd.DataFrame: 로드된 데이터
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        # 파일 타입 결정
        if file_type is None:
            file_type = file_path.suffix.lower()
        
        if file_type not in self.supported_formats:
            supported = ', '.join(self.supported_formats.keys())
            raise ValueError(f"지원하지 않는 파일 형식: {file_type}. 지원 형식: {supported}")
        
        self.logger.info(f"데이터 로딩 시작: {file_path} ({file_type})")
        
        try:
            # 적절한 로더 사용
            loader_func = self.supported_formats[file_type]
            dataframe = loader_func(file_path, **kwargs)
            
            # 기본 검증
            if dataframe.empty:
                self.logger.warning("로드된 데이터가 비어있습니다.")
                return dataframe
            
            # 컬럼명 정리 (공백 제거, 특수문자 처리)
            dataframe.columns = dataframe.columns.str.strip()
            
            self.logger.info(f"데이터 로딩 완료: {dataframe.shape[0]}행 {dataframe.shape[1]}열")
            return dataframe
            
        except Exception as e:
            self.logger.error(f"데이터 로딩 실패: {e}")
            raise
    
    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """CSV 파일 로드"""
        default_options = {
            'encoding': 'utf-8',
            'low_memory': False
        }
        default_options.update(kwargs)
        
        try:
            return pd.read_csv(file_path, **default_options)
        except UnicodeDecodeError:
            # 인코딩 문제시 다른 인코딩 시도
            self.logger.warning("UTF-8 인코딩 실패, cp949로 재시도")
            default_options['encoding'] = 'cp949'
            try:
                return pd.read_csv(file_path, **default_options)
            except UnicodeDecodeError:
                # 마지막 시도: 자동 감지
                self.logger.warning("cp949 인코딩도 실패, latin-1로 재시도")
                default_options['encoding'] = 'latin-1'
                return pd.read_csv(file_path, **default_options)
    
    def _load_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Excel 파일 로드"""
        default_options = {
            'engine': 'openpyxl' if file_path.suffix == '.xlsx' else 'xlrd'
        }
        default_options.update(kwargs)
        
        try:
            return pd.read_excel(file_path, **default_options)
        except Exception as e:
            # 다른 엔진으로 재시도
            if default_options['engine'] == 'openpyxl':
                self.logger.warning("openpyxl 엔진 실패, xlrd로 재시도")
                default_options['engine'] = 'xlrd'
            else:
                self.logger.warning("xlrd 엔진 실패, openpyxl로 재시도")
                default_options['engine'] = 'openpyxl'
            return pd.read_excel(file_path, **default_options)
    
    def _load_json(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """JSON 파일 로드"""
        default_options = {
            'orient': 'records'
        }
        default_options.update(kwargs)
        
        try:
            return pd.read_json(file_path, **default_options)
        except ValueError:
            # orient 옵션을 다르게 시도
            for orient in ['index', 'values', 'split', 'table']:
                try:
                    default_options['orient'] = orient
                    self.logger.info(f"JSON 로딩: orient={orient}로 재시도")
                    return pd.read_json(file_path, **default_options)
                except:
                    continue
            raise ValueError("JSON 파일을 로드할 수 없습니다. 파일 형식을 확인해주세요.")
    
    def _load_parquet(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Parquet 파일 로드"""
        try:
            return pd.read_parquet(file_path, **kwargs)
        except ImportError:
            self.logger.error("pyarrow 또는 fastparquet 라이브러리가 설치되지 않았습니다.")
            self.logger.error("다음 명령어로 설치하세요: pip install pyarrow")
            raise
        except Exception as e:
            self.logger.error(f"Parquet 파일 로딩 실패: {e}")
            raise
    
    def get_data_profile(self, dataframe: pd.DataFrame, 
                        unique_threshold: int = 10) -> Dict[str, Any]:
        """
        DataFrame의 각 컬럼에 대한 프로파일링을 수행합니다.
        
        Args:
            dataframe: 분석할 DataFrame
            unique_threshold: 범주형 변수 판단 기준 (고유값 개수)
            
        Returns:
            Dict: 데이터 프로파일 정보
        """
        if dataframe.empty:
            return {"error": "데이터가 비어있습니다."}
        
        profile = {
            "basic_info": self._get_basic_info(dataframe),
            "columns": self._analyze_columns(dataframe, unique_threshold),
            "missing_values": self._analyze_missing_values(dataframe),
            "data_types": self._analyze_data_types(dataframe),
            "summary_statistics": self._get_summary_statistics(dataframe)
        }
        
        return profile
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터프레임 기본 정보"""
        return {
            "shape": df.shape,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "column_names": list(df.columns)
        }
    
    def _analyze_columns(self, df: pd.DataFrame, unique_threshold: int) -> Dict[str, Dict]:
        """컬럼별 상세 분석"""
        column_analysis = {}
        
        for col in df.columns:
            col_data = df[col]
            unique_count = col_data.nunique()
            
            analysis = {
                "data_type": str(col_data.dtype),
                "unique_count": unique_count,
                "null_count": col_data.isnull().sum(),
                "null_percentage": (col_data.isnull().sum() / len(df)) * 100,
                "sample_values": col_data.dropna().head(5).tolist()
            }
            
            # 변수 타입 추정
            analysis["inferred_type"] = self._infer_variable_type(col_data, unique_threshold)
            
            # 숫자형 변수인 경우 추가 통계
            if pd.api.types.is_numeric_dtype(col_data):
                analysis.update(self._get_numeric_stats(col_data))
            
            # 범주형 변수인 경우 빈도 분석
            if unique_count <= unique_threshold and unique_count > 0:
                analysis["value_counts"] = col_data.value_counts().head(10).to_dict()
            
            column_analysis[col] = analysis
        
        return column_analysis
    
    def _infer_variable_type(self, series: pd.Series, unique_threshold: int) -> str:
        """변수 타입을 추정합니다."""
        unique_count = series.nunique()
        
        # 수치형 데이터
        if pd.api.types.is_numeric_dtype(series):
            if unique_count == 2:
                return "binary_numeric"
            elif unique_count <= unique_threshold:
                return "categorical_numeric"
            elif series.dtype in ['int64', 'int32']:
                # 정수형이지만 고유값이 많은 경우
                if unique_count / len(series) > 0.8:
                    return "continuous_integer"
                else:
                    return "discrete_integer"
            else:
                return "continuous"
        
        # 문자형 데이터
        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            if unique_count == 2:
                return "binary_categorical"
            elif unique_count <= unique_threshold:
                return "categorical"
            else:
                return "text"
        
        # 날짜/시간 데이터
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        
        # 불린 데이터
        elif pd.api.types.is_bool_dtype(series):
            return "boolean"
        
        else:
            return "unknown"
    
    def _get_numeric_stats(self, series: pd.Series) -> Dict[str, Any]:
        """수치형 변수의 기술통계량"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            stats = {
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "q25": series.quantile(0.25),
                "q75": series.quantile(0.75),
                "skewness": series.skew(),
                "kurtosis": series.kurtosis()
            }
            
            # NaN 값을 None으로 변환 (JSON 직렬화 가능)
            return {k: (None if pd.isna(v) else v) for k, v in stats.items()}
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """결측값 분석"""
        missing_count = df.isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        
        return {
            "total_missing": missing_count.sum(),
            "columns_with_missing": missing_count[missing_count > 0].to_dict(),
            "missing_percentages": missing_percentage[missing_percentage > 0].to_dict(),
            "complete_rows": len(df) - df.isnull().any(axis=1).sum()
        }
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """데이터 타입별 컬럼 분류"""
        type_groups = {
            "numeric": [],
            "categorical": [],
            "datetime": [],
            "boolean": [],
            "object": []
        }
        
        for col in df.columns:
            dtype = df[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                type_groups["numeric"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                type_groups["datetime"].append(col)
            elif pd.api.types.is_bool_dtype(dtype):
                type_groups["boolean"].append(col)
            else:
                # 문자형이지만 고유값이 적으면 범주형으로 분류
                if df[col].nunique() <= 20:
                    type_groups["categorical"].append(col)
                else:
                    type_groups["object"].append(col)
        
        return type_groups
    
    def _get_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """전체 데이터 요약 통계"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {"message": "수치형 변수가 없습니다."}
        
        summary = df[numeric_cols].describe()
        
        return {
            "numeric_summary": summary.to_dict(),
            "correlations": df[numeric_cols].corr().to_dict() if len(numeric_cols) > 1 else {}
        }
    
    def validate_data_for_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """통계 분석을 위한 데이터 유효성 검사"""
        issues = []
        recommendations = []
        
        # 최소 데이터 크기 확인
        if len(df) < 30:
            issues.append("표본 크기가 30 미만입니다.")
            recommendations.append("더 많은 데이터 수집을 권장합니다.")
        
        # 결측값 확인
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        high_missing_cols = missing_percentage[missing_percentage > 50].index.tolist()
        
        if high_missing_cols:
            issues.append(f"결측값이 50% 이상인 컬럼: {high_missing_cols}")
            recommendations.append("해당 컬럼 제거 또는 고급 결측값 처리 기법 적용을 고려하세요.")
        
        # 중복 행 확인
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"중복된 행: {duplicate_count}개")
            recommendations.append("중복 행 제거를 고려하세요.")
        
        # 수치형 변수 확인
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            issues.append("수치형 변수가 없습니다.")
            recommendations.append("범주형 변수에 대한 카이제곱 검정 등을 고려하세요.")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations
        }
    
    def suggest_analysis_types(self, df: pd.DataFrame) -> List[str]:
        """데이터 특성에 따른 분석 방법 제안"""
        suggestions = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # 수치형 변수 기반 제안
        if len(numeric_cols) >= 2:
            suggestions.append("상관분석")
            suggestions.append("회귀분석")
        
        if len(numeric_cols) >= 1:
            suggestions.append("일표본 t-검정")
            suggestions.append("정규성 검정")
        
        # 범주형 변수 기반 제안
        if len(categorical_cols) >= 1:
            if len(numeric_cols) >= 1:
                suggestions.append("독립표본 t-검정")
                suggestions.append("ANOVA")
            
            if len(categorical_cols) >= 2:
                suggestions.append("카이제곱 검정")
        
        return suggestions 