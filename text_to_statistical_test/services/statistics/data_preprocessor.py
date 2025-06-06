"""
Data Preprocessor

데이터 전처리 서비스
- 결측값 처리
- 이상치 탐지 및 처리
- 데이터 변환 및 정규화
- 데이터 검증
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

from utils.error_handler import ErrorHandler, DataProcessingException
from utils.helpers import safe_divide

logger = logging.getLogger(__name__)

class MissingValueStrategy(Enum):
    """결측값 처리 전략"""
    DROP = "drop"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    INTERPOLATE = "interpolate"
    CONSTANT = "constant"

class OutlierMethod(Enum):
    """이상치 탐지 방법"""
    IQR = "iqr"
    Z_SCORE = "z_score"
    MODIFIED_Z_SCORE = "modified_z_score"
    ISOLATION_FOREST = "isolation_forest"

class ScalingMethod(Enum):
    """스케일링 방법"""
    STANDARD = "standard"
    MIN_MAX = "min_max"
    ROBUST = "robust"
    QUANTILE = "quantile"

@dataclass
class PreprocessingConfig:
    """전처리 설정"""
    missing_value_strategy: MissingValueStrategy = MissingValueStrategy.MEAN
    outlier_method: OutlierMethod = OutlierMethod.IQR
    scaling_method: Optional[ScalingMethod] = None
    outlier_threshold: float = 1.5
    z_score_threshold: float = 3.0
    constant_fill_value: Any = 0
    categorical_encoding: str = "label"  # "label", "onehot", "target"
    remove_duplicates: bool = True
    validate_data: bool = True

@dataclass
class PreprocessingResult:
    """전처리 결과"""
    data: pd.DataFrame
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    missing_values_handled: Dict[str, int]
    outliers_detected: Dict[str, int]
    transformations_applied: List[str]
    metadata: Dict[str, Any]

class DataPreprocessor:
    """데이터 전처리 메인 클래스"""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        전처리기 초기화
        
        Args:
            config: 전처리 설정
        """
        self.config = config or PreprocessingConfig()
        self.error_handler = ErrorHandler()
        
        # 변환 히스토리
        self.transformation_history = []
        
        # 스케일러 저장 (역변환용)
        self.scalers = {}
        
        logger.info("데이터 전처리기 초기화 완료")
    
    def preprocess(self, 
                  data: pd.DataFrame,
                  target_column: Optional[str] = None,
                  config: Optional[PreprocessingConfig] = None) -> PreprocessingResult:
        """
        데이터 전처리 실행
        
        Args:
            data: 원본 데이터
            target_column: 타겟 컬럼명
            config: 전처리 설정 (없으면 기본 설정 사용)
            
        Returns:
            PreprocessingResult: 전처리 결과
        """
        try:
            # 설정 업데이트
            if config:
                self.config = config
            
            # 원본 데이터 복사
            processed_data = data.copy()
            original_shape = data.shape
            
            # 전처리 단계별 실행
            result_metadata = {
                'target_column': target_column,
                'original_dtypes': data.dtypes.to_dict(),
                'processing_steps': []
            }
            
            missing_values_handled = {}
            outliers_detected = {}
            transformations_applied = []
            
            # 1. 데이터 검증
            if self.config.validate_data:
                validation_result = self._validate_data(processed_data)
                result_metadata['validation'] = validation_result
                transformations_applied.append("data_validation")
            
            # 2. 중복 제거
            if self.config.remove_duplicates:
                before_count = len(processed_data)
                processed_data = processed_data.drop_duplicates()
                after_count = len(processed_data)
                
                if before_count != after_count:
                    result_metadata['duplicates_removed'] = before_count - after_count
                    transformations_applied.append("duplicate_removal")
            
            # 3. 결측값 처리
            missing_info = self._handle_missing_values(processed_data, target_column)
            processed_data = missing_info['data']
            missing_values_handled = missing_info['handled_counts']
            if missing_values_handled:
                transformations_applied.append("missing_value_handling")
            
            # 4. 이상치 탐지 및 처리
            outlier_info = self._handle_outliers(processed_data, target_column)
            processed_data = outlier_info['data']
            outliers_detected = outlier_info['detected_counts']
            if outliers_detected:
                transformations_applied.append("outlier_handling")
            
            # 5. 범주형 변수 인코딩
            encoding_info = self._encode_categorical_variables(processed_data, target_column)
            processed_data = encoding_info['data']
            if encoding_info['encoded_columns']:
                result_metadata['encoded_columns'] = encoding_info['encoded_columns']
                transformations_applied.append("categorical_encoding")
            
            # 6. 스케일링
            if self.config.scaling_method:
                scaling_info = self._scale_features(processed_data, target_column)
                processed_data = scaling_info['data']
                self.scalers = scaling_info['scalers']
                result_metadata['scaling_info'] = scaling_info['metadata']
                transformations_applied.append("feature_scaling")
            
            final_shape = processed_data.shape
            
            # 결과 생성
            result = PreprocessingResult(
                data=processed_data,
                original_shape=original_shape,
                final_shape=final_shape,
                missing_values_handled=missing_values_handled,
                outliers_detected=outliers_detected,
                transformations_applied=transformations_applied,
                metadata=result_metadata
            )
            
            logger.info(f"데이터 전처리 완료: {original_shape} -> {final_shape}")
            return result
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'data_shape': data.shape})
            raise DataProcessingException(f"데이터 전처리 실패: {error_info['message']}")
    
    def _validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """데이터 검증"""
        validation_result = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # 기본 검증
        if data.empty:
            validation_result['is_valid'] = False
            validation_result['issues'].append("데이터가 비어있습니다")
            return validation_result
        
        # 컬럼명 검증
        duplicate_columns = data.columns[data.columns.duplicated()].tolist()
        if duplicate_columns:
            validation_result['warnings'].append(f"중복된 컬럼명: {duplicate_columns}")
        
        # 데이터 타입 검증
        for col in data.columns:
            if data[col].dtype == 'object':
                # 숫자로 변환 가능한지 확인
                try:
                    pd.to_numeric(data[col], errors='raise')
                    validation_result['warnings'].append(f"컬럼 '{col}'은 숫자로 변환 가능합니다")
                except:
                    pass
        
        # 메모리 사용량 확인
        memory_usage = data.memory_usage(deep=True).sum() / 1024**2  # MB
        if memory_usage > 1000:  # 1GB
            validation_result['warnings'].append(f"메모리 사용량이 큽니다: {memory_usage:.1f}MB")
        
        return validation_result
    
    def _handle_missing_values(self, 
                              data: pd.DataFrame, 
                              target_column: Optional[str] = None) -> Dict[str, Any]:
        """결측값 처리"""
        handled_counts = {}
        processed_data = data.copy()
        
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            if missing_count == 0:
                continue
            
            handled_counts[column] = missing_count
            
            # 타겟 컬럼은 특별 처리
            if column == target_column:
                # 타겟 컬럼의 결측값이 있는 행 제거
                processed_data = processed_data.dropna(subset=[column])
                continue
            
            # 전략별 처리
            if self.config.missing_value_strategy == MissingValueStrategy.DROP:
                processed_data = processed_data.dropna(subset=[column])
            
            elif self.config.missing_value_strategy == MissingValueStrategy.MEAN:
                if pd.api.types.is_numeric_dtype(data[column]):
                    fill_value = data[column].mean()
                    processed_data[column].fillna(fill_value, inplace=True)
                else:
                    # 범주형은 최빈값으로
                    fill_value = data[column].mode().iloc[0] if not data[column].mode().empty else "Unknown"
                    processed_data[column].fillna(fill_value, inplace=True)
            
            elif self.config.missing_value_strategy == MissingValueStrategy.MEDIAN:
                if pd.api.types.is_numeric_dtype(data[column]):
                    fill_value = data[column].median()
                    processed_data[column].fillna(fill_value, inplace=True)
                else:
                    fill_value = data[column].mode().iloc[0] if not data[column].mode().empty else "Unknown"
                    processed_data[column].fillna(fill_value, inplace=True)
            
            elif self.config.missing_value_strategy == MissingValueStrategy.MODE:
                fill_value = data[column].mode().iloc[0] if not data[column].mode().empty else "Unknown"
                processed_data[column].fillna(fill_value, inplace=True)
            
            elif self.config.missing_value_strategy == MissingValueStrategy.FORWARD_FILL:
                processed_data[column].fillna(method='ffill', inplace=True)
            
            elif self.config.missing_value_strategy == MissingValueStrategy.BACKWARD_FILL:
                processed_data[column].fillna(method='bfill', inplace=True)
            
            elif self.config.missing_value_strategy == MissingValueStrategy.INTERPOLATE:
                if pd.api.types.is_numeric_dtype(data[column]):
                    processed_data[column].interpolate(inplace=True)
                else:
                    processed_data[column].fillna(method='ffill', inplace=True)
            
            elif self.config.missing_value_strategy == MissingValueStrategy.CONSTANT:
                processed_data[column].fillna(self.config.constant_fill_value, inplace=True)
        
        return {
            'data': processed_data,
            'handled_counts': handled_counts
        }
    
    def _handle_outliers(self, 
                        data: pd.DataFrame, 
                        target_column: Optional[str] = None) -> Dict[str, Any]:
        """이상치 탐지 및 처리"""
        detected_counts = {}
        processed_data = data.copy()
        
        # 수치형 컬럼만 처리
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numeric_columns:
            numeric_columns.remove(target_column)  # 타겟 컬럼 제외
        
        for column in numeric_columns:
            outliers_mask = self._detect_outliers(data[column])
            outlier_count = outliers_mask.sum()
            
            if outlier_count > 0:
                detected_counts[column] = outlier_count
                
                # 이상치 처리 (현재는 제거만 구현)
                processed_data = processed_data[~outliers_mask]
        
        return {
            'data': processed_data,
            'detected_counts': detected_counts
        }
    
    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """이상치 탐지"""
        if self.config.outlier_method == OutlierMethod.IQR:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.config.outlier_threshold * IQR
            upper_bound = Q3 + self.config.outlier_threshold * IQR
            return (series < lower_bound) | (series > upper_bound)
        
        elif self.config.outlier_method == OutlierMethod.Z_SCORE:
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > self.config.z_score_threshold
        
        elif self.config.outlier_method == OutlierMethod.MODIFIED_Z_SCORE:
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad
            return np.abs(modified_z_scores) > self.config.z_score_threshold
        
        elif self.config.outlier_method == OutlierMethod.ISOLATION_FOREST:
            try:
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(series.values.reshape(-1, 1))
                return outliers == -1
            except ImportError:
                logger.warning("scikit-learn이 설치되지 않아 IQR 방법을 사용합니다")
                return self._detect_outliers_iqr(series)
        
        return pd.Series([False] * len(series), index=series.index)
    
    def _encode_categorical_variables(self, 
                                    data: pd.DataFrame, 
                                    target_column: Optional[str] = None) -> Dict[str, Any]:
        """범주형 변수 인코딩"""
        processed_data = data.copy()
        encoded_columns = {}
        
        # 범주형 컬럼 식별
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_column and target_column in categorical_columns:
            categorical_columns.remove(target_column)  # 타겟 컬럼 제외
        
        for column in categorical_columns:
            unique_count = data[column].nunique()
            
            # 고유값이 너무 많으면 스킵
            if unique_count > 50:
                logger.warning(f"컬럼 '{column}'의 고유값이 너무 많습니다 ({unique_count}개). 인코딩을 스킵합니다.")
                continue
            
            if self.config.categorical_encoding == "label":
                # 라벨 인코딩
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                processed_data[column] = le.fit_transform(data[column].astype(str))
                encoded_columns[column] = {
                    'method': 'label',
                    'classes': le.classes_.tolist()
                }
            
            elif self.config.categorical_encoding == "onehot":
                # 원핫 인코딩
                dummies = pd.get_dummies(data[column], prefix=column)
                processed_data = processed_data.drop(column, axis=1)
                processed_data = pd.concat([processed_data, dummies], axis=1)
                encoded_columns[column] = {
                    'method': 'onehot',
                    'new_columns': dummies.columns.tolist()
                }
        
        return {
            'data': processed_data,
            'encoded_columns': encoded_columns
        }
    
    def _scale_features(self, 
                       data: pd.DataFrame, 
                       target_column: Optional[str] = None) -> Dict[str, Any]:
        """특성 스케일링"""
        processed_data = data.copy()
        scalers = {}
        
        # 수치형 컬럼만 스케일링
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numeric_columns:
            numeric_columns.remove(target_column)  # 타겟 컬럼 제외
        
        if not numeric_columns:
            return {
                'data': processed_data,
                'scalers': scalers,
                'metadata': {'scaled_columns': []}
            }
        
        try:
            if self.config.scaling_method == ScalingMethod.STANDARD:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
            
            elif self.config.scaling_method == ScalingMethod.MIN_MAX:
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            
            elif self.config.scaling_method == ScalingMethod.ROBUST:
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            
            elif self.config.scaling_method == ScalingMethod.QUANTILE:
                from sklearn.preprocessing import QuantileTransformer
                scaler = QuantileTransformer()
            
            # 스케일링 적용
            scaled_data = scaler.fit_transform(data[numeric_columns])
            processed_data[numeric_columns] = scaled_data
            
            scalers['feature_scaler'] = {
                'scaler': scaler,
                'columns': numeric_columns
            }
            
        except ImportError:
            logger.warning("scikit-learn이 설치되지 않아 스케일링을 스킵합니다")
            numeric_columns = []
        
        return {
            'data': processed_data,
            'scalers': scalers,
            'metadata': {'scaled_columns': numeric_columns}
        }
    
    def inverse_transform(self, 
                         data: pd.DataFrame, 
                         scaler_name: str = 'feature_scaler') -> pd.DataFrame:
        """스케일링 역변환"""
        if scaler_name not in self.scalers:
            logger.warning(f"스케일러 '{scaler_name}'를 찾을 수 없습니다")
            return data
        
        scaler_info = self.scalers[scaler_name]
        scaler = scaler_info['scaler']
        columns = scaler_info['columns']
        
        processed_data = data.copy()
        
        try:
            # 해당 컬럼들만 역변환
            original_data = scaler.inverse_transform(data[columns])
            processed_data[columns] = original_data
        except Exception as e:
            logger.error(f"역변환 실패: {str(e)}")
        
        return processed_data
    
    def get_preprocessing_summary(self, result: PreprocessingResult) -> Dict[str, Any]:
        """전처리 요약 정보"""
        summary = {
            'data_shape_change': {
                'original': result.original_shape,
                'final': result.final_shape,
                'rows_removed': result.original_shape[0] - result.final_shape[0],
                'columns_added': result.final_shape[1] - result.original_shape[1]
            },
            'missing_values': {
                'columns_affected': len(result.missing_values_handled),
                'total_values_handled': sum(result.missing_values_handled.values()),
                'details': result.missing_values_handled
            },
            'outliers': {
                'columns_affected': len(result.outliers_detected),
                'total_outliers_detected': sum(result.outliers_detected.values()),
                'details': result.outliers_detected
            },
            'transformations': result.transformations_applied,
            'data_quality_score': self._calculate_quality_score(result)
        }
        
        return summary
    
    def _calculate_quality_score(self, result: PreprocessingResult) -> float:
        """데이터 품질 점수 계산 (0-1)"""
        score = 1.0
        
        # 결측값 비율에 따른 감점
        if result.missing_values_handled:
            total_missing = sum(result.missing_values_handled.values())
            total_values = result.original_shape[0] * result.original_shape[1]
            missing_ratio = total_missing / total_values
            score -= missing_ratio * 0.3
        
        # 이상치 비율에 따른 감점
        if result.outliers_detected:
            total_outliers = sum(result.outliers_detected.values())
            outlier_ratio = total_outliers / result.original_shape[0]
            score -= outlier_ratio * 0.2
        
        # 데이터 손실에 따른 감점
        row_loss_ratio = (result.original_shape[0] - result.final_shape[0]) / result.original_shape[0]
        score -= row_loss_ratio * 0.1
        
        return max(0.0, min(1.0, score)) 