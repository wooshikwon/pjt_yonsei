"""JSON 직렬화 관련 유틸리티"""

import json
import numpy as np
import pandas as pd

class CustomJSONEncoder(json.JSONEncoder):
    """
    NumPy 및 Pandas 데이터 타입을 Python 기본 타입으로 변환하여
    JSON 직렬화가 가능하도록 하는 커스텀 인코더.
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if pd.isna(obj):
            return None
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super(CustomJSONEncoder, self).default(obj) 