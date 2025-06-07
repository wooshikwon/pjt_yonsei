"""
프로젝트 전반에서 사용되는 범용 헬퍼 클래스 및 함수를 정의합니다.
"""

import threading
from typing import Any
import json
import numpy as np

# Custom JSON Encoder for NumPy types
class CustomJSONEncoder(json.JSONEncoder):
    """
    NumPy 데이터 타입(ndarray, int64, float64, bool_ 등)을
    Python 기본 타입으로 변환하여 JSON 직렬화가 가능하도록 만드는 커스텀 인코더.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.datetime64, np.timedelta64)):
            return str(obj)
        return super(CustomJSONEncoder, self).default(obj)

class Singleton(type):
    """
    싱글턴 디자인 패턴을 구현하는 메타클래스.
    이 메타클래스를 사용하는 클래스는 프로그램 내에서 단 하나의 인스턴스만 갖게 됩니다.
    """
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls] 