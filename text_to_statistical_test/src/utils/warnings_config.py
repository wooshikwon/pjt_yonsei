"""
통합된 경고 및 로깅 설정 모듈

모든 라이브러리의 경고 메시지와 로깅을 일관성 있게 관리합니다.
"""

import os
import warnings
import logging
import sys
import contextlib

# 시각화 GUI 완전 차단
import matplotlib
matplotlib.use('Agg')  # GUI 없는 백엔드로 강제 설정

def setup_warnings_and_logging():
    """
    모든 라이브러리의 경고 메시지와 로깅을 설정합니다.
    시스템 전체에서 일관된 경고 처리를 보장합니다.
    """
    # 환경변수 설정 (HuggingFace 관련)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    
    # 시각화 GUI 비활성화
    os.environ["MPLBACKEND"] = "Agg"
    if "DISPLAY" in os.environ:
        del os.environ["DISPLAY"]  # X11 디스플레이 비활성화
    
    # Python 기본 경고 숨기기
    warnings.filterwarnings("ignore")
    
    # 라이브러리별 로깅 레벨 설정
    logging.getLogger("llama_index").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)

@contextlib.contextmanager
def suppress_stdout():
    """stdout 출력을 일시적으로 차단하는 컨텍스트 매니저"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

@contextlib.contextmanager
def suppress_warnings():
    """경고 메시지를 일시적으로 차단하는 컨텍스트 매니저"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield 

def suppress_warnings_and_logs():
    """
    모든 경고와 특정 라이브러리의 로그를 무시하고, 
    matplotlib 백엔드를 'Agg'로 설정하여 GUI 창이 뜨지 않도록 합니다.
    """
    # 모든 경고를 무시하도록 설정
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # LlamaIndex 로깅 레벨 설정
    logging.getLogger("llama_index.readers.file.base").setLevel(logging.CRITICAL)

@contextlib.contextmanager
def suppress_all_warnings():
    """모든 경고를 무시하는 컨텍스트 매니저"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield 