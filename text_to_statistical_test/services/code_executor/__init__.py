"""
Code Executor Package

LLM이 생성하거나 참조한 코드의 안전한 실행 환경
- 샌드박스 환경에서의 안전한 코드 실행
- 리소스 제한 및 보안 관리
- 실행 결과 및 오류 처리
"""

from .safe_code_runner import SafeCodeRunner

__all__ = [
    'SafeCodeRunner'
] 