# 파일명: services/rag/context_builder.py

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ContextBuilder:
    """검색된 문서들을 바탕으로 LLM 프롬프트에 사용될 컨텍스트를 구성합니다."""

    def build_context(self, search_results: List[Dict[str, Any]], max_length: int = 3000) -> str:
        """검색 결과 목록을 하나의 컨텍스트 문자열로 만듭니다."""
        context_str = ""
        
        for i, result in enumerate(search_results):
            content = f"## 참고자료 {i+1} (출처: {result['metadata'].get('source', 'N/A')}, 관련도: {result.get('score', 0):.2f})\n"
            content += f"{result['text']}\n\n"
            
            # 최대 길이를 초과하지 않는지 확인
            if len(context_str) + len(content) > max_length:
                break
                
            context_str += content
            
        return context_str.strip()