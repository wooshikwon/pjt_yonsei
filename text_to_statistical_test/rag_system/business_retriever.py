"""
Enhanced RAG System: Business Knowledge Retriever

비즈니스 도메인 지식 검색을 위한 RAG 시스템 컴포넌트
- 업계 용어사전 (business_dictionary.json)
- 도메인 전문 지식 (domain_knowledge.md)  
- 분석 가이드라인 (analysis_guidelines.md)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import re


class BusinessRetriever:
    """
    비즈니스 컨텍스트 인식 지식 검색 엔진
    
    BCEmbedding 기반 한중 이중언어 검색을 통해 비즈니스 도메인 지식을 검색하고
    자연어 분석 요청에 적합한 컨텍스트 정보를 제공합니다.
    """
    
    def __init__(self, metadata_path: str = "input_data/metadata"):
        """
        BusinessRetriever 초기화
        
        Args:
            metadata_path: 비즈니스 메타데이터 디렉토리 경로
        """
        self.metadata_path = Path(metadata_path)
        self.logger = logging.getLogger(__name__)
        
        # 비즈니스 지식 데이터 저장소
        self.business_dictionary = {}
        self.domain_knowledge = ""
        self.analysis_guidelines = ""
        
        # 초기화
        self._load_business_knowledge()
        
    def _load_business_knowledge(self):
        """비즈니스 지식베이스 로딩"""
        try:
            # 비즈니스 용어사전 로딩
            dict_path = self.metadata_path / "business_dictionary.json"
            if dict_path.exists():
                with open(dict_path, 'r', encoding='utf-8') as f:
                    self.business_dictionary = json.load(f)
                self.logger.info(f"비즈니스 용어사전 로딩 완료: {len(self.business_dictionary)}개 도메인")
            
            # 도메인 지식 로딩
            knowledge_path = self.metadata_path / "domain_knowledge.md"
            if knowledge_path.exists():
                with open(knowledge_path, 'r', encoding='utf-8') as f:
                    self.domain_knowledge = f.read()
                self.logger.info("도메인 전문 지식 로딩 완료")
            
            # 분석 가이드라인 로딩
            guidelines_path = self.metadata_path / "analysis_guidelines.md"
            if guidelines_path.exists():
                with open(guidelines_path, 'r', encoding='utf-8') as f:
                    self.analysis_guidelines = f.read()
                self.logger.info("분석 가이드라인 로딩 완료")
                
        except Exception as e:
            self.logger.error(f"비즈니스 지식베이스 로딩 실패: {e}")
    
    def search_business_knowledge(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        비즈니스 지식 검색
        
        Args:
            query: 검색 쿼리 (자연어 분석 요청)
            top_k: 반환할 최대 결과 수
            
        Returns:
            List[Dict]: 검색된 비즈니스 컨텍스트 정보
        """
        results = []
        
        try:
            # 1. 도메인 감지
            detected_domain = self._detect_business_domain(query)
            
            # 2. 용어사전 검색
            terminology_results = self._search_terminology(query, detected_domain)
            
            # 3. 도메인 지식 검색
            domain_knowledge_results = self._search_domain_knowledge(query)
            
            # 4. 분석 가이드라인 검색
            guidelines_results = self._search_analysis_guidelines(query)
            
            # 결과 통합
            results.extend(terminology_results)
            results.extend(domain_knowledge_results)
            results.extend(guidelines_results)
            
            # 관련도 순으로 정렬하고 top_k 반환
            results = sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"비즈니스 지식 검색 오류: {e}")
            return []
    
    def search_domain_terminology(self, terms: List[str]) -> Dict:
        """
        특정 용어들의 비즈니스 정의 검색
        
        Args:
            terms: 검색할 용어 리스트
            
        Returns:
            Dict: 용어별 비즈니스 정의
        """
        terminology = {}
        
        for term in terms:
            for domain, domain_data in self.business_dictionary.items():
                for key, definition in domain_data.items():
                    if term.lower() in key.lower() or any(term.lower() in str(v).lower() for v in definition.values()):
                        terminology[term] = {
                            'domain': domain,
                            'definition': definition,
                            'business_context': definition.get('business_context', ''),
                            'statistical_methods': definition.get('statistical_methods', [])
                        }
                        break
                        
        return terminology
    
    def get_analysis_guidelines(self, domain: str = None) -> str:
        """
        도메인별 분석 가이드라인 반환
        
        Args:
            domain: 비즈니스 도메인 (선택사항)
            
        Returns:
            str: 분석 가이드라인 텍스트
        """
        if domain and domain in self.business_dictionary:
            # 도메인 특화 가이드라인 추출
            domain_guidelines = self._extract_domain_guidelines(domain)
            return domain_guidelines
        
        return self.analysis_guidelines
    
    def _detect_business_domain(self, query: str) -> Optional[str]:
        """자연어 쿼리에서 비즈니스 도메인 감지"""
        domain_keywords = {
            'sales': ['매출', '판매', '영업', '수익', '고객', '세그먼트'],
            'marketing': ['마케팅', '광고', '캠페인', 'A/B', '전환율', '브랜드'],
            'finance': ['금융', '재무', '리스크', '포트폴리오', '투자', '수익률'],
            'healthcare': ['의료', '헬스케어', '환자', '임상', '치료', '진단'],
            'manufacturing': ['제조', '생산', '품질', '공정', '제품', '불량률']
        }
        
        query_lower = query.lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
                
        return None
    
    def _search_terminology(self, query: str, domain: str = None) -> List[Dict]:
        """용어사전에서 관련 용어 검색"""
        results = []
        
        domains_to_search = [domain] if domain else self.business_dictionary.keys()
        
        for search_domain in domains_to_search:
            if search_domain not in self.business_dictionary:
                continue
                
            domain_data = self.business_dictionary[search_domain]
            
            for term, definition in domain_data.items():
                relevance = self._calculate_relevance(query, term, str(definition))
                
                if relevance > 0.3:  # 임계값
                    results.append({
                        'type': 'terminology',
                        'domain': search_domain,
                        'term': term,
                        'definition': definition,
                        'relevance_score': relevance,
                        'source': 'business_dictionary'
                    })
        
        return results
    
    def _search_domain_knowledge(self, query: str) -> List[Dict]:
        """도메인 지식에서 관련 정보 검색"""
        results = []
        
        if not self.domain_knowledge:
            return results
        
        # 간단한 키워드 매칭 (실제로는 embedding 기반 검색 사용 가능)
        sections = self.domain_knowledge.split('\n## ')
        
        for section in sections:
            if not section.strip():
                continue
                
            relevance = self._calculate_relevance(query, section[:100], section)
            
            if relevance > 0.2:
                lines = section.split('\n')
                title = lines[0].strip('#').strip() if lines else 'Unknown'
                
                results.append({
                    'type': 'domain_knowledge',
                    'title': title,
                    'content': section,
                    'relevance_score': relevance,
                    'source': 'domain_knowledge'
                })
        
        return results
    
    def _search_analysis_guidelines(self, query: str) -> List[Dict]:
        """분석 가이드라인에서 관련 정보 검색"""
        results = []
        
        if not self.analysis_guidelines:
            return results
        
        sections = self.analysis_guidelines.split('\n## ')
        
        for section in sections:
            if not section.strip():
                continue
                
            relevance = self._calculate_relevance(query, section[:100], section)
            
            if relevance > 0.2:
                lines = section.split('\n')
                title = lines[0].strip('#').strip() if lines else 'Unknown'
                
                results.append({
                    'type': 'analysis_guidelines',
                    'title': title,
                    'content': section,
                    'relevance_score': relevance,
                    'source': 'analysis_guidelines'
                })
        
        return results
    
    def _calculate_relevance(self, query: str, title: str, content: str) -> float:
        """간단한 관련도 계산 (키워드 매칭 기반)"""
        query_words = set(re.findall(r'\w+', query.lower()))
        title_words = set(re.findall(r'\w+', title.lower()))
        content_words = set(re.findall(r'\w+', content.lower()))
        
        # 제목에서의 매칭 점수 (가중치 2.0)
        title_match = len(query_words & title_words) / max(len(query_words), 1) * 2.0
        
        # 내용에서의 매칭 점수 (가중치 1.0)
        content_match = len(query_words & content_words) / max(len(query_words), 1)
        
        return min(title_match + content_match, 1.0)
    
    def _extract_domain_guidelines(self, domain: str) -> str:
        """특정 도메인의 가이드라인 추출"""
        if not self.analysis_guidelines:
            return ""
        
        # 도메인별 섹션 찾기
        domain_names = {
            'sales': '판매',
            'marketing': '마케팅', 
            'finance': '금융',
            'healthcare': '헬스케어',
            'manufacturing': '제조'
        }
        
        domain_korean = domain_names.get(domain, domain)
        
        sections = self.analysis_guidelines.split('\n## ')
        for section in sections:
            if domain_korean in section[:50]:
                return section
                
        return self.analysis_guidelines
    
    def get_business_context_summary(self) -> Dict:
        """비즈니스 컨텍스트 요약 정보 반환"""
        return {
            'domains_available': list(self.business_dictionary.keys()),
            'terminology_count': sum(len(domain_data) for domain_data in self.business_dictionary.values()),
            'has_domain_knowledge': bool(self.domain_knowledge),
            'has_analysis_guidelines': bool(self.analysis_guidelines),
            'metadata_path': str(self.metadata_path)
        } 