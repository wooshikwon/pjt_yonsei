"""
Enhanced RAG System: RAG Manager

비즈니스 지식 검색과 DB 스키마 검색을 통합 관리하는 RAG 시스템 매니저
- BusinessRetriever + SchemaRetriever 통합
- 자연어 요청 기반 종합 컨텍스트 생성
- AI 추천 엔진을 위한 RAG 컨텍스트 제공
"""

import logging
from typing import Dict, List, Any, Optional

from .business_retriever import BusinessRetriever
from .schema_retriever import SchemaRetriever


class RAGManager:
    """
    비즈니스 지식 검색 + DB 스키마 검색 통합 관리자
    
    Enhanced RAG 시스템의 중앙 컨트롤러로서 자연어 분석 요청에 대해
    비즈니스 컨텍스트와 스키마 컨텍스트를 통합하여 제공합니다.
    """
    
    def __init__(self, 
                 metadata_path: str = "input_data/metadata",
                 schema_path: str = "input_data/metadata/database_schemas"):
        """
        RAGManager 초기화
        
        Args:
            metadata_path: 비즈니스 메타데이터 디렉토리 경로
            schema_path: DB 스키마 메타데이터 디렉토리 경로
        """
        self.logger = logging.getLogger(__name__)
        
        # RAG 시스템 컴포넌트 초기화
        try:
            self.business_retriever = BusinessRetriever(metadata_path)
            self.schema_retriever = SchemaRetriever(schema_path)
            self.logger.info("Enhanced RAG 시스템 컴포넌트 초기화 완료")
        except Exception as e:
            self.logger.error(f"RAG 시스템 초기화 실패: {e}")
            raise
        
        # RAG 세션 상태 관리
        self.search_history = []
        self.context_cache = {}
        
    def search_comprehensive_context(self, 
                                   natural_language_query: str,
                                   data_context: Dict) -> Dict:
        """
        종합적인 RAG 컨텍스트 검색
        
        Args:
            natural_language_query: 사용자의 자연어 분석 요청
            data_context: 현재 로딩된 데이터 정보 (컬럼, 타입 등)
            
        Returns:
            Dict: 비즈니스 + 스키마 통합 컨텍스트
        """
        comprehensive_context = {
            'business_context': {},
            'schema_context': {},
            'integration_insights': [],
            'search_metadata': {}
        }
        
        try:
            self.logger.info(f"RAG 통합 검색 시작: {natural_language_query[:50]}...")
            
            # 1. 비즈니스 지식 검색
            business_results = self.business_retriever.search_business_knowledge(
                natural_language_query
            )
            comprehensive_context['business_context'] = self._process_business_results(business_results)
            
            # 2. 스키마 컨텍스트 검색
            data_columns = data_context.get('columns', [])
            schema_context = self.schema_retriever.get_schema_context(
                data_columns, natural_language_query
            )
            comprehensive_context['schema_context'] = schema_context
            
            # 3. 비즈니스와 스키마 컨텍스트 통합 인사이트
            integration_insights = self._generate_integration_insights(
                comprehensive_context['business_context'],
                schema_context,
                natural_language_query
            )
            comprehensive_context['integration_insights'] = integration_insights
            
            # 4. 검색 메타데이터
            comprehensive_context['search_metadata'] = {
                'query': natural_language_query,
                'business_results_count': len(business_results),
                'schema_matches_count': len(schema_context.get('matched_tables', {})),
                'integration_insights_count': len(integration_insights)
            }
            
            # 검색 기록 저장
            self._save_search_history(natural_language_query, comprehensive_context)
            
            self.logger.info("RAG 통합 검색 완료")
            return comprehensive_context
            
        except Exception as e:
            self.logger.error(f"RAG 통합 검색 오류: {e}")
            return comprehensive_context
    
    def get_contextual_recommendations(self, 
                                     query: str, 
                                     business_domain: str,
                                     schema_info: Dict) -> List[Dict]:
        """
        컨텍스트 기반 분석 방법 추천 정보 생성
        
        Args:
            query: 자연어 분석 요청
            business_domain: 감지된 비즈니스 도메인
            schema_info: 스키마 컨텍스트 정보
            
        Returns:
            List[Dict]: AI 추천 엔진을 위한 컨텍스트 정보
        """
        recommendations = []
        
        try:
            # 1. 도메인별 분석 가이드라인
            domain_guidelines = self.business_retriever.get_analysis_guidelines(business_domain)
            
            # 2. 스키마 기반 분석 패턴
            analytical_patterns = schema_info.get('analytical_patterns', {})
            
            # 3. 비즈니스 용어 해석
            query_terms = self._extract_key_terms(query)
            terminology = self.business_retriever.search_domain_terminology(query_terms)
            
            # 4. 통합 추천 정보 생성
            for pattern_name, pattern_info in analytical_patterns.items():
                recommendation = {
                    'pattern_name': pattern_name,
                    'pattern_info': pattern_info,
                    'business_guidelines': self._extract_relevant_guidelines(
                        domain_guidelines, pattern_name
                    ),
                    'terminology_context': terminology,
                    'schema_considerations': self._get_schema_considerations(
                        schema_info, pattern_name
                    )
                }
                recommendations.append(recommendation)
            
            # 관련도 순으로 정렬
            recommendations.sort(
                key=lambda x: x['pattern_info'].get('relevance_score', 0), 
                reverse=True
            )
            
        except Exception as e:
            self.logger.error(f"컨텍스트 기반 추천 생성 오류: {e}")
        
        return recommendations
    
    def get_rag_status(self) -> Dict:
        """RAG 시스템 상태 정보 반환"""
        business_summary = self.business_retriever.get_business_context_summary()
        schema_summary = self.schema_retriever.get_schema_context_summary()
        
        return {
            'business_retriever_status': business_summary,
            'schema_retriever_status': schema_summary,
            'search_history_count': len(self.search_history),
            'cache_size': len(self.context_cache)
        }
    
    def clear_cache(self):
        """RAG 캐시 및 기록 초기화"""
        self.search_history.clear()
        self.context_cache.clear()
        self.logger.info("RAG 캐시 및 검색 기록 초기화 완료")
    
    def _process_business_results(self, business_results: List[Dict]) -> Dict:
        """비즈니스 검색 결과 처리 및 구조화"""
        processed = {
            'terminology': [],
            'domain_knowledge': [],
            'analysis_guidelines': [],
            'detected_domain': None,
            'key_insights': []
        }
        
        for result in business_results:
            result_type = result.get('type', 'unknown')
            
            if result_type == 'terminology':
                processed['terminology'].append(result)
                if not processed['detected_domain']:
                    processed['detected_domain'] = result.get('domain')
            elif result_type == 'domain_knowledge':
                processed['domain_knowledge'].append(result)
            elif result_type == 'analysis_guidelines':
                processed['analysis_guidelines'].append(result)
        
        # 핵심 인사이트 추출
        processed['key_insights'] = self._extract_key_insights(business_results)
        
        return processed
    
    def _generate_integration_insights(self, 
                                     business_context: Dict, 
                                     schema_context: Dict,
                                     query: str) -> List[str]:
        """비즈니스와 스키마 컨텍스트 통합 인사이트 생성"""
        insights = []
        
        # 1. 도메인과 스키마 패턴 매칭
        detected_domain = business_context.get('detected_domain')
        analytical_patterns = schema_context.get('analytical_patterns', {})
        
        if detected_domain and analytical_patterns:
            insights.append(
                f"{detected_domain} 도메인에 특화된 분석 패턴 {len(analytical_patterns)}개 발견"
            )
        
        # 2. 비즈니스 용어와 스키마 컬럼 매칭
        terminology = business_context.get('terminology', [])
        column_details = schema_context.get('column_details', {})
        
        matched_terms = 0
        for term_result in terminology:
            term = term_result.get('term', '')
            if any(term.lower() in col.lower() for col in column_details.keys()):
                matched_terms += 1
        
        if matched_terms > 0:
            insights.append(
                f"비즈니스 용어 {matched_terms}개가 데이터 스키마와 매칭됨"
            )
        
        # 3. 분석 방법 제약사항 통합
        guidelines = business_context.get('analysis_guidelines', [])
        schema_suggestions = schema_context.get('suggestions', [])
        
        if guidelines and schema_suggestions:
            insights.append(
                "비즈니스 가이드라인과 스키마 제약사항을 모두 고려한 분석 방법 추천 가능"
            )
        
        # 4. 데이터 품질 고려사항
        for suggestion in schema_suggestions:
            if '제약조건' in suggestion or '무결성' in suggestion:
                insights.append("데이터 무결성 제약조건이 분석 결과 신뢰성을 보장")
                break
        
        return insights
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """자연어 쿼리에서 핵심 용어 추출"""
        # 간단한 키워드 추출 (향후 NLP 기반으로 개선 가능)
        import re
        
        # 한글, 영문, 숫자가 포함된 단어들 추출
        terms = re.findall(r'[가-힣a-zA-Z0-9]+', query)
        
        # 불용어 제거
        stopwords = {'을', '를', '이', '가', '은', '는', '과', '와', '의', '에', '에서', 
                    '으로', '로', '하고', '하는', '한다', '합니다', '해주세요', '분석', 
                    '확인', '알고', '싶어요', 'the', 'and', 'or', 'in', 'on', 'at'}
        
        filtered_terms = [term for term in terms if term.lower() not in stopwords and len(term) > 1]
        
        return filtered_terms[:10]  # 상위 10개 용어만 반환
    
    def _extract_relevant_guidelines(self, guidelines: str, pattern_name: str) -> str:
        """패턴과 관련된 가이드라인 추출"""
        if not guidelines:
            return ""
        
        # 패턴명과 관련된 섹션 찾기
        pattern_keywords = pattern_name.split('_')
        relevant_sections = []
        
        sections = guidelines.split('\n## ')
        for section in sections:
            section_lower = section.lower()
            if any(keyword in section_lower for keyword in pattern_keywords):
                relevant_sections.append(section)
        
        return '\n\n'.join(relevant_sections) if relevant_sections else guidelines[:500]
    
    def _get_schema_considerations(self, schema_info: Dict, pattern_name: str) -> List[str]:
        """패턴별 스키마 고려사항 생성"""
        considerations = []
        
        # 매칭된 테이블 정보
        matched_tables = schema_info.get('matched_tables', {})
        for table_name, columns in matched_tables.items():
            considerations.append(f"테이블 '{table_name}'의 컬럼 {columns} 활용")
        
        # 관계 정보
        relationships = schema_info.get('relationships', [])
        if relationships:
            considerations.append(f"{len(relationships)}개 테이블 관계 고려 필요")
        
        # 패턴별 특화 고려사항
        if 'comparison' in pattern_name:
            considerations.append("그룹 비교 분석 시 데이터 분포 확인 필요")
        elif 'regression' in pattern_name:
            considerations.append("회귀 분석 시 다중공선성 검토 필요")
        elif 'time_series' in pattern_name:
            considerations.append("시계열 분석 시 데이터 연속성 확인 필요")
        
        return considerations
    
    def _extract_key_insights(self, business_results: List[Dict]) -> List[str]:
        """비즈니스 검색 결과에서 핵심 인사이트 추출"""
        insights = []
        
        # 높은 관련도를 가진 결과들의 인사이트 추출
        high_relevance_results = [
            result for result in business_results 
            if result.get('relevance_score', 0) > 0.7
        ]
        
        for result in high_relevance_results[:3]:  # 상위 3개만
            if result.get('type') == 'terminology':
                definition = result.get('definition', {})
                if 'statistical_methods' in definition:
                    methods = definition['statistical_methods']
                    insights.append(f"추천 통계 방법: {', '.join(methods[:2])}")
            elif result.get('type') == 'analysis_guidelines':
                title = result.get('title', '')
                if title:
                    insights.append(f"가이드라인: {title}")
        
        return insights
    
    def _save_search_history(self, query: str, context: Dict):
        """검색 기록 저장"""
        search_record = {
            'query': query,
            'timestamp': self._get_current_timestamp(),
            'business_results_count': len(context.get('business_context', {}).get('terminology', [])),
            'schema_matches_count': len(context.get('schema_context', {}).get('matched_tables', {}))
        }
        
        self.search_history.append(search_record)
        
        # 기록 개수 제한 (최근 100개만 유지)
        if len(self.search_history) > 100:
            self.search_history = self.search_history[-100:]
    
    def _get_current_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S") 