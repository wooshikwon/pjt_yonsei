"""
Knowledge Store

통계 분석 관련 지식 문서를 관리하는 지식 저장소
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from .vector_store import VectorStore
from .retriever import Retriever
from utils.helpers import generate_unique_id, safe_json_dumps, safe_json_loads
from utils.error_handler import ErrorHandler, RAGError

logger = logging.getLogger(__name__)

class KnowledgeStore:
    """
    통계 분석 관련 지식을 저장하고 검색하는 지식 저장소
    """
    
    def __init__(self, 
                 storage_path: Optional[str] = None,
                 vector_store: Optional[VectorStore] = None):
        """
        지식저장소 초기화
        
        Args:
            storage_path: 저장소 경로
            vector_store: 벡터 저장소 (None이면 새로 생성)
        """
        self.storage_path = Path(storage_path) if storage_path else None
        
        # 벡터 저장소 초기화
        if vector_store:
            self.vector_store = vector_store
        else:
            vector_path = str(self.storage_path / "vectors") if self.storage_path else None
            self.vector_store = VectorStore(storage_path=vector_path)
        
        # 검색기 초기화
        self.retriever = Retriever(self.vector_store)
        
        # 오류 처리
        self.error_handler = ErrorHandler()
        
        # 지식저장소 메타데이터
        self.metadata = {
            'created_at': datetime.now(),
            'last_updated': datetime.now(),
            'total_documents': 0,
            'categories': {}
        }
        
        # 저장소 설정
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load_metadata()
        
        logger.info("지식저장소 초기화 완료")
    
    def add_statistical_knowledge(self, 
                                knowledge_data: Dict[str, Any]) -> str:
        """
        통계 지식 문서 추가
        
        Args:
            knowledge_data: 지식 데이터
                {
                    'title': str,
                    'content': str,
                    'category': str,  # 'test', 'assumption', 'interpretation', etc.
                    'statistical_tests': List[str],  # 관련 통계 검정들
                    'data_types': List[str],  # 적용 가능한 데이터 타입들
                    'examples': List[Dict],  # 예시들
                    'references': List[str]  # 참고 문헌
                }
                
        Returns:
            str: 문서 ID
        """
        try:
            # 필수 필드 검증
            required_fields = ['title', 'content', 'category']
            for field in required_fields:
                if field not in knowledge_data:
                    raise RAGError(f"필수 필드가 누락되었습니다: {field}")
            
            # 메타데이터 구성
            metadata = {
                'category': knowledge_data['category'],
                'statistical_tests': knowledge_data.get('statistical_tests', []),
                'data_types': knowledge_data.get('data_types', []),
                'added_at': datetime.now().isoformat(),
                'document_type': 'statistical_knowledge'
            }
            
            # 문서 텍스트 구성
            content_parts = [
                f"제목: {knowledge_data['title']}",
                f"분류: {knowledge_data['category']}",
                f"내용: {knowledge_data['content']}"
            ]
            
            # 관련 통계 검정 추가
            if knowledge_data.get('statistical_tests'):
                tests_text = ", ".join(knowledge_data['statistical_tests'])
                content_parts.append(f"관련 통계 검정: {tests_text}")
            
            # 예시 추가
            if knowledge_data.get('examples'):
                for i, example in enumerate(knowledge_data['examples']):
                    if isinstance(example, dict) and 'description' in example:
                        content_parts.append(f"예시 {i+1}: {example['description']}")
                    elif isinstance(example, str):
                        content_parts.append(f"예시 {i+1}: {example}")
            
            document_text = "\n\n".join(content_parts)
            
            # 벡터 저장소에 추가
            documents = [{
                'text': document_text,
                'metadata': metadata
            }]
            
            doc_ids = self.vector_store.add_documents(documents)
            doc_id = doc_ids[0]
            
            # 메타데이터 업데이트
            self._update_metadata(knowledge_data['category'])
            
            logger.info(f"통계 지식 문서 추가됨: {knowledge_data['title']} (ID: {doc_id})")
            return doc_id
            
        except Exception as e:
            logger.error(f"통계 지식 추가 실패: {str(e)}")
            raise RAGError(f"통계 지식 추가 실패: {str(e)}")
    
    def search_knowledge(self, 
                        query: str,
                        category: Optional[str] = None,
                        statistical_test: Optional[str] = None,
                        data_type: Optional[str] = None,
                        k: int = 5) -> List[Dict[str, Any]]:
        """
        지식 검색
        
        Args:
            query: 검색 쿼리
            category: 지식 카테고리 필터
            statistical_test: 통계 검정 필터
            data_type: 데이터 타입 필터
            k: 반환할 결과 수
            
        Returns:
            List[Dict[str, Any]]: 검색 결과
        """
        try:
            # 필터 조건 구성
            filter_metadata = {}
            if category:
                filter_metadata['category'] = category
            
            # 컨텍스트 정보 구성
            context = {}
            if statistical_test:
                context['analysis_type'] = statistical_test
            if data_type:
                context['data_type'] = data_type
            
            # 검색 실행
            results = self.retriever.retrieve(
                query=query,
                k=k,
                filter_metadata=filter_metadata if filter_metadata else None,
                context=context if context else None
            )
            
            # 결과 후처리
            processed_results = []
            for result in results:
                processed_result = {
                    'doc_id': result['doc_id'],
                    'content': result['text'],
                    'score': result['score'],
                    'category': result['metadata'].get('category'),
                    'statistical_tests': result['metadata'].get('statistical_tests', []),
                    'data_types': result['metadata'].get('data_types', []),
                    'relevance_explanation': self._explain_relevance(result, query)
                }
                processed_results.append(processed_result)
            
            logger.info(f"지식 검색 완료: {len(processed_results)}개 결과")
            return processed_results
            
        except Exception as e:
            logger.error(f"지식 검색 실패: {str(e)}")
            raise RAGError(f"지식 검색 실패: {str(e)}")
    
    def get_knowledge_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        카테고리별 지식 조회
        
        Args:
            category: 지식 카테고리
            
        Returns:
            List[Dict[str, Any]]: 해당 카테고리의 지식 문서들
        """
        try:
            filter_metadata = {'category': category}
            documents = self.vector_store.list_documents(filter_metadata)
            
            results = []
            for doc_info in documents:
                doc = self.vector_store.get_document(doc_info['doc_id'])
                if doc:
                    result = {
                        'doc_id': doc_info['doc_id'],
                        'content': doc['text'],
                        'metadata': doc['metadata'],
                        'category': doc['metadata'].get('category'),
                        'statistical_tests': doc['metadata'].get('statistical_tests', []),
                        'data_types': doc['metadata'].get('data_types', [])
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"카테고리별 지식 조회 실패: {str(e)}")
            raise RAGError(f"카테고리별 지식 조회 실패: {str(e)}")
    
    def get_recommendations(self, 
                          user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        사용자 컨텍스트에 기반한 지식 추천
        
        Args:
            user_context: 사용자 컨텍스트
                {
                    'data_type': str,
                    'analysis_goal': str,
                    'user_request': str,
                    'data_characteristics': Dict
                }
                
        Returns:
            List[Dict[str, Any]]: 추천 지식 리스트
        """
        try:
            recommendations = []
            
            # 데이터 타입 기반 추천
            data_type = user_context.get('data_type')
            if data_type:
                data_type_results = self.search_knowledge(
                    query=f"{data_type} 데이터 분석",
                    data_type=data_type,
                    k=3
                )
                recommendations.extend(data_type_results)
            
            # 분석 목표 기반 추천
            analysis_goal = user_context.get('analysis_goal')
            if analysis_goal:
                goal_results = self.search_knowledge(
                    query=analysis_goal,
                    k=3
                )
                recommendations.extend(goal_results)
            
            # 사용자 요청 기반 추천
            user_request = user_context.get('user_request')
            if user_request:
                request_results = self.search_knowledge(
                    query=user_request,
                    k=3
                )
                recommendations.extend(request_results)
            
            # 중복 제거 및 점수 기반 정렬
            unique_recommendations = {}
            for rec in recommendations:
                doc_id = rec['doc_id']
                if doc_id not in unique_recommendations or rec['score'] > unique_recommendations[doc_id]['score']:
                    unique_recommendations[doc_id] = rec
            
            final_recommendations = list(unique_recommendations.values())
            final_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return final_recommendations[:5]  # 상위 5개 추천
            
        except Exception as e:
            logger.error(f"지식 추천 실패: {str(e)}")
            raise RAGError(f"지식 추천 실패: {str(e)}")
    
    def load_default_knowledge(self):
        """
        기본 통계 지식 로드
        """
        try:
            default_knowledge = self._get_default_statistical_knowledge()
            
            for knowledge in default_knowledge:
                try:
                    self.add_statistical_knowledge(knowledge)
                except Exception as e:
                    logger.warning(f"기본 지식 로드 실패: {knowledge.get('title', 'Unknown')} - {str(e)}")
            
            logger.info(f"{len(default_knowledge)}개의 기본 지식이 로드되었습니다.")
            
        except Exception as e:
            logger.error(f"기본 지식 로드 실패: {str(e)}")
    
    def _get_default_statistical_knowledge(self) -> List[Dict[str, Any]]:
        """기본 통계 지식 반환"""
        return [
            {
                'title': 'T-검정의 기본 가정',
                'content': 'T-검정을 수행하기 위해서는 다음 가정들이 만족되어야 합니다: 1) 정규성: 데이터가 정규분포를 따라야 함, 2) 독립성: 관측값들이 서로 독립적이어야 함, 3) 등분산성(두 집단 비교 시): 두 집단의 분산이 같아야 함',
                'category': 'assumptions',
                'statistical_tests': ['t-test', 'one-sample t-test', 'two-sample t-test', 'paired t-test'],
                'data_types': ['numeric'],
                'examples': [
                    {'description': '두 그룹의 평균 점수 비교 시 각 그룹의 데이터가 정규분포를 따르는지 확인'},
                    {'description': 'Shapiro-Wilk 검정으로 정규성 검정, Levene 검정으로 등분산성 검정'}
                ]
            },
            {
                'title': 'ANOVA의 기본 가정',
                'content': 'ANOVA(분산분석)를 수행하기 위한 가정: 1) 정규성: 각 집단의 데이터가 정규분포를 따라야 함, 2) 독립성: 관측값들이 서로 독립적이어야 함, 3) 등분산성: 모든 집단의 분산이 같아야 함',
                'category': 'assumptions',
                'statistical_tests': ['anova', 'one-way anova', 'two-way anova'],
                'data_types': ['numeric'],
                'examples': [
                    {'description': '여러 교육 방법의 효과를 비교할 때 각 방법군의 점수 분포가 정규분포를 따르는지 확인'},
                    {'description': 'Bartlett 검정이나 Levene 검정으로 등분산성 확인'}
                ]
            },
            {
                'title': '카이제곱 검정의 적용',
                'content': '카이제곱 검정은 범주형 데이터의 독립성이나 적합도를 검정하는 방법입니다. 적용 조건: 1) 범주형 데이터, 2) 기댓값이 5 이상인 셀이 전체의 80% 이상, 3) 기댓값이 1 미만인 셀이 없어야 함',
                'category': 'test_selection',
                'statistical_tests': ['chi-square test', 'chi-square independence test', 'chi-square goodness-of-fit'],
                'data_types': ['categorical'],
                'examples': [
                    {'description': '성별과 선호도 간의 관련성 검정'},
                    {'description': '관찰된 빈도와 기댓값 비교'}
                ]
            },
            {
                'title': '상관분석 해석',
                'content': '상관계수 해석 가이드: 피어슨 상관계수 r의 절댓값이 0.1-0.3이면 약한 상관, 0.3-0.7이면 중간 상관, 0.7-0.9이면 강한 상관, 0.9 이상이면 매우 강한 상관. 상관관계는 인과관계를 의미하지 않음에 주의',
                'category': 'interpretation',
                'statistical_tests': ['correlation', 'pearson correlation', 'spearman correlation'],
                'data_types': ['numeric'],
                'examples': [
                    {'description': '키와 몸무게의 상관관계 r=0.8 (강한 양의 상관)'},
                    {'description': '온도와 아이스크림 판매량의 상관관계 분석'}
                ]
            },
            {
                'title': '비모수 검정의 활용',
                'content': '정규성 가정이 위배될 때 사용하는 비모수 검정들: 1) Mann-Whitney U 검정 (독립 두 집단 비교), 2) Wilcoxon 부호순위 검정 (대응 표본), 3) Kruskal-Wallis 검정 (세 개 이상 집단 비교), 4) Spearman 순위 상관',
                'category': 'test_selection',
                'statistical_tests': ['mann-whitney', 'wilcoxon', 'kruskal-wallis', 'spearman correlation'],
                'data_types': ['numeric', 'ordinal'],
                'examples': [
                    {'description': '두 치료법의 효과 비교 시 데이터가 정규분포를 따르지 않는 경우'},
                    {'description': '만족도 점수(서열 데이터)의 집단 간 비교'}
                ]
            },
            {
                'title': '회귀분석의 가정',
                'content': '선형회귀분석의 기본 가정: 1) 선형성: 독립변수와 종속변수 간 선형관계, 2) 독립성: 잔차들이 서로 독립적, 3) 등분산성: 잔차의 분산이 일정, 4) 정규성: 잔차가 정규분포를 따름, 5) 다중공선성 없음',
                'category': 'assumptions',
                'statistical_tests': ['linear regression', 'multiple regression'],
                'data_types': ['numeric'],
                'examples': [
                    {'description': '잔차 플롯으로 등분산성과 선형성 확인'},
                    {'description': 'VIF(분산팽창인자)로 다중공선성 진단'}
                ]
            }
        ]
    
    def _explain_relevance(self, result: Dict[str, Any], query: str) -> str:
        """검색 결과의 관련성 설명"""
        try:
            metadata = result.get('metadata', {})
            score = result.get('score', 0)
            
            explanations = []
            
            # 점수 기반 설명
            if score > 0.8:
                explanations.append("매우 높은 관련성")
            elif score > 0.6:
                explanations.append("높은 관련성")
            elif score > 0.4:
                explanations.append("중간 관련성")
            else:
                explanations.append("낮은 관련성")
            
            # 카테고리 기반 설명
            category = metadata.get('category')
            if category:
                category_names = {
                    'assumptions': '검정 가정',
                    'test_selection': '검정 선택',
                    'interpretation': '결과 해석'
                }
                explanations.append(f"{category_names.get(category, category)} 관련")
            
            return " - ".join(explanations)
            
        except Exception:
            return "관련성 설명 생성 실패"
    
    def _update_metadata(self, category: str):
        """메타데이터 업데이트"""
        self.metadata['last_updated'] = datetime.now()
        self.metadata['total_documents'] = len(self.vector_store.document_metadata)
        
        if category not in self.metadata['categories']:
            self.metadata['categories'][category] = 0
        self.metadata['categories'][category] += 1
        
        if self.storage_path:
            self._save_metadata()
    
    def _save_metadata(self):
        """메타데이터 저장"""
        try:
            metadata_path = self.storage_path / "kb_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(safe_json_dumps(self.metadata))
        except Exception as e:
            logger.error(f"메타데이터 저장 실패: {str(e)}")
    
    def _load_metadata(self):
        """메타데이터 로드"""
        try:
            metadata_path = self.storage_path / "kb_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    loaded_metadata = safe_json_loads(f.read())
                if loaded_metadata:
                    self.metadata.update(loaded_metadata)
        except Exception as e:
            logger.error(f"메타데이터 로드 실패: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        지식저장소 통계 정보 반환
        
        Returns:
            Dict[str, Any]: 통계 정보
        """
        return {
            'total_documents': self.metadata.get('total_documents', 0),
            'categories': self.metadata.get('categories', {}),
            'storage_path': str(self.storage_path) if self.storage_path else None,
            'last_updated': self.metadata.get('last_updated')
        }
    
    def get_collections(self) -> List[str]:
        """
        사용 가능한 지식 컬렉션(카테고리) 목록 반환
        
        Returns:
            List[str]: 컬렉션 목록
        """
        try:
            categories = self.metadata.get('categories', {})
            return list(categories.keys())
        except Exception as e:
            logger.error(f"컬렉션 목록 조회 오류: {e}")
            return []
    
    def get_collection_info(self, collection: str) -> Dict[str, Any]:
        """
        특정 컬렉션(카테고리) 정보 반환
        
        Args:
            collection: 컬렉션 이름
            
        Returns:
            Dict[str, Any]: 컬렉션 정보
        """
        try:
            categories = self.metadata.get('categories', {})
            if collection not in categories:
                return {
                    'name': collection,
                    'document_count': 0,
                    'exists': False
                }
            
            return {
                'name': collection,
                'document_count': categories[collection],
                'exists': True,
                'last_updated': self.metadata.get('last_updated')
            }
        except Exception as e:
            logger.error(f"컬렉션 정보 조회 오류: {e}")
            return {
                'name': collection,
                'document_count': 0,
                'exists': False,
                'error': str(e)
            }
    
    def reload(self) -> bool:
        """
        지식 베이스 재로드
        
        Returns:
            bool: 재로드 성공 여부
        """
        try:
            if self.storage_path:
                self._load_metadata()
            logger.info("지식 베이스 재로드 완료")
            return True
        except Exception as e:
            logger.error(f"지식 베이스 재로드 오류: {e}")
            return False 