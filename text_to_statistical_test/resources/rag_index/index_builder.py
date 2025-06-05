"""
Enhanced RAG Index Builder

비즈니스 컨텍스트 인식 AI 통계 분석 시스템을 위한 RAG 인덱스 구축 및 관리 시스템.
다음과 같은 종류의 인덱스를 생성하고 관리합니다:

1. Business Context Index: 도메인별 비즈니스 지식
2. Schema Context Index: 데이터베이스 스키마 및 테이블 관계
3. Statistical Knowledge Index: 통계 방법론 및 해석 가이드
4. Analysis Pattern Index: 분석 패턴 및 모범 사례
5. Error Resolution Index: 오류 해결 및 대안 방법
"""

import json
import logging
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import uuid

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Using simple vector search instead.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not available. Using TF-IDF vectorization instead.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class IndexDocument:
    """인덱스 문서 클래스"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    category: str = "general"
    domain: str = "general"
    relevance_score: float = 1.0
    created_at: str = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    """검색 결과 클래스"""
    document: IndexDocument
    similarity_score: float
    rank: int
    retrieval_context: Dict[str, Any]


class BusinessContextIndexer:
    """비즈니스 컨텍스트 인덱스 구축기"""
    
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir) / "business_context"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.documents = []
        self.embeddings = []
        self.metadata_db = {}
        
    def add_business_domain(self, domain: str, knowledge_base: Dict[str, Any]):
        """비즈니스 도메인 지식 추가"""
        try:
            # 도메인별 핵심 개념들
            concepts = knowledge_base.get('concepts', [])
            for concept in concepts:
                doc_id = f"business_{domain}_{concept['name']}"
                content = f"Domain: {domain}\nConcept: {concept['name']}\nDescription: {concept['description']}\nKeywords: {', '.join(concept.get('keywords', []))}"
                
                doc = IndexDocument(
                    id=doc_id,
                    content=content,
                    metadata={
                        'type': 'business_concept',
                        'domain': domain,
                        'concept_name': concept['name'],
                        'importance': concept.get('importance', 1.0)
                    },
                    category='business_concept',
                    domain=domain
                )
                self.documents.append(doc)
            
            # 도메인별 KPI 및 메트릭
            metrics = knowledge_base.get('metrics', [])
            for metric in metrics:
                doc_id = f"metric_{domain}_{metric['name']}"
                content = f"Domain: {domain}\nMetric: {metric['name']}\nDefinition: {metric['definition']}\nCalculation: {metric.get('calculation', 'N/A')}\nBusiness Impact: {metric.get('business_impact', 'N/A')}"
                
                doc = IndexDocument(
                    id=doc_id,
                    content=content,
                    metadata={
                        'type': 'business_metric',
                        'domain': domain,
                        'metric_name': metric['name'],
                        'category': metric.get('category', 'general')
                    },
                    category='business_metric',
                    domain=domain
                )
                self.documents.append(doc)
            
            # 도메인별 분석 패턴
            patterns = knowledge_base.get('analysis_patterns', [])
            for pattern in patterns:
                doc_id = f"pattern_{domain}_{pattern['name']}"
                content = f"Domain: {domain}\nPattern: {pattern['name']}\nDescription: {pattern['description']}\nWhen to use: {pattern.get('when_to_use', 'N/A')}\nExpected outcomes: {pattern.get('expected_outcomes', 'N/A')}"
                
                doc = IndexDocument(
                    id=doc_id,
                    content=content,
                    metadata={
                        'type': 'analysis_pattern',
                        'domain': domain,
                        'pattern_name': pattern['name'],
                        'complexity': pattern.get('complexity', 'medium')
                    },
                    category='analysis_pattern',
                    domain=domain
                )
                self.documents.append(doc)
                
            logging.info(f"Added {len(concepts + metrics + patterns)} documents for domain: {domain}")
            
        except Exception as e:
            logging.error(f"Error adding business domain {domain}: {e}")
    
    def save_index(self):
        """비즈니스 컨텍스트 인덱스 저장"""
        try:
            index_file = self.index_dir / "business_context_index.json"
            
            # 문서들을 JSON으로 저장
            documents_data = [doc.to_dict() for doc in self.documents]
            
            index_data = {
                'documents': documents_data,
                'metadata': {
                    'total_documents': len(self.documents),
                    'domains': list(set(doc.domain for doc in self.documents)),
                    'categories': list(set(doc.category for doc in self.documents)),
                    'created_at': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
            
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
            
            logging.info(f"Business context index saved to {index_file}")
            
        except Exception as e:
            logging.error(f"Error saving business context index: {e}")


class SchemaContextIndexer:
    """스키마 컨텍스트 인덱스 구축기"""
    
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir) / "schema_context"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.documents = []
        
    def add_database_schema(self, schema_name: str, schema_info: Dict[str, Any]):
        """데이터베이스 스키마 정보 추가"""
        try:
            # 테이블 정보
            tables = schema_info.get('tables', {})
            for table_name, table_info in tables.items():
                doc_id = f"table_{schema_name}_{table_name}"
                
                columns = table_info.get('columns', [])
                column_desc = "\n".join([f"- {col['name']} ({col['type']}): {col.get('description', 'N/A')}" for col in columns])
                
                content = f"Schema: {schema_name}\nTable: {table_name}\nDescription: {table_info.get('description', 'N/A')}\nColumns:\n{column_desc}"
                
                doc = IndexDocument(
                    id=doc_id,
                    content=content,
                    metadata={
                        'type': 'table_schema',
                        'schema_name': schema_name,
                        'table_name': table_name,
                        'column_count': len(columns),
                        'primary_keys': table_info.get('primary_keys', []),
                        'foreign_keys': table_info.get('foreign_keys', [])
                    },
                    category='table_schema',
                    domain=schema_name
                )
                self.documents.append(doc)
            
            # 테이블 간 관계 정보
            relationships = schema_info.get('relationships', [])
            for rel in relationships:
                doc_id = f"relationship_{schema_name}_{rel['from_table']}_{rel['to_table']}"
                content = f"Schema: {schema_name}\nRelationship: {rel['from_table']} -> {rel['to_table']}\nType: {rel['type']}\nDescription: {rel.get('description', 'N/A')}"
                
                doc = IndexDocument(
                    id=doc_id,
                    content=content,
                    metadata={
                        'type': 'table_relationship',
                        'schema_name': schema_name,
                        'from_table': rel['from_table'],
                        'to_table': rel['to_table'],
                        'relationship_type': rel['type']
                    },
                    category='table_relationship',
                    domain=schema_name
                )
                self.documents.append(doc)
            
            logging.info(f"Added schema information for {schema_name}: {len(tables)} tables, {len(relationships)} relationships")
            
        except Exception as e:
            logging.error(f"Error adding schema {schema_name}: {e}")
    
    def save_index(self):
        """스키마 컨텍스트 인덱스 저장"""
        try:
            index_file = self.index_dir / "schema_context_index.json"
            
            documents_data = [doc.to_dict() for doc in self.documents]
            
            index_data = {
                'documents': documents_data,
                'metadata': {
                    'total_documents': len(self.documents),
                    'schemas': list(set(doc.domain for doc in self.documents)),
                    'document_types': list(set(doc.metadata.get('type') for doc in self.documents)),
                    'created_at': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
            
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
            
            logging.info(f"Schema context index saved to {index_file}")
            
        except Exception as e:
            logging.error(f"Error saving schema context index: {e}")


class StatisticalKnowledgeIndexer:
    """통계 지식 인덱스 구축기"""
    
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir) / "statistical_knowledge"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.documents = []
        
    def add_statistical_methods(self, methods_db: Dict[str, Any]):
        """통계 방법론 지식 추가"""
        try:
            methods = methods_db.get('methods', [])
            for method in methods:
                doc_id = f"method_{method['name']}"
                
                assumptions_text = "\n".join([f"- {assumption}" for assumption in method.get('assumptions', [])])
                interpretation_text = "\n".join([f"- {interp}" for interp in method.get('interpretation_guide', [])])
                
                content = f"""Statistical Method: {method['name']}
Category: {method.get('category', 'General')}
Description: {method.get('description', 'N/A')}
Use Cases: {', '.join(method.get('use_cases', []))}
Assumptions:
{assumptions_text}
Interpretation Guide:
{interpretation_text}
Python Implementation: {method.get('python_example', 'N/A')}"""

                doc = IndexDocument(
                    id=doc_id,
                    content=content,
                    metadata={
                        'type': 'statistical_method',
                        'method_name': method['name'],
                        'category': method.get('category', 'general'),
                        'complexity': method.get('complexity', 'medium'),
                        'data_requirements': method.get('data_requirements', []),
                        'output_type': method.get('output_type', 'numeric')
                    },
                    category='statistical_method',
                    domain='statistics'
                )
                self.documents.append(doc)
            
            # 해석 가이드라인
            interpretations = methods_db.get('interpretation_guidelines', [])
            for guideline in interpretations:
                doc_id = f"interpretation_{guideline['context']}"
                content = f"Context: {guideline['context']}\nGuidelines: {guideline['guidelines']}\nCommon Mistakes: {guideline.get('common_mistakes', 'N/A')}"
                
                doc = IndexDocument(
                    id=doc_id,
                    content=content,
                    metadata={
                        'type': 'interpretation_guideline',
                        'context': guideline['context'],
                        'domain': guideline.get('domain', 'general')
                    },
                    category='interpretation_guideline',
                    domain='statistics'
                )
                self.documents.append(doc)
            
            logging.info(f"Added {len(methods)} statistical methods and {len(interpretations)} interpretation guidelines")
            
        except Exception as e:
            logging.error(f"Error adding statistical methods: {e}")
    
    def save_index(self):
        """통계 지식 인덱스 저장"""
        try:
            index_file = self.index_dir / "statistical_knowledge_index.json"
            
            documents_data = [doc.to_dict() for doc in self.documents]
            
            index_data = {
                'documents': documents_data,
                'metadata': {
                    'total_documents': len(self.documents),
                    'categories': list(set(doc.category for doc in self.documents)),
                    'method_count': len([d for d in self.documents if d.metadata.get('type') == 'statistical_method']),
                    'guideline_count': len([d for d in self.documents if d.metadata.get('type') == 'interpretation_guideline']),
                    'created_at': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
            
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
            
            logging.info(f"Statistical knowledge index saved to {index_file}")
            
        except Exception as e:
            logging.error(f"Error saving statistical knowledge index: {e}")


class VectorIndexManager:
    """벡터 인덱스 관리자"""
    
    def __init__(self, index_dir: Path, model_name: str = "all-MiniLM-L6-v2"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        
        # Embedding 모델 초기화
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(model_name)
                self.use_transformers = True
                logging.info(f"Using SentenceTransformers model: {model_name}")
            except Exception as e:
                logging.warning(f"Failed to load SentenceTransformers model: {e}. Using TF-IDF instead.")
                self.use_transformers = False
        else:
            self.use_transformers = False
        
        if not self.use_transformers:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            logging.info("Using TF-IDF vectorization")
        
        # FAISS 인덱스
        self.faiss_index = None
        self.document_store = []
        
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """텍스트 임베딩 생성"""
        if self.use_transformers:
            embeddings = self.embedding_model.encode(texts)
            return embeddings
        else:
            # TF-IDF 사용
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            return tfidf_matrix.toarray()
    
    def build_vector_index(self, documents: List[IndexDocument]):
        """벡터 인덱스 구축"""
        try:
            texts = [doc.content for doc in documents]
            embeddings = self.create_embeddings(texts)
            
            # 문서에 임베딩 저장
            for doc, embedding in zip(documents, embeddings):
                doc.embedding = embedding.tolist()
            
            self.document_store = documents
            
            # FAISS 인덱스 구축 (사용 가능한 경우)
            if FAISS_AVAILABLE and self.use_transformers:
                dimension = embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product (Cosine similarity)
                
                # L2 normalize for cosine similarity
                faiss.normalize_L2(embeddings)
                self.faiss_index.add(embeddings.astype('float32'))
                
                logging.info(f"FAISS index built with {len(documents)} documents")
            else:
                # 간단한 인메모리 저장
                self.embeddings_matrix = embeddings
                logging.info(f"In-memory vector index built with {len(documents)} documents")
            
        except Exception as e:
            logging.error(f"Error building vector index: {e}")
    
    def search(self, query: str, top_k: int = 5, filter_criteria: Dict[str, Any] = None) -> List[SearchResult]:
        """벡터 검색 수행"""
        try:
            # 쿼리 임베딩 생성
            if self.use_transformers:
                query_embedding = self.embedding_model.encode([query])
            else:
                query_embedding = self.vectorizer.transform([query]).toarray()
            
            results = []
            
            if FAISS_AVAILABLE and self.faiss_index is not None:
                # FAISS 검색
                faiss.normalize_L2(query_embedding.astype('float32'))
                scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
                
                for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
                    if idx < len(self.document_store):
                        doc = self.document_store[idx]
                        if self._passes_filter(doc, filter_criteria):
                            result = SearchResult(
                                document=doc,
                                similarity_score=float(score),
                                rank=rank + 1,
                                retrieval_context={'method': 'faiss', 'query': query}
                            )
                            results.append(result)
            else:
                # 코사인 유사도 기반 검색
                similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
                top_indices = similarities.argsort()[-top_k:][::-1]
                
                for rank, idx in enumerate(top_indices):
                    doc = self.document_store[idx]
                    if self._passes_filter(doc, filter_criteria):
                        result = SearchResult(
                            document=doc,
                            similarity_score=float(similarities[idx]),
                            rank=rank + 1,
                            retrieval_context={'method': 'cosine', 'query': query}
                        )
                        results.append(result)
            
            return results[:top_k]
            
        except Exception as e:
            logging.error(f"Error during vector search: {e}")
            return []
    
    def _passes_filter(self, document: IndexDocument, filter_criteria: Dict[str, Any]) -> bool:
        """필터 조건 확인"""
        if not filter_criteria:
            return True
        
        for key, value in filter_criteria.items():
            if key == 'domain' and document.domain != value:
                return False
            elif key == 'category' and document.category != value:
                return False
            elif key == 'type' and document.metadata.get('type') != value:
                return False
        
        return True
    
    def save_index(self, index_name: str):
        """벡터 인덱스 저장"""
        try:
            index_path = self.index_dir / f"{index_name}_vector_index"
            index_path.mkdir(exist_ok=True)
            
            # 문서 저장
            documents_file = index_path / "documents.json"
            documents_data = [doc.to_dict() for doc in self.document_store]
            
            with open(documents_file, 'w', encoding='utf-8') as f:
                json.dump(documents_data, f, ensure_ascii=False, indent=2)
            
            # FAISS 인덱스 저장 (사용 가능한 경우)
            if FAISS_AVAILABLE and self.faiss_index is not None:
                faiss_file = index_path / "faiss_index.bin"
                faiss.write_index(self.faiss_index, str(faiss_file))
            
            # 메타데이터 저장
            metadata = {
                'index_name': index_name,
                'model_name': self.model_name,
                'document_count': len(self.document_store),
                'use_transformers': self.use_transformers,
                'use_faiss': FAISS_AVAILABLE and self.faiss_index is not None,
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            metadata_file = index_path / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logging.info(f"Vector index saved to {index_path}")
            
        except Exception as e:
            logging.error(f"Error saving vector index: {e}")
    
    def load_index(self, index_name: str) -> bool:
        """벡터 인덱스 로드"""
        try:
            index_path = self.index_dir / f"{index_name}_vector_index"
            
            if not index_path.exists():
                logging.warning(f"Vector index not found: {index_path}")
                return False
            
            # 문서 로드
            documents_file = index_path / "documents.json"
            with open(documents_file, 'r', encoding='utf-8') as f:
                documents_data = json.load(f)
            
            self.document_store = []
            for doc_data in documents_data:
                doc = IndexDocument(**doc_data)
                self.document_store.append(doc)
            
            # 임베딩 매트릭스 재구성
            if self.document_store and self.document_store[0].embedding:
                embeddings = np.array([doc.embedding for doc in self.document_store])
                self.embeddings_matrix = embeddings
            
            # FAISS 인덱스 로드 (사용 가능한 경우)
            faiss_file = index_path / "faiss_index.bin"
            if FAISS_AVAILABLE and faiss_file.exists():
                self.faiss_index = faiss.read_index(str(faiss_file))
            
            logging.info(f"Vector index loaded from {index_path}: {len(self.document_store)} documents")
            return True
            
        except Exception as e:
            logging.error(f"Error loading vector index: {e}")
            return False


class RAGIndexBuilder:
    """Enhanced RAG 시스템을 위한 통합 인덱스 구축기"""
    
    def __init__(self, base_index_dir: str = "resources/rag_index"):
        self.base_index_dir = Path(base_index_dir)
        self.base_index_dir.mkdir(parents=True, exist_ok=True)
        
        # 개별 인덱서들
        self.business_indexer = BusinessContextIndexer(self.base_index_dir)
        self.schema_indexer = SchemaContextIndexer(self.base_index_dir)
        self.statistical_indexer = StatisticalKnowledgeIndexer(self.base_index_dir)
        self.vector_manager = VectorIndexManager(self.base_index_dir)
        
        # 통합 인덱스 정보
        self.index_registry = {}
        
    def build_all_indices(self, knowledge_base: Dict[str, Any]):
        """모든 RAG 인덱스 구축"""
        try:
            logging.info("Starting RAG index building process...")
            
            # 1. 비즈니스 컨텍스트 인덱스
            if 'business_domains' in knowledge_base:
                for domain, domain_data in knowledge_base['business_domains'].items():
                    self.business_indexer.add_business_domain(domain, domain_data)
                self.business_indexer.save_index()
            
            # 2. 스키마 컨텍스트 인덱스
            if 'database_schemas' in knowledge_base:
                for schema_name, schema_data in knowledge_base['database_schemas'].items():
                    self.schema_indexer.add_database_schema(schema_name, schema_data)
                self.schema_indexer.save_index()
            
            # 3. 통계 지식 인덱스
            if 'statistical_methods' in knowledge_base:
                self.statistical_indexer.add_statistical_methods(knowledge_base['statistical_methods'])
                self.statistical_indexer.save_index()
            
            # 4. 통합 벡터 인덱스
            all_documents = []
            all_documents.extend(self.business_indexer.documents)
            all_documents.extend(self.schema_indexer.documents)
            all_documents.extend(self.statistical_indexer.documents)
            
            if all_documents:
                self.vector_manager.build_vector_index(all_documents)
                self.vector_manager.save_index("unified_rag_index")
            
            # 5. 인덱스 레지스트리 업데이트
            self._update_index_registry()
            
            logging.info(f"RAG index building completed. Total documents: {len(all_documents)}")
            return True
            
        except Exception as e:
            logging.error(f"Error building RAG indices: {e}")
            return False
    
    def _update_index_registry(self):
        """인덱스 레지스트리 업데이트"""
        self.index_registry = {
            'unified_rag_index': {
                'type': 'vector',
                'document_count': len(self.vector_manager.document_store),
                'categories': list(set(doc.category for doc in self.vector_manager.document_store)),
                'domains': list(set(doc.domain for doc in self.vector_manager.document_store)),
                'last_updated': datetime.now().isoformat()
            },
            'business_context': {
                'type': 'structured',
                'document_count': len(self.business_indexer.documents),
                'last_updated': datetime.now().isoformat()
            },
            'schema_context': {
                'type': 'structured',
                'document_count': len(self.schema_indexer.documents),
                'last_updated': datetime.now().isoformat()
            },
            'statistical_knowledge': {
                'type': 'structured',
                'document_count': len(self.statistical_indexer.documents),
                'last_updated': datetime.now().isoformat()
            }
        }
        
        # 레지스트리 저장
        registry_file = self.base_index_dir / "index_registry.json"
        with open(registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.index_registry, f, ensure_ascii=False, indent=2)
    
    def get_index_status(self) -> Dict[str, Any]:
        """인덱스 상태 정보 반환"""
        try:
            registry_file = self.base_index_dir / "index_registry.json"
            if registry_file.exists():
                with open(registry_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return self.index_registry
        except Exception as e:
            logging.error(f"Error getting index status: {e}")
            return {}


# 사용 예시 및 기본 지식 베이스
DEFAULT_KNOWLEDGE_BASE = {
    "business_domains": {
        "finance": {
            "concepts": [
                {
                    "name": "Risk Assessment",
                    "description": "Financial risk evaluation and management",
                    "keywords": ["risk", "volatility", "VaR", "portfolio"],
                    "importance": 0.9
                },
                {
                    "name": "Performance Analysis",
                    "description": "Investment and portfolio performance evaluation",
                    "keywords": ["returns", "performance", "benchmark", "alpha"],
                    "importance": 0.8
                }
            ],
            "metrics": [
                {
                    "name": "Sharpe Ratio",
                    "definition": "Risk-adjusted return measure",
                    "calculation": "(Portfolio Return - Risk Free Rate) / Portfolio Standard Deviation",
                    "business_impact": "Higher values indicate better risk-adjusted performance"
                }
            ],
            "analysis_patterns": [
                {
                    "name": "Portfolio Optimization",
                    "description": "Optimize asset allocation for risk-return tradeoff",
                    "when_to_use": "When building or rebalancing investment portfolios",
                    "expected_outcomes": "Optimal asset weights, risk metrics"
                }
            ]
        },
        "healthcare": {
            "concepts": [
                {
                    "name": "Clinical Trial Analysis",
                    "description": "Statistical analysis of medical research data",
                    "keywords": ["efficacy", "safety", "endpoints", "biostatistics"],
                    "importance": 0.95
                }
            ],
            "metrics": [
                {
                    "name": "Relative Risk",
                    "definition": "Ratio of probability of event in exposed vs unexposed groups",
                    "calculation": "Risk in exposed group / Risk in control group",
                    "business_impact": "Measures treatment effect magnitude"
                }
            ],
            "analysis_patterns": [
                {
                    "name": "Survival Analysis",
                    "description": "Time-to-event analysis for patient outcomes",
                    "when_to_use": "When analyzing patient survival or treatment duration",
                    "expected_outcomes": "Survival curves, hazard ratios"
                }
            ]
        }
    },
    "statistical_methods": {
        "methods": [
            {
                "name": "Linear Regression",
                "category": "regression",
                "description": "Models linear relationship between variables",
                "use_cases": ["prediction", "relationship_analysis", "trend_analysis"],
                "assumptions": [
                    "Linear relationship between variables",
                    "Independence of observations",
                    "Homoscedasticity",
                    "Normal distribution of residuals"
                ],
                "interpretation_guide": [
                    "Coefficients represent change in Y per unit change in X",
                    "R-squared shows proportion of variance explained",
                    "P-values test significance of coefficients"
                ],
                "complexity": "basic",
                "python_example": "from sklearn.linear_model import LinearRegression"
            },
            {
                "name": "ANOVA",
                "category": "hypothesis_testing",
                "description": "Compares means across multiple groups",
                "use_cases": ["group_comparison", "treatment_effects", "experimental_design"],
                "assumptions": [
                    "Independence of observations",
                    "Normal distribution within groups",
                    "Equal variances across groups"
                ],
                "interpretation_guide": [
                    "F-statistic tests if group means differ significantly",
                    "Post-hoc tests identify which groups differ",
                    "Effect size measures practical significance"
                ],
                "complexity": "intermediate"
            }
        ],
        "interpretation_guidelines": [
            {
                "context": "p-value_interpretation",
                "guidelines": "P-values indicate strength of evidence against null hypothesis. Values < 0.05 typically considered statistically significant.",
                "common_mistakes": "Interpreting p-values as probability that hypothesis is true"
            }
        ]
    }
}


if __name__ == "__main__":
    # 예시 실행
    logging.basicConfig(level=logging.INFO)
    
    builder = RAGIndexBuilder()
    success = builder.build_all_indices(DEFAULT_KNOWLEDGE_BASE)
    
    if success:
        print("RAG indices built successfully!")
        status = builder.get_index_status()
        print(f"Index status: {status}")
    else:
        print("Failed to build RAG indices.") 