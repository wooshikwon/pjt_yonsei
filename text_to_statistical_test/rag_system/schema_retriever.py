"""
Enhanced RAG System: Database Schema Retriever

데이터베이스 스키마 구조 검색을 위한 RAG 시스템 컴포넌트
- 테이블 구조 정의 (schema_definitions.json)
- 테이블 관계 매핑 (relationship_maps.json)
- 컬럼 상세 설명 (column_descriptions.json)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional


class SchemaRetriever:
    """
    DB 스키마 및 테이블 관계 정보 검색 엔진
    
    데이터베이스 구조 정보를 활용하여 자연어 분석 요청에 적합한
    스키마 컨텍스트를 제공합니다.
    """
    
    def __init__(self, schema_path: str = "input_data/metadata/database_schemas"):
        """
        SchemaRetriever 초기화
        
        Args:
            schema_path: DB 스키마 메타데이터 디렉토리 경로
        """
        self.schema_path = Path(schema_path)
        self.logger = logging.getLogger(__name__)
        
        # 스키마 정보 저장소
        self.schema_definitions = {}
        self.relationship_maps = {}
        self.column_descriptions = {}
        
        # 초기화
        self._load_schema_information()
        
    def _load_schema_information(self):
        """스키마 정보 로딩"""
        try:
            # 테이블 구조 정의 로딩
            definitions_path = self.schema_path / "schema_definitions.json"
            if definitions_path.exists():
                with open(definitions_path, 'r', encoding='utf-8') as f:
                    self.schema_definitions = json.load(f)
                self.logger.info(f"테이블 구조 정의 로딩 완료: {len(self.schema_definitions)}개 테이블")
            
            # 테이블 관계 매핑 로딩
            relationships_path = self.schema_path / "relationship_maps.json"
            if relationships_path.exists():
                with open(relationships_path, 'r', encoding='utf-8') as f:
                    self.relationship_maps = json.load(f)
                self.logger.info("테이블 관계 매핑 로딩 완료")
            
            # 컬럼 상세 설명 로딩
            descriptions_path = self.schema_path / "column_descriptions.json"
            if descriptions_path.exists():
                with open(descriptions_path, 'r', encoding='utf-8') as f:
                    self.column_descriptions = json.load(f)
                self.logger.info("컬럼 상세 설명 로딩 완료")
                
        except Exception as e:
            self.logger.error(f"스키마 정보 로딩 실패: {e}")
    
    def search_table_schema(self, table_name: str) -> Dict:
        """
        특정 테이블의 스키마 정보 검색
        
        Args:
            table_name: 검색할 테이블명
            
        Returns:
            Dict: 테이블 스키마 정보
        """
        schema_info = {}
        
        # 테이블 정의 검색
        if table_name in self.schema_definitions:
            schema_info['definition'] = self.schema_definitions[table_name]
        
        # 컬럼 설명 검색
        if table_name in self.column_descriptions:
            schema_info['column_details'] = self.column_descriptions[table_name]
        
        # 관련 테이블 검색
        related_tables = self.find_related_tables(table_name)
        if related_tables:
            schema_info['related_tables'] = related_tables
            
        return schema_info
    
    def search_column_relationships(self, columns: List[str]) -> List[Dict]:
        """
        컬럼 관계 정보 검색
        
        Args:
            columns: 검색할 컬럼 리스트
            
        Returns:
            List[Dict]: 컬럼 관계 정보
        """
        relationships = []
        
        if 'relationships' not in self.relationship_maps:
            return relationships
        
        for relationship in self.relationship_maps['relationships']:
            # 관계에 포함된 컬럼들 확인
            join_condition = relationship.get('join_condition', '')
            
            for column in columns:
                if column in join_condition:
                    relationships.append({
                        'column': column,
                        'relationship': relationship,
                        'business_context': relationship.get('business_context', ''),
                        'relationship_type': relationship.get('relationship_type', '')
                    })
        
        return relationships
    
    def get_schema_context(self, data_columns: List[str], query: str = "") -> Dict:
        """
        데이터 컬럼 기반 스키마 컨텍스트 생성
        
        Args:
            data_columns: 데이터의 컬럼 리스트
            query: 분석 요청 쿼리 (선택사항)
            
        Returns:
            Dict: 스키마 컨텍스트 정보
        """
        context = {
            'matched_tables': {},
            'column_details': {},
            'relationships': [],
            'analytical_patterns': {},
            'suggestions': []
        }
        
        try:
            # 1. 컬럼명 매칭으로 테이블 식별
            matched_tables = self._match_tables_by_columns(data_columns)
            context['matched_tables'] = matched_tables
            
            # 2. 매칭된 컬럼들의 상세 정보
            for table_name, matched_columns in matched_tables.items():
                if table_name in self.column_descriptions:
                    table_descriptions = self.column_descriptions[table_name]
                    for column in matched_columns:
                        if column in table_descriptions:
                            context['column_details'][column] = table_descriptions[column]
            
            # 3. 컬럼 관계 정보
            context['relationships'] = self.search_column_relationships(data_columns)
            
            # 4. 분석 패턴 추천
            analytical_patterns = self._get_analytical_patterns(data_columns, query)
            context['analytical_patterns'] = analytical_patterns
            
            # 5. 스키마 기반 분석 제안
            context['suggestions'] = self._generate_schema_suggestions(matched_tables, data_columns)
            
        except Exception as e:
            self.logger.error(f"스키마 컨텍스트 생성 오류: {e}")
        
        return context
    
    def find_related_tables(self, primary_table: str) -> List[Dict]:
        """
        주테이블과 관련된 테이블들 검색
        
        Args:
            primary_table: 기준 테이블명
            
        Returns:
            List[Dict]: 관련 테이블 정보
        """
        related_tables = []
        
        if 'relationships' not in self.relationship_maps:
            return related_tables
        
        for relationship in self.relationship_maps['relationships']:
            parent_table = relationship.get('parent_table', '')
            child_table = relationship.get('child_table', '')
            
            if primary_table == parent_table:
                related_tables.append({
                    'table': child_table,
                    'relationship_type': relationship.get('relationship_type', ''),
                    'join_condition': relationship.get('join_condition', ''),
                    'business_context': relationship.get('business_context', ''),
                    'direction': 'child'
                })
            elif primary_table == child_table:
                related_tables.append({
                    'table': parent_table,
                    'relationship_type': relationship.get('relationship_type', ''),
                    'join_condition': relationship.get('join_condition', ''),
                    'business_context': relationship.get('business_context', ''),
                    'direction': 'parent'
                })
        
        return related_tables
    
    def _match_tables_by_columns(self, data_columns: List[str]) -> Dict[str, List[str]]:
        """데이터 컬럼명으로 테이블 매칭"""
        matched_tables = {}
        
        for table_name, table_info in self.schema_definitions.items():
            if 'columns' not in table_info:
                continue
                
            table_columns = table_info['columns'].keys()
            matched_columns = []
            
            for data_col in data_columns:
                # 정확한 매칭
                if data_col in table_columns:
                    matched_columns.append(data_col)
                # 부분 매칭 (대소문자 무시)
                elif any(data_col.lower() == tc.lower() for tc in table_columns):
                    matched_columns.append(data_col)
            
            if matched_columns:
                matched_tables[table_name] = matched_columns
        
        return matched_tables
    
    def _get_analytical_patterns(self, data_columns: List[str], query: str) -> Dict:
        """분석 패턴 추천"""
        patterns = {}
        
        if 'analytical_patterns' not in self.relationship_maps:
            return patterns
        
        analytical_patterns = self.relationship_maps['analytical_patterns']
        
        # 쿼리와 컬럼 기반으로 적합한 패턴 찾기
        for pattern_name, pattern_info in analytical_patterns.items():
            relevance_score = 0
            
            # 컬럼 매칭 점수
            pattern_columns = pattern_info.get('common_metrics', []) + pattern_info.get('typical_groupby', [])
            matching_columns = set(data_columns) & set(pattern_columns)
            if matching_columns:
                relevance_score += len(matching_columns) / len(data_columns)
            
            # 쿼리 키워드 매칭
            pattern_keywords = pattern_name.split('_')
            query_lower = query.lower()
            for keyword in pattern_keywords:
                if keyword in query_lower:
                    relevance_score += 0.2
            
            if relevance_score > 0.3:
                patterns[pattern_name] = {
                    'pattern_info': pattern_info,
                    'relevance_score': relevance_score,
                    'matching_columns': list(matching_columns)
                }
        
        return patterns
    
    def _generate_schema_suggestions(self, matched_tables: Dict, data_columns: List[str]) -> List[str]:
        """스키마 기반 분석 제안 생성"""
        suggestions = []
        
        # 테이블이 매칭된 경우
        if matched_tables:
            for table_name, columns in matched_tables.items():
                if table_name in self.schema_definitions:
                    table_info = self.schema_definitions[table_name]
                    
                    # 비즈니스 규칙 기반 제안
                    business_rules = table_info.get('business_rules', [])
                    for rule in business_rules:
                        suggestions.append(f"비즈니스 규칙 고려: {rule}")
                    
                    # 인덱스 정보 기반 제안
                    indexes = table_info.get('indexes', [])
                    for index in indexes:
                        if index in data_columns:
                            suggestions.append(f"인덱스 컬럼 '{index}' 활용하여 그룹별 분석 가능")
        
        # 컬럼 타입 기반 제안
        for table_name, columns in matched_tables.items():
            if table_name in self.schema_definitions:
                table_columns = self.schema_definitions[table_name].get('columns', {})
                
                for column in columns:
                    if column in table_columns:
                        col_info = table_columns[column]
                        col_type = col_info.get('type', '').upper()
                        
                        if 'DECIMAL' in col_type or 'FLOAT' in col_type:
                            suggestions.append(f"숫자형 컬럼 '{column}': 평균, 합계, 분산 분석 적합")
                        elif 'VARCHAR' in col_type:
                            suggestions.append(f"범주형 컬럼 '{column}': 그룹별 비교 분석 적합")
                        elif 'DATE' in col_type:
                            suggestions.append(f"날짜형 컬럼 '{column}': 시계열 분석 고려")
        
        return suggestions
    
    def get_schema_context_summary(self) -> Dict:
        """스키마 컨텍스트 요약 정보 반환"""
        return {
            'tables_available': list(self.schema_definitions.keys()),
            'relationships_count': len(self.relationship_maps.get('relationships', [])),
            'analytical_patterns_count': len(self.relationship_maps.get('analytical_patterns', {})),
            'has_column_descriptions': bool(self.column_descriptions),
            'schema_path': str(self.schema_path)
        } 