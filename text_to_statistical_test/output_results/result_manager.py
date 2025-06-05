"""
ResultManager: 통계 분석 결과 저장 및 관리

Enhanced RAG 기반 Multi-turn 통계 분석 시스템의 모든 출력 결과를
체계적으로 저장하고 관리하는 핵심 클래스입니다.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import logging
import os
import shutil


class ResultManager:
    """
    통계 분석 결과의 체계적 저장 및 관리
    
    세션별로 구조화된 디렉토리에 모든 분석 결과를 저장하고,
    결과의 검색, 백업, 정리 기능을 제공합니다.
    """
    
    def __init__(self, base_output_dir: str = "output_results"):
        """
        ResultManager 초기화
        
        Args:
            base_output_dir: 기본 출력 디렉토리 경로
        """
        self.base_output_dir = Path(base_output_dir)
        self.logger = logging.getLogger(__name__)
        
        # 기본 디렉토리 구조 생성
        self._create_directory_structure()
        
        # 현재 세션 정보
        self.current_session_id = None
        self.current_session_dir = None
        
    def _create_directory_structure(self):
        """기본 디렉토리 구조 생성"""
        subdirs = [
            'sessions',           # 세션별 결과
            'statistical_results', # 통계 분석 결과
            'reports',            # 생성된 보고서
            'visualizations',     # 시각화 결과
            'logs',              # 세션 로그
            'exports',           # 내보내기 결과
            'archived'           # 아카이브된 결과
        ]
        
        for subdir in subdirs:
            (self.base_output_dir / subdir).mkdir(parents=True, exist_ok=True)
            
        self.logger.info(f"디렉토리 구조 생성 완료: {self.base_output_dir}")
    
    def start_session(self, session_id: str) -> str:
        """
        새로운 세션 시작 및 세션별 디렉토리 생성
        
        Args:
            session_id: 세션 식별자
            
        Returns:
            str: 생성된 세션 디렉토리 경로
        """
        self.current_session_id = session_id
        self.current_session_dir = self.base_output_dir / 'sessions' / session_id
        
        # 세션 디렉토리 구조 생성
        session_subdirs = [
            'analysis_results',   # 분석 결과
            'intermediate_data',  # 중간 처리 데이터
            'code_execution',     # 실행된 코드
            'user_interactions',  # 사용자 상호작용 로그
            'rag_context',        # RAG 검색 컨텍스트
            'reports'             # 생성된 보고서
        ]
        
        for subdir in session_subdirs:
            (self.current_session_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # 세션 메타데이터 생성
        session_metadata = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'analysis_count': 0
        }
        
        self.save_session_metadata(session_metadata)
        
        self.logger.info(f"세션 시작: {session_id} -> {self.current_session_dir}")
        return str(self.current_session_dir)
    
    def save_statistical_result(self, result_data: Dict[str, Any], 
                              analysis_type: str = None, 
                              metadata: Dict[str, Any] = None) -> str:
        """
        통계 분석 결과 저장
        
        Args:
            result_data: 통계 분석 결과 데이터
            analysis_type: 분석 유형 (t-test, anova, regression 등)
            metadata: 추가 메타데이터
            
        Returns:
            str: 저장된 파일 경로
        """
        if not self.current_session_dir:
            raise ValueError("활성 세션이 없습니다. start_session()을 먼저 호출하세요.")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analysis_type = analysis_type or 'unknown'
        
        # 결과 파일명 생성
        filename = f"{analysis_type}_{timestamp}.json"
        filepath = self.current_session_dir / 'analysis_results' / filename
        
        # 결과 데이터 구조화
        structured_result = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': analysis_type,
                'session_id': self.current_session_id,
                **(metadata or {})
            },
            'statistical_results': result_data,
            'summary': self._generate_result_summary(result_data, analysis_type)
        }
        
        # JSON 파일로 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(structured_result, f, indent=2, ensure_ascii=False, default=str)
        
        # 전역 통계 결과 디렉토리에도 복사
        global_filepath = self.base_output_dir / 'statistical_results' / filename
        shutil.copy2(filepath, global_filepath)
        
        self.logger.info(f"통계 결과 저장: {filepath}")
        return str(filepath)
    
    def save_visualization(self, plot_data: Any, plot_type: str, 
                         filename: str = None, format: str = 'png') -> str:
        """
        시각화 결과 저장
        
        Args:
            plot_data: 플롯 데이터 (matplotlib figure, plotly figure 등)
            plot_type: 플롯 유형 (histogram, scatter, boxplot 등)
            filename: 파일명 (자동 생성 가능)
            format: 파일 포맷 (png, svg, html)
            
        Returns:
            str: 저장된 파일 경로
        """
        if not self.current_session_dir:
            raise ValueError("활성 세션이 없습니다.")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = filename or f"{plot_type}_{timestamp}.{format}"
        
        # 세션별 시각화 디렉토리
        viz_dir = self.current_session_dir / 'analysis_results' / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        filepath = viz_dir / filename
        
        # 플롯 타입에 따른 저장 처리
        try:
            if hasattr(plot_data, 'savefig'):  # matplotlib
                plot_data.savefig(filepath, dpi=300, bbox_inches='tight')
            elif hasattr(plot_data, 'write_html'):  # plotly
                plot_data.write_html(str(filepath))
            elif hasattr(plot_data, 'write_image'):  # plotly
                plot_data.write_image(str(filepath))
            else:
                self.logger.warning(f"지원되지 않는 플롯 타입: {type(plot_data)}")
                return None
                
            # 전역 시각화 디렉토리에도 복사
            global_filepath = self.base_output_dir / 'visualizations' / filename
            shutil.copy2(filepath, global_filepath)
            
            self.logger.info(f"시각화 저장: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"시각화 저장 실패: {e}")
            return None
    
    def save_code_execution_result(self, code: str, output: str, 
                                 execution_time: float = None,
                                 error: str = None) -> str:
        """
        코드 실행 결과 저장
        
        Args:
            code: 실행된 코드
            output: 실행 결과 출력
            execution_time: 실행 시간
            error: 에러 메시지 (있는 경우)
            
        Returns:
            str: 저장된 파일 경로
        """
        if not self.current_session_dir:
            raise ValueError("활성 세션이 없습니다.")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"code_execution_{timestamp}.json"
        filepath = self.current_session_dir / 'code_execution' / filename
        
        execution_data = {
            'timestamp': datetime.now().isoformat(),
            'code': code,
            'output': output,
            'execution_time': execution_time,
            'error': error,
            'success': error is None
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(execution_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"코드 실행 결과 저장: {filepath}")
        return str(filepath)
    
    def save_rag_context(self, business_context: Dict, schema_context: Dict,
                        query: str, recommendations: List[Dict]) -> str:
        """
        RAG 시스템 컨텍스트 저장
        
        Args:
            business_context: 비즈니스 컨텍스트
            schema_context: 스키마 컨텍스트
            query: 사용자 쿼리
            recommendations: AI 추천 결과
            
        Returns:
            str: 저장된 파일 경로
        """
        if not self.current_session_dir:
            raise ValueError("활성 세션이 없습니다.")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"rag_context_{timestamp}.json"
        filepath = self.current_session_dir / 'rag_context' / filename
        
        rag_data = {
            'timestamp': datetime.now().isoformat(),
            'user_query': query,
            'business_context': business_context,
            'schema_context': schema_context,
            'ai_recommendations': recommendations
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(rag_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"RAG 컨텍스트 저장: {filepath}")
        return str(filepath)
    
    def save_user_interaction(self, interaction_data: Dict) -> str:
        """
        사용자 상호작용 로그 저장
        
        Args:
            interaction_data: 상호작용 데이터
            
        Returns:
            str: 저장된 파일 경로
        """
        if not self.current_session_dir:
            raise ValueError("활성 세션이 없습니다.")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"interaction_{timestamp}.json"
        filepath = self.current_session_dir / 'user_interactions' / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(interaction_data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def save_session_metadata(self, metadata: Dict) -> str:
        """
        세션 메타데이터 저장
        
        Args:
            metadata: 세션 메타데이터
            
        Returns:
            str: 저장된 파일 경로
        """
        if not self.current_session_dir:
            raise ValueError("활성 세션이 없습니다.")
        
        filepath = self.current_session_dir / 'session_metadata.json'
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def export_session_results(self, export_format: str = 'zip') -> str:
        """
        세션 결과를 지정된 형식으로 내보내기
        
        Args:
            export_format: 내보내기 형식 (zip, tar)
            
        Returns:
            str: 내보내기 파일 경로
        """
        if not self.current_session_dir:
            raise ValueError("활성 세션이 없습니다.")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_filename = f"{self.current_session_id}_{timestamp}"
        export_dir = self.base_output_dir / 'exports'
        
        if export_format == 'zip':
            import zipfile
            export_path = export_dir / f"{export_filename}.zip"
            
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in self.current_session_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(self.current_session_dir)
                        zipf.write(file_path, arcname)
        
        elif export_format == 'tar':
            import tarfile
            export_path = export_dir / f"{export_filename}.tar.gz"
            
            with tarfile.open(export_path, 'w:gz') as tar:
                tar.add(self.current_session_dir, arcname=self.current_session_id)
        
        self.logger.info(f"세션 결과 내보내기 완료: {export_path}")
        return str(export_path)
    
    def get_session_summary(self) -> Dict:
        """
        현재 세션의 요약 정보 반환
        
        Returns:
            Dict: 세션 요약 정보
        """
        if not self.current_session_dir:
            return {'error': '활성 세션이 없습니다.'}
        
        # 각 디렉토리의 파일 수 계산
        summary = {
            'session_id': self.current_session_id,
            'session_dir': str(self.current_session_dir),
            'file_counts': {}
        }
        
        for subdir in ['analysis_results', 'code_execution', 'user_interactions', 'rag_context']:
            subdir_path = self.current_session_dir / subdir
            if subdir_path.exists():
                summary['file_counts'][subdir] = len(list(subdir_path.glob('*')))
        
        return summary
    
    def _generate_result_summary(self, result_data: Dict, analysis_type: str) -> Dict:
        """
        분석 결과의 요약 정보 생성
        
        Args:
            result_data: 원본 결과 데이터
            analysis_type: 분석 유형
            
        Returns:
            Dict: 요약 정보
        """
        summary = {
            'analysis_type': analysis_type,
            'has_statistical_test': 'test_statistic' in result_data or 'p_value' in result_data,
            'has_confidence_interval': 'confidence_interval' in result_data,
            'significant': False
        }
        
        # p-value 기준 유의성 판단
        if 'p_value' in result_data:
            try:
                p_val = float(result_data['p_value'])
                summary['significant'] = p_val < 0.05
            except (ValueError, TypeError):
                pass
        
        return summary
    
    def list_available_sessions(self) -> List[Dict]:
        """
        사용 가능한 세션 목록 반환
        
        Returns:
            List[Dict]: 세션 정보 목록
        """
        sessions_dir = self.base_output_dir / 'sessions'
        sessions = []
        
        for session_dir in sessions_dir.iterdir():
            if session_dir.is_dir():
                metadata_file = session_dir / 'session_metadata.json'
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        sessions.append(metadata)
                    except Exception as e:
                        self.logger.warning(f"세션 메타데이터 읽기 실패: {session_dir.name} - {e}")
        
        return sorted(sessions, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def cleanup_old_sessions(self, days_to_keep: int = 30):
        """
        오래된 세션 정리
        
        Args:
            days_to_keep: 보관할 일수
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        sessions_dir = self.base_output_dir / 'sessions'
        archived_count = 0
        
        for session_dir in sessions_dir.iterdir():
            if session_dir.is_dir():
                # 세션 생성 시간 확인
                if session_dir.stat().st_ctime < cutoff_date.timestamp():
                    # 아카이브 디렉토리로 이동
                    archive_dir = self.base_output_dir / 'archived' / session_dir.name
                    shutil.move(str(session_dir), str(archive_dir))
                    archived_count += 1
        
        self.logger.info(f"오래된 세션 {archived_count}개를 아카이브했습니다.")
        return archived_count 