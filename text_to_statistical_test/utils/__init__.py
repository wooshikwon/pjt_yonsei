"""
Utils Package

Enhanced RAG 시스템 기반 통계 분석을 위한 유틸리티 함수들
"""

# Enhanced RAG 워크플로우 관련
from .workflow_utils import (
    # 워크플로우 진행 함수들
    handle_data_selection_workflow,
    handle_natural_language_request_workflow,
    handle_rag_activation_workflow,
    handle_ai_recommendation_workflow,
    handle_method_confirmation_workflow,
    handle_session_continuation_workflow,
    
    # main.py에서 사용하는 핵심 워크플로우 함수들
    run_interactive_mode,  # main.py 호환성 wrapper 함수
    run_enhanced_multiturn_workflow  # 실제 워크플로우 함수
)

# 시스템 설정 관련 - main.py에서 사용
from .system_setup import (
    setup_dependencies,  # main.py에서 사용
    create_agent_instance,
    setup_logging
)

# config.settings에서 validate_settings import
from config.settings import validate_settings  # main.py에서 사용

# 분석 추천 관련
from .analysis_recommender import AnalysisRecommender, AnalysisRecommendation

# 데이터 관련 유틸리티
from .data_utils import (
    get_available_data_files,
    validate_data_file,
    get_file_info,
    format_file_size,
    preview_selected_data,
    analyze_data_structure,
    get_data_summary_for_rag
)

# UI 관련 헬퍼 함수들
from .ui_helpers import (
    print_welcome_message,  # main.py에서 사용
    print_welcome,
    print_enhanced_rag_features,
    print_usage_examples,
    print_analysis_guide,
    display_file_selection_menu,
    print_session_status,
    display_rag_search_results,
    display_ai_recommendations,
    display_analysis_progress,
    print_workflow_completion_message,
    print_error_message,
    print_help_message,
    ask_user_confirmation
)

__all__ = [
    # main.py에서 사용하는 핵심 함수들
    'setup_dependencies',
    'validate_settings', 
    'run_interactive_mode',
    'print_welcome_message',
    
    # 워크플로우 함수들
    'handle_data_selection_workflow',
    'handle_natural_language_request_workflow', 
    'handle_rag_activation_workflow',
    'handle_ai_recommendation_workflow',
    'handle_method_confirmation_workflow',
    'handle_session_continuation_workflow',
    'run_enhanced_multiturn_workflow',
    
    # 시스템 설정
    'create_agent_instance',
    'setup_logging',
    
    # 분석 추천
    'AnalysisRecommender',
    'AnalysisRecommendation',
    
    # 데이터 유틸리티
    'get_available_data_files',
    'validate_data_file',
    'get_file_info',
    'format_file_size',
    'preview_selected_data',
    'analyze_data_structure',
    'get_data_summary_for_rag',
    
    # UI 헬퍼
    'print_welcome',
    'print_enhanced_rag_features',
    'print_usage_examples',
    'print_analysis_guide',
    'display_file_selection_menu',
    'print_session_status',
    'display_rag_search_results',
    'display_ai_recommendations',
    'display_analysis_progress',
    'print_workflow_completion_message',
    'print_error_message',
    'print_help_message',
    'ask_user_confirmation'
] 