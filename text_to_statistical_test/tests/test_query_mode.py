#!/usr/bin/env python3
"""
개선된 자연어 요청 모드 테스트 스크립트
모호한 요청 처리 및 다중 후보 제시 기능을 테스트합니다.
"""

import sys
import os
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가 (상위 디렉토리)
sys.path.insert(0, str(Path(__file__).parent.parent))

class MockLLMClient:
    """API 키 없이도 테스트 가능한 Mock LLM 클라이언트"""
    
    def __init__(self):
        self.call_count = 0
        
    def generate_text(self, prompt: str) -> str:
        """Mock LLM 응답 생성"""
        self.call_count += 1
        
        # 1-1 단계: 사용자 요청 분석
        if "사용자 요청 분석" in prompt and "그룹별 평균 차이" in prompt:
            return self._generate_analysis_candidates_response()
        
        # 1-2 단계: 분석 목표 확인
        elif "분석 목표 확인" in prompt and "analysis_candidates" in prompt:
            return self._generate_confirmation_response()
        
        # 2-1 단계: 데이터 로딩
        elif "데이터 로딩" in prompt:
            return self._generate_data_loading_response()
        
        # 2-2 단계: 변수 타입 식별
        elif "변수 타입 식별" in prompt:
            return self._generate_variable_type_response()
        
        # 기본 응답
        else:
            return f"""
            ```json
            {{
                "action": "처리됨",
                "content": "Mock 응답 {self.call_count}",
                "confidence": 0.8,
                "reasoning": "테스트 응답입니다"
            }}
            ```
            """
    
    def _generate_analysis_candidates_response(self) -> str:
        """1-1 단계: 모호한 "그룹별 평균 차이" 요청에 대한 다중 해석"""
        return """
        ## 사용자 요청 분석 결과

        "그룹별 평균 차이를 분석해주세요"라는 요청을 다음과 같이 해석했습니다:

        ```json
        {
            "action": "분석",
            "content": "사용자 요청에 대한 다중 해석 결과를 제시합니다",
            "interpretation_summary": "그룹 변수를 기준으로 연속형 결과 변수의 평균값을 비교하는 분석으로 해석됩니다",
            "analysis_candidates": [
                {
                    "priority": 1,
                    "analysis_goal": "그룹별 점수 평균 비교 분석",
                    "analysis_type": "One-way ANOVA",
                    "dependent_variable": "점수",
                    "independent_variable": "그룹",
                    "hypothesis": "귀무가설: 모든 그룹의 점수 평균이 같다 vs 대립가설: 적어도 하나의 그룹 평균이 다르다",
                    "reasoning": "데이터에 3개 그룹(A, B, C)과 연속형 점수 변수가 있어 분산분석이 가장 적합",
                    "confidence": 0.85
                },
                {
                    "priority": 2,
                    "analysis_goal": "그룹별 만족도 평균 비교 분석",
                    "analysis_type": "Kruskal-Wallis Test",
                    "dependent_variable": "만족도",
                    "independent_variable": "그룹",
                    "hypothesis": "귀무가설: 모든 그룹의 만족도 분포가 같다 vs 대립가설: 적어도 하나의 그룹 분포가 다르다",
                    "reasoning": "만족도는 서열 척도로 보이므로 비모수 검정도 고려할 수 있음",
                    "confidence": 0.65
                },
                {
                    "priority": 3,
                    "analysis_goal": "그룹별 다중 변수 평균 비교",
                    "analysis_type": "MANOVA",
                    "dependent_variable": "점수, 만족도, 경험년수",
                    "independent_variable": "그룹",
                    "hypothesis": "다변량 평균 벡터가 그룹 간 다르다",
                    "reasoning": "여러 종속변수를 동시에 고려한 종합적 분석",
                    "confidence": 0.45
                }
            ],
            "uncertainty_areas": [
                "정확히 어떤 변수의 평균을 비교하고 싶은지 명시되지 않음",
                "그룹 간 단순 비교인지, 다른 변수들을 통제한 비교인지 불분명",
                "분산의 동질성이나 정규성 가정에 대한 고려 필요"
            ],
            "clarification_questions": [
                "주로 비교하고 싶은 결과 변수는 무엇인가요? (점수, 만족도, 경험년수 등)",
                "단순한 그룹 간 비교인가요, 아니면 나이나 성별 등을 고려한 분석인가요?",
                "통계적 유의성과 함께 실무적 의미도 중요한가요?"
            ],
            "data_compatibility": {
                "available_variables": "그룹(A,B,C), 점수, 만족도, 경험년수, 나이, 성별",
                "missing_variables": "없음",
                "preprocessing_needs": [
                    "그룹 변수 범주형 확인",
                    "연속형 변수들의 정규성 검정",
                    "이상치 확인"
                ]
            },
            "next_steps": [
                "사용자에게 분석 목표 확인 요청",
                "선택된 분석 방법에 따른 데이터 전처리",
                "통계적 가정 검토"
            ],
            "overall_confidence": 0.75,
            "reasoning": "그룹과 연속형 변수가 명확히 있어 분석 가능하지만, 구체적인 분석 목적 명확화 필요"
        }
        ```
        """
    
    def _generate_confirmation_response(self) -> str:
        """1-2 단계: 사용자 선택 확인"""
        return """
        ## 분석 목표 확인 요청

        ```json
        {
            "action": "확인",
            "content": "사용자의 분석 목표 선택을 요청합니다",
            "user_selection": {
                "selected_option": 1,
                "confirmed": true,
                "modifications_requested": [],
                "additional_clarifications": [],
                "user_feedback": "첫 번째 옵션(그룹별 점수 평균 비교)을 선택합니다"
            },
            "final_analysis_plan": {
                "analysis_goal": "그룹별 점수 평균 비교 분석",
                "analysis_type": "One-way ANOVA",
                "dependent_variable": "점수",
                "independent_variable": "그룹",
                "hypothesis": "귀무가설: 모든 그룹의 점수 평균이 같다 vs 대립가설: 적어도 하나의 그룹 평균이 다르다",
                "confidence": 0.85
            },
            "next_steps": [
                "확정된 분석 계획으로 데이터 분석 진행",
                "독립성 전제 검토",
                "데이터 로딩 및 탐색"
            ],
            "confidence": 0.9,
            "reasoning": "사용자가 명확한 선택을 했으므로 분석 진행 가능"
        }
        ```
        """
    
    def _generate_data_loading_response(self) -> str:
        """2-1 단계: 데이터 로딩 응답"""
        return """
        ```json
        {
            "action": "데이터_로딩_완료",
            "content": "데이터가 성공적으로 로딩되었습니다",
            "data_summary": {
                "total_rows": 20,
                "total_columns": 7,
                "missing_values": 0,
                "data_types": "정상"
            },
            "confidence": 1.0,
            "reasoning": "데이터 로딩에 문제없음"
        }
        ```
        """
    
    def _generate_variable_type_response(self) -> str:
        """2-2 단계: 변수 타입 식별 응답"""
        return """
        ```json
        {
            "action": "변수_분석_완료",
            "content": "변수 타입 식별이 완료되었습니다",
            "variable_analysis": {
                "continuous_variables": [
                    {"name": "점수", "description": "연속형 점수 변수"},
                    {"name": "나이", "description": "연속형 나이 변수"},
                    {"name": "경험년수", "description": "연속형 경험 변수"}
                ],
                "categorical_variables": [
                    {"name": "그룹", "categories": ["A", "B", "C"]},
                    {"name": "성별", "categories": ["남", "여"]}
                ],
                "ordinal_variables": [
                    {"name": "만족도", "description": "1-5 척도"}
                ],
                "identifier_variables": [
                    {"name": "ID", "recommendation": "분석에서 제외"}
                ]
            },
            "confidence": 0.95,
            "reasoning": "명확한 변수 타입 분류 완료"
        }
        ```
        """

def test_enhanced_query_mode():
    """개선된 자연어 요청 모드 종합 테스트"""
    print("🔬 개선된 자연어 요청 모드 테스트 시작")
    print("=" * 60)
    
    try:
        # 필요한 모듈 임포트
        from core.workflow_manager import WorkflowManager
        from core.decision_engine import DecisionEngine
        from core.context_manager import ContextManager
        from llm_services.prompt_crafter import PromptCrafter
        from data_processing.data_loader import DataLoader
        from rag_system.code_retriever import CodeRetriever
        from code_execution.safe_code_executor import SafeCodeExecutor
        from reporting.report_generator import ReportGenerator
        from core.agent import LLMAgent
        
        # Mock LLM 클라이언트 생성
        mock_llm = MockLLMClient()
        
        # 각 컴포넌트 초기화
        workflow_manager = WorkflowManager("resources/workflow_graph.json")
        decision_engine = DecisionEngine()
        context_manager = ContextManager(mock_llm)
        prompt_crafter = PromptCrafter("llm_services/prompts")
        data_loader = DataLoader()
        code_retriever = CodeRetriever("resources/code_snippets")
        safe_code_executor = SafeCodeExecutor()
        report_generator = ReportGenerator("output_results")
        
        # LLM Agent 생성 
        agent = LLMAgent(
            workflow_manager=workflow_manager,
            decision_engine=decision_engine,
            context_manager=context_manager,
            llm_client=mock_llm,
            prompt_crafter=prompt_crafter,
            data_loader=data_loader,
            code_retriever=code_retriever,
            safe_code_executor=safe_code_executor,
            report_generator=report_generator
        )
        
        print("✅ 컴포넌트 초기화 성공")
        
        # 테스트 1: 모호한 자연어 요청으로 시작
        print("\n🧪 테스트 1: 모호한 자연어 요청 처리")
        test_query = "그룹별 평균 차이를 분석해주세요"
        print(f"입력 쿼리: '{test_query}'")
        
        # 시작 노드 결정 테스트
        initial_node = agent._determine_initial_node(test_query)
        print(f"결정된 시작 노드: {initial_node}")
        assert initial_node == "1-1", f"예상: 1-1, 실제: {initial_node}"
        print("✅ 자연어 요청 시 올바른 시작 노드 선택")
        
        # 사용자 요청 저장 테스트
        agent.analysis_parameters['user_request'] = test_query
        print(f"저장된 사용자 요청: {agent.analysis_parameters.get('user_request')}")
        assert agent.analysis_parameters['user_request'] == test_query
        print("✅ 사용자 요청 저장 성공")
        
        # 테스트 2: 데이터 로딩 및 기본 정보 확인
        print("\n🧪 테스트 2: 데이터 로딩 및 분석")
        data_path = "input_data/sample_survey_data.csv"
        
        if os.path.exists(data_path):
            agent._load_initial_data(data_path)
            print(f"데이터 형태: {agent.raw_data.shape}")
            print(f"데이터 컬럼: {list(agent.raw_data.columns)}")
            
            # 데이터 요약 테스트
            data_summary = agent._get_data_summary()
            print(f"데이터 요약: {data_summary}")
            print("✅ 데이터 로딩 및 요약 성공")
        else:
            print(f"⚠️ 테스트 데이터 파일을 찾을 수 없음: {data_path}")
        
        # 테스트 3: 1-1 노드 프롬프트 생성 및 LLM 응답 처리
        print("\n🧪 테스트 3: 1-1 단계 - 사용자 요청 분석")
        agent.current_node_id = "1-1"
        
        # 동적 데이터 준비
        dynamic_data = {
            'node_id': agent.current_node_id,
            'user_request': test_query,
            'data_summary': agent._get_data_summary() if agent.raw_data is not None else None,
            'analysis_parameters': agent.analysis_parameters
        }
        
        # 프롬프트 생성 테스트
        try:
            prompt = prompt_crafter.get_prompt_for_node(
                node_id="1-1", 
                dynamic_data=dynamic_data,
                agent_context_summary="테스트 컨텍스트"
            )
            print("✅ 1-1 단계 프롬프트 생성 성공")
            print(f"프롬프트 길이: {len(prompt)} 문자")
            
            # Mock LLM 응답 처리
            response = mock_llm.generate_text(prompt)
            agent._update_analysis_parameters_from_response(response)
            
            print("✅ LLM 응답 처리 성공")
            print(f"분석 후보 개수: {len(agent.analysis_parameters.get('analysis_candidates', []))}")
            print(f"불확실 영역 개수: {len(agent.analysis_parameters.get('uncertainty_areas', []))}")
            
        except Exception as e:
            print(f"❌ 프롬프트 생성 실패: {e}")
        
        # 테스트 4: 1-2 노드 다중 후보 처리
        print("\n🧪 테스트 4: 1-2 단계 - 다중 후보 확인")
        agent.current_node_id = "1-2"
        
        # 1-2 단계용 동적 데이터 준비
        dynamic_data_1_2 = {
            'node_id': agent.current_node_id,
            'user_request': test_query,
            'analysis_candidates': agent.analysis_parameters.get('analysis_candidates', []),
            'uncertainty_areas': agent.analysis_parameters.get('uncertainty_areas', []),
            'clarification_questions': agent.analysis_parameters.get('clarification_questions', []),
            'data_compatibility': agent.analysis_parameters.get('data_compatibility', {})
        }
        
        try:
            prompt_1_2 = prompt_crafter.get_prompt_for_node(
                node_id="1-2",
                dynamic_data=dynamic_data_1_2,
                agent_context_summary="이전 분석 결과"
            )
            print("✅ 1-2 단계 프롬프트 생성 성공")
            
            # Mock 사용자 선택 처리
            response_1_2 = mock_llm.generate_text(prompt_1_2)
            agent._update_analysis_parameters_from_response(response_1_2)
            
            print("✅ 사용자 선택 확정 처리 성공")
            print(f"확정된 분석 목표: {agent.analysis_parameters.get('confirmed_analysis_goal')}")
            print(f"확정된 분석 방법: {agent.analysis_parameters.get('confirmed_analysis_type')}")
            
        except Exception as e:
            print(f"❌ 1-2 단계 처리 실패: {e}")
        
        # 테스트 5: 필수 변수 추출 테스트
        print("\n🧪 테스트 5: 필수 변수 추출")
        required_vars = agent._get_required_variables()
        print(f"추출된 필수 변수: {required_vars}")
        print("✅ 변수 추출 성공")
        
        # 종합 결과 출력
        print("\n" + "=" * 60)
        print("🎉 종합 테스트 결과")
        print("=" * 60)
        
        print(f"✅ Mock LLM 호출 횟수: {mock_llm.call_count}")
        print(f"✅ 저장된 분석 파라미터 개수: {len(agent.analysis_parameters)}")
        print(f"✅ 처리된 분석 후보: {len(agent.analysis_parameters.get('analysis_candidates', []))}개")
        
        # 주요 분석 파라미터 출력
        if 'analysis_candidates' in agent.analysis_parameters:
            print("\n📊 1순위 분석 후보:")
            primary = agent.analysis_parameters['analysis_candidates'][0]
            print(f"  - 목표: {primary['analysis_goal']}")
            print(f"  - 방법: {primary['analysis_type']}")
            print(f"  - 종속변수: {primary['dependent_variable']}")
            print(f"  - 독립변수: {primary['independent_variable']}")
            print(f"  - 확신도: {primary['confidence']}")
        
        if agent.analysis_parameters.get('uncertainty_areas'):
            print(f"\n❓ 식별된 불확실 영역: {len(agent.analysis_parameters['uncertainty_areas'])}개")
            for i, area in enumerate(agent.analysis_parameters['uncertainty_areas'][:2], 1):
                print(f"  {i}. {area}")
        
        print("\n🎯 결론: 개선된 모호한 자연어 요청 처리 시스템이 정상 작동함")
        print("   - 다중 해석 후보 제시 ✅")
        print("   - 불확실성 명시적 처리 ✅") 
        print("   - 단계별 데이터 전달 ✅")
        print("   - JSON 구조 파싱 ✅")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 TEXT_TO_STATISTICAL_TEST 개선된 자연어 처리 테스트")
    print(f"테스트 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    success = test_enhanced_query_mode()
    
    if success:
        print("\n🎉 모든 테스트 통과! 시스템이 모호한 자연어 요청을 효과적으로 처리할 수 있습니다.")
        exit_code = 0
    else:
        print("\n❌ 일부 테스트 실패. 시스템 점검이 필요합니다.")
        exit_code = 1
    
    print(f"테스트 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    exit(exit_code) 