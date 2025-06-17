import pytest
import sys
import os

# 테스트 대상 모듈을 import하기 위해 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import Agent
from src.components.context import Context

def test_generate_analysis_plan_formatting(mocker):
    """
    generate_analysis_plan 메서드가 컨텍스트를 올바르게 포매팅하여
    _call_api를 호출하고, 그 응답을 성공적으로 파싱하는지 테스트합니다.
    """
    # 1. 테스트 설정
    agent = Agent()
    context = Context()
    context.set_user_input(file_path="dummy.csv", request="A/B 테스트 분석")
    context.add_rag_result("RAG 결과: A는 대조군, B는 실험군입니다.")
    context.set_data_info(schema={'group': 'object', 'value': 'int64'}, null_values={}, sample_data="")

    # 2. Mocking
    mock_response = "1. Step one\n2. Step two"
    mock_call_api = mocker.patch.object(agent, '_call_api', return_value=mock_response)

    # 3. 테스트 로직 실행
    plan = agent.generate_analysis_plan(context)

    # 4. 검증
    # _call_api가 한 번 호출되었는지 확인
    mock_call_api.assert_called_once()
    
    # 반환된 계획이 올바르게 파싱되었는지 확인
    assert plan == ['1. Step one', '2. Step two']

    # _call_api에 전달된 프롬프트가 올바른 내용을 포함하는지 확인
    call_args = mock_call_api.call_args
    prompt_messages = call_args.args[0]
    formatted_prompt = prompt_messages[0]['content']

    assert "A/B 테스트 분석" in formatted_prompt
    assert "RAG 결과: A는 대조군, B는 실험군입니다." in formatted_prompt
    assert "'group': 'object'" in formatted_prompt 