"""
User Request Pipeline

2단계: 사용자의 자연어 요청 및 목표 정의 (Multi-turn)
사용자가 자연어로 분석 목표와 궁금증을 전달하고,
시스템이 대화형으로 추가 질문을 통해 분석의 범위와 구체적인 목표를 명확히 합니다.
"""

import logging
from typing import Dict, Any, Optional, List
import re

from .base_pipeline_step import BasePipelineStep, PipelineStepRegistry
from utils.ui_helpers import print_analysis_guide


class UserRequestStep(BasePipelineStep):
    """2단계: 사용자의 자연어 요청 및 목표 정의 (Multi-turn)"""
    
    def __init__(self):
        """UserRequestStep 초기화"""
        super().__init__("사용자의 자연어 요청 및 목표 정의", 2)
        self.min_request_length = 5
        self.max_request_length = 1000
        self.conversation_history = []
        self.clarification_questions = []
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        입력 데이터 유효성 검증
        
        Args:
            input_data: 입력 데이터 (1단계에서 전달받은 데이터)
            
        Returns:
            bool: 유효성 검증 결과
        """
        required_fields = ['selected_file', 'file_info']
        return all(field in input_data for field in required_fields)
    
    def get_expected_output_schema(self) -> Dict[str, Any]:
        """
        예상 출력 스키마 반환
        
        Returns:
            Dict[str, Any]: 출력 데이터 스키마
        """
        return {
            'user_request': str,
            'refined_objectives': list,
            'analysis_scope': dict,
            'conversation_history': list,
            'request_metadata': {
                'analysis_type': str,
                'target_variables': list,
                'group_variables': list,
                'specific_tests': list,
                'complexity_level': str
            },
            'clarification_completed': bool
        }
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        사용자 자연어 요청 파이프라인 실행
        
        Args:
            input_data: 파이프라인 실행 컨텍스트
                - selected_file: 선택된 파일 경로
                - file_info: 파일 정보
                - user_request (optional): 직접 제공된 사용자 요청
                - interactive (optional): 대화형 모드 여부 (기본값: True)
                - max_clarifications (optional): 최대 명확화 질문 수 (기본값: 3)
            
        Returns:
            Dict: 실행 결과
        """
        self.logger.info("2단계: 사용자의 자연어 요청 및 목표 정의 시작")
        
        try:
            # 대화 히스토리 초기화
            self.conversation_history = []
            self.clarification_questions = []
            
            # 직접 요청이 제공된 경우
            if 'user_request' in input_data and input_data['user_request']:
                initial_request = input_data['user_request']
                self.conversation_history.append({
                    'type': 'user_input',
                    'content': initial_request,
                    'timestamp': self._get_timestamp()
                })
            else:
                # 대화형 요청 입력
                interactive = input_data.get('interactive', True)
                if interactive:
                    initial_request = self._get_initial_request_interactive(input_data)
                    if not initial_request:
                        return {
                            'error': True,
                            'error_message': '자연어 요청 입력이 취소되었습니다.',
                            'cancelled': True
                        }
                else:
                    return {
                        'error': True,
                        'error_message': '사용자 요청이 제공되지 않았습니다.',
                        'error_type': 'missing_request'
                    }
            
            # 특수 명령어 처리
            special_action = self._handle_special_commands(initial_request)
            if special_action:
                return special_action
            
            # 초기 요청 검증 및 정제
            validation_result = self._validate_and_process_request(initial_request)
            if validation_result.get('error'):
                return validation_result
            
            # Multi-turn 대화를 통한 목표 명확화
            max_clarifications = input_data.get('max_clarifications', 3)
            clarification_result = self._conduct_clarification_dialogue(
                initial_request, input_data, max_clarifications
            )
            
            # 최종 분석 목표 및 범위 정리
            final_objectives = self._finalize_analysis_objectives(
                initial_request, clarification_result
            )
            
            self.logger.info(f"사용자 요청 및 목표 정의 완료")
            
            return {
                'user_request': initial_request,
                'refined_objectives': final_objectives['objectives'],
                'analysis_scope': final_objectives['scope'],
                'conversation_history': self.conversation_history,
                'request_metadata': final_objectives['metadata'],
                'clarification_completed': True,
                'success_message': f"📝 분석 목표가 명확히 정의되었습니다."
            }
                
        except Exception as e:
            self.logger.error(f"사용자 요청 파이프라인 오류: {e}")
            return {
                'error': True,
                'error_message': str(e)
            }
    
    def _get_initial_request_interactive(self, input_data: Dict[str, Any]) -> Optional[str]:
        """대화형 초기 요청 입력"""
        try:
            # 사용자에게 가이드 표시
            self._display_request_guide(input_data)
            
            user_request = input("\n📝 분석 요청: ").strip()
            
            if user_request:
                self.conversation_history.append({
                    'type': 'user_input',
                    'content': user_request,
                    'timestamp': self._get_timestamp()
                })
            
            return user_request if user_request else None
            
        except KeyboardInterrupt:
            self.logger.info("사용자가 요청 입력을 취소했습니다.")
            return None
        except Exception as e:
            self.logger.error(f"대화형 요청 입력 오류: {e}")
            return None
    
    def _display_request_guide(self, input_data: Dict[str, Any]) -> None:
        """사용자에게 요청 가이드 표시"""
        print_analysis_guide()
        
        selected_file = input_data.get('selected_file', '')
        file_name = selected_file.split('/')[-1] if selected_file else 'Unknown'
        
        # 파일 정보 표시
        file_info = input_data.get('file_info', {})
        if file_info:
            print(f"\n📂 선택된 데이터: {file_name}")
            print(f"   • 행 수: {file_info.get('row_count', 'N/A'):,}")
            print(f"   • 열 수: {file_info.get('column_count', 'N/A')}")
            print(f"   • 주요 변수: {', '.join(file_info.get('columns', [])[:5])}")
            if len(file_info.get('columns', [])) > 5:
                print(f"     ... 외 {len(file_info.get('columns', [])) - 5}개")
        
        print("\n💬 분석하고 싶은 내용을 자연어로 말씀해주세요:")
        print("   예시:")
        print("   • '그룹별 평균 차이를 분석해주세요'")
        print("   • '변수들 간의 상관관계를 알고 싶어요'")
        print("   • '회귀분석을 통해 예측모델을 만들어주세요'")
        print("   • '범주형 변수들의 연관성을 확인해주세요'")
    
    def _conduct_clarification_dialogue(self, initial_request: str, input_data: Dict[str, Any], max_clarifications: int) -> Dict[str, Any]:
        """Multi-turn 대화를 통한 목표 명확화"""
        clarification_count = 0
        current_understanding = self._analyze_initial_request(initial_request, input_data)
        
        while clarification_count < max_clarifications:
            # 명확화가 필요한 부분 식별
            questions = self._generate_clarification_questions(current_understanding, input_data)
            
            if not questions:
                # 더 이상 명확화할 것이 없음
                break
            
            # 가장 중요한 질문 선택
            primary_question = questions[0]
            
            # 대화형 모드에서만 질문
            if input_data.get('interactive', True):
                print(f"\n🤔 {primary_question['question']}")
                if primary_question.get('options'):
                    for i, option in enumerate(primary_question['options'], 1):
                        print(f"   {i}. {option}")
                
                try:
                    user_response = input("👤 답변: ").strip()
                    if not user_response:
                        break
                    
                    self.conversation_history.append({
                        'type': 'system_question',
                        'content': primary_question['question'],
                        'timestamp': self._get_timestamp()
                    })
                    self.conversation_history.append({
                        'type': 'user_response',
                        'content': user_response,
                        'timestamp': self._get_timestamp()
                    })
                    
                    # 답변을 바탕으로 이해도 업데이트
                    current_understanding = self._update_understanding(
                        current_understanding, primary_question, user_response
                    )
                    
                except KeyboardInterrupt:
                    print("\n명확화 대화를 종료합니다.")
                    break
            else:
                # 비대화형 모드에서는 기본값 사용
                break
            
            clarification_count += 1
        
        return {
            'final_understanding': current_understanding,
            'clarification_count': clarification_count,
            'questions_asked': self.clarification_questions
        }
    
    def _analyze_initial_request(self, request: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """초기 요청 분석"""
        analysis = {
            'analysis_type': self._detect_analysis_type(request),
            'target_variables': self._extract_mentioned_variables(request, input_data),
            'group_variables': self._extract_group_variables(request, input_data),
            'specific_tests': self._extract_specific_tests(request),
            'uncertainty_areas': [],
            'confidence_level': 'medium'
        }
        
        # 불확실한 영역 식별
        if not analysis['target_variables']:
            analysis['uncertainty_areas'].append('target_variables')
        if analysis['analysis_type'] == 'group_comparison' and not analysis['group_variables']:
            analysis['uncertainty_areas'].append('group_variables')
        if analysis['analysis_type'] == 'unknown':
            analysis['uncertainty_areas'].append('analysis_type')
        
        return analysis
    
    def _generate_clarification_questions(self, understanding: Dict[str, Any], input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """명확화 질문 생성"""
        questions = []
        file_info = input_data.get('file_info', {})
        available_columns = file_info.get('columns', [])
        
        # 분석 유형이 불명확한 경우
        if 'analysis_type' in understanding['uncertainty_areas']:
            questions.append({
                'type': 'analysis_type',
                'question': '어떤 종류의 분석을 원하시나요?',
                'options': [
                    '그룹 간 비교 (평균, 비율 차이 등)',
                    '변수 간 관계 분석 (상관관계, 회귀분석)',
                    '범주형 데이터 연관성 분석',
                    '기술 통계 (요약, 분포 등)'
                ]
            })
        
        # 타겟 변수가 불명확한 경우
        if 'target_variables' in understanding['uncertainty_areas'] and available_columns:
            questions.append({
                'type': 'target_variables',
                'question': '주요 분석 대상이 되는 변수는 무엇인가요?',
                'options': available_columns[:10]  # 최대 10개까지만 표시
            })
        
        # 그룹 변수가 불명확한 경우
        if 'group_variables' in understanding['uncertainty_areas'] and available_columns:
            questions.append({
                'type': 'group_variables',
                'question': '그룹을 나누는 기준이 되는 변수는 무엇인가요?',
                'options': available_columns[:10]
            })
        
        return questions
    
    def _update_understanding(self, understanding: Dict[str, Any], question: Dict[str, Any], response: str) -> Dict[str, Any]:
        """사용자 답변을 바탕으로 이해도 업데이트"""
        question_type = question['type']
        
        if question_type == 'analysis_type':
            if '1' in response or '그룹' in response or '비교' in response:
                understanding['analysis_type'] = 'group_comparison'
            elif '2' in response or '관계' in response or '회귀' in response:
                understanding['analysis_type'] = 'relationship'
            elif '3' in response or '범주' in response or '연관' in response:
                understanding['analysis_type'] = 'categorical'
            elif '4' in response or '기술' in response or '요약' in response:
                understanding['analysis_type'] = 'descriptive'
            
            if 'analysis_type' in understanding['uncertainty_areas']:
                understanding['uncertainty_areas'].remove('analysis_type')
        
        elif question_type == 'target_variables':
            # 변수명 추출 로직
            mentioned_vars = [var for var in question.get('options', []) if var.lower() in response.lower()]
            if mentioned_vars:
                understanding['target_variables'] = mentioned_vars
                if 'target_variables' in understanding['uncertainty_areas']:
                    understanding['uncertainty_areas'].remove('target_variables')
        
        elif question_type == 'group_variables':
            mentioned_vars = [var for var in question.get('options', []) if var.lower() in response.lower()]
            if mentioned_vars:
                understanding['group_variables'] = mentioned_vars
                if 'group_variables' in understanding['uncertainty_areas']:
                    understanding['uncertainty_areas'].remove('group_variables')
        
        return understanding
    
    def _finalize_analysis_objectives(self, initial_request: str, clarification_result: Dict[str, Any]) -> Dict[str, Any]:
        """최종 분석 목표 및 범위 정리"""
        understanding = clarification_result['final_understanding']
        
        objectives = [
            f"분석 유형: {self._get_analysis_type_description(understanding['analysis_type'])}"
        ]
        
        if understanding['target_variables']:
            objectives.append(f"주요 분석 변수: {', '.join(understanding['target_variables'])}")
        
        if understanding['group_variables']:
            objectives.append(f"그룹 변수: {', '.join(understanding['group_variables'])}")
        
        if understanding['specific_tests']:
            objectives.append(f"특정 통계 기법: {', '.join(understanding['specific_tests'])}")
        
        scope = {
            'analysis_complexity': self._determine_complexity_level(understanding),
            'estimated_steps': self._estimate_analysis_steps(understanding),
            'data_requirements': self._identify_data_requirements(understanding)
        }
        
        metadata = {
            'analysis_type': understanding['analysis_type'],
            'target_variables': understanding['target_variables'],
            'group_variables': understanding['group_variables'],
            'specific_tests': understanding['specific_tests'],
            'complexity_level': scope['analysis_complexity'],
            'clarification_count': clarification_result['clarification_count']
        }
        
        return {
            'objectives': objectives,
            'scope': scope,
            'metadata': metadata
        }
    
    def _handle_special_commands(self, user_request: str) -> Optional[Dict[str, Any]]:
        """특수 명령어 처리"""
        request_lower = user_request.lower().strip()
        
        # 종료 명령어
        if request_lower in ['quit', 'exit', '종료', 'q']:
            return {
                'action': 'quit',
                'success_message': '분석을 종료합니다.'
            }
        
        # 새 파일 선택
        elif request_lower in ['new', '새파일', 'new file']:
            return {
                'action': 'new_file',
                'success_message': '새로운 데이터 파일을 선택합니다.'
            }
        
        # 상태 확인
        elif request_lower in ['status', '상태', 'help', '도움말']:
            return {
                'action': 'show_status',
                'success_message': '현재 상태를 확인합니다.'
            }
        
        return None
    
    def _validate_and_process_request(self, user_request: str) -> Dict[str, Any]:
        """자연어 요청 검증 및 정제"""
        # 기본 검증
        if not user_request or not user_request.strip():
            return {
                'error': True,
                'error_message': '분석 요청을 입력해주세요.',
                'error_type': 'empty_request'
            }
        
        # 길이 검증
        if len(user_request) < self.min_request_length:
            return {
                'error': True,
                'error_message': f'요청이 너무 짧습니다. 최소 {self.min_request_length}자 이상 입력해주세요.',
                'error_type': 'too_short'
            }
        
        if len(user_request) > self.max_request_length:
            return {
                'error': True,
                'error_message': f'요청이 너무 깁니다. 최대 {self.max_request_length}자 이하로 입력해주세요.',
                'error_type': 'too_long'
            }
        
        # 무의미한 요청 검증
        if self._is_meaningless_request(user_request):
            return {
                'error': True,
                'error_message': '구체적인 분석 요청을 입력해주세요.',
                'error_type': 'meaningless_request'
            }
        
        return {'processed_request': user_request.strip()}
    
    def _detect_analysis_type(self, request: str) -> str:
        """요청에서 분석 유형 감지"""
        request_lower = request.lower()
        
        # 그룹 비교 키워드
        group_keywords = ['그룹', '비교', '차이', '평균', 't-test', 'anova', '집단']
        if any(keyword in request_lower for keyword in group_keywords):
            return 'group_comparison'
        
        # 관계 분석 키워드
        relationship_keywords = ['상관', '관계', '회귀', '예측', 'correlation', 'regression']
        if any(keyword in request_lower for keyword in relationship_keywords):
            return 'relationship'
        
        # 범주형 분석 키워드
        categorical_keywords = ['범주', '카이제곱', '연관성', 'chi-square', '독립성']
        if any(keyword in request_lower for keyword in categorical_keywords):
            return 'categorical'
        
        # 기술 통계 키워드
        descriptive_keywords = ['요약', '분포', '기술통계', '평균', '표준편차']
        if any(keyword in request_lower for keyword in descriptive_keywords):
            return 'descriptive'
        
        return 'unknown'
    
    def _extract_mentioned_variables(self, request: str, input_data: Dict[str, Any]) -> List[str]:
        """요청에서 언급된 변수명 추출"""
        file_info = input_data.get('file_info', {})
        available_columns = file_info.get('columns', [])
        
        mentioned_vars = []
        for col in available_columns:
            if col.lower() in request.lower():
                mentioned_vars.append(col)
        
        return mentioned_vars
    
    def _extract_group_variables(self, request: str, input_data: Dict[str, Any]) -> List[str]:
        """그룹 변수 추출"""
        # 그룹 관련 키워드 근처의 변수명 찾기
        group_keywords = ['그룹별', '집단별', '카테고리별', '유형별']
        # 구현 단순화 - 실제로는 더 정교한 NLP 처리 필요
        return []
    
    def _extract_specific_tests(self, request: str) -> List[str]:
        """특정 통계 기법 추출"""
        test_keywords = {
            't-test': ['t-test', 'ttest', 't검정'],
            'anova': ['anova', '분산분석', '일원분산분석'],
            'regression': ['regression', '회귀분석', '선형회귀'],
            'correlation': ['correlation', '상관분석', '피어슨'],
            'chi-square': ['chi-square', '카이제곱', 'chi2']
        }
        
        mentioned_tests = []
        request_lower = request.lower()
        
        for test_name, keywords in test_keywords.items():
            if any(keyword in request_lower for keyword in keywords):
                mentioned_tests.append(test_name)
        
        return mentioned_tests
    
    def _is_meaningless_request(self, request: str) -> bool:
        """무의미한 요청인지 확인"""
        meaningless_patterns = [
            r'^[a-zA-Z\s]*$',  # 영문자와 공백만
            r'^[0-9\s]*$',     # 숫자와 공백만
            r'^[!@#$%^&*()_+\-=\[\]{};:\'",.<>/?`~\s]*$'  # 특수문자와 공백만
        ]
        
        request_clean = request.strip()
        
        for pattern in meaningless_patterns:
            if re.match(pattern, request_clean):
                return True
        
        # 너무 반복적인 문자
        if len(set(request_clean.replace(' ', ''))) < 3:
            return True
        
        return False
    
    def _get_analysis_type_description(self, analysis_type: str) -> str:
        """분석 유형 설명 반환"""
        descriptions = {
            'group_comparison': '그룹 간 비교 분석',
            'relationship': '변수 간 관계 분석',
            'categorical': '범주형 데이터 연관성 분석',
            'descriptive': '기술 통계 분석',
            'unknown': '일반적인 데이터 분석'
        }
        return descriptions.get(analysis_type, '사용자 정의 분석')
    
    def _determine_complexity_level(self, understanding: Dict[str, Any]) -> str:
        """분석 복잡도 수준 결정"""
        complexity_score = 0
        
        if understanding['target_variables']:
            complexity_score += len(understanding['target_variables'])
        if understanding['group_variables']:
            complexity_score += len(understanding['group_variables']) * 2
        if understanding['specific_tests']:
            complexity_score += len(understanding['specific_tests'])
        
        if complexity_score <= 2:
            return 'simple'
        elif complexity_score <= 5:
            return 'medium'
        else:
            return 'complex'
    
    def _estimate_analysis_steps(self, understanding: Dict[str, Any]) -> int:
        """예상 분석 단계 수 추정"""
        base_steps = 3  # 기본적인 전처리, 분석, 보고서
        
        if understanding['target_variables']:
            base_steps += len(understanding['target_variables'])
        if understanding['specific_tests']:
            base_steps += len(understanding['specific_tests'])
        
        return min(base_steps, 10)  # 최대 10단계
    
    def _identify_data_requirements(self, understanding: Dict[str, Any]) -> List[str]:
        """데이터 요구사항 식별"""
        requirements = []
        
        if understanding['analysis_type'] == 'group_comparison':
            requirements.append('그룹을 나눌 수 있는 범주형 변수')
            requirements.append('비교할 연속형 변수')
        elif understanding['analysis_type'] == 'relationship':
            requirements.append('연속형 변수들 간의 관계 분석 가능')
        elif understanding['analysis_type'] == 'categorical':
            requirements.append('범주형 변수들')
        
        return requirements
    
    def _get_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_step_info(self) -> Dict[str, Any]:
        """단계 정보 반환 (부모 클래스 메서드 확장)"""
        base_info = super().get_step_info()
        base_info.update({
            'description': '사용자의 자연어 요청 및 목표 정의 (Multi-turn)',
            'input_requirements': ['selected_file', 'file_info'],
            'output_provides': ['user_request', 'refined_objectives', 'analysis_scope', 'conversation_history', 'request_metadata'],
            'supports_multiturn': True,
            'max_clarifications': 3
        })
        return base_info


# 단계 등록
PipelineStepRegistry.register_step(2, UserRequestStep) 