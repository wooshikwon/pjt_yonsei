import typer
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os
from dotenv import load_dotenv

# 통합된 경고 및 로깅 설정
from src.utils.warnings_config import setup_warnings_and_logging
setup_warnings_and_logging()

# .env 파일 로드
load_dotenv()

from src.components.context import Context
from src.components.rag_retriever import RAGRetriever
from src.components.code_executor import CodeExecutor
from src.agent import Agent
from src.utils.logger import get_logger

app = typer.Typer()

@app.command()
def analyze(
    file_name: str = typer.Option(..., "--file", help="Name of the data file in 'input_data/data_files/'"),
    request: str = typer.Option(..., "--request", help="Your natural language request for analysis.")
):
    """
    데이터 파일과 사용자 요청을 기반으로 전체 통계 분석 파이프라인을 실행합니다.
    """
    # 로거 초기화
    logger = get_logger()
    
    # --- 초기화 ---
    logger.log_system_info("시스템 초기화 중...")
    
    # .env 파일 다시 로드 (실시간 변경사항 반영)
    load_dotenv(override=True)
    
    # 환경변수 읽기
    use_rag = os.getenv("USE_RAG", "True").lower() == "true"
    rebuild_vector_store = os.getenv("REBUILD_VECTOR_STORE", "False").lower() == "true"
    


    # 경로 설정
    base_path = Path.cwd()
    input_file_path = base_path / "input_data/data_files" / file_name
    knowledge_base_path = str(base_path / "resources/knowledge_base")
    vector_store_path = str(base_path / "resources/rag_index")
    report_path = base_path / "output_data/reports"
    report_path.mkdir(parents=True, exist_ok=True)

    # 컴포넌트 인스턴스화
    context = Context()
    agent = Agent()
    executor = CodeExecutor()

    context.set_user_input(file_path=str(input_file_path), request=request)
    logger.log_detailed(f"User Request: {request}")
    logger.log_detailed(f"Data File: {input_file_path}")

    # --- Step 0: 벡터 스토어 관리 (독립적 실행) ---
    if rebuild_vector_store:
        logger.log_step_start(0, "벡터 스토어 재구축")
        try:
            # 임시 RAGRetriever 인스턴스로 인덱스 재구축만 수행
            temp_retriever = RAGRetriever(
                knowledge_base_path=knowledge_base_path,
                vector_store_path=vector_store_path,
                rebuild=True
            )
            temp_retriever.load()  # rebuild=True이므로 기존 삭제 후 재구축
            logger.log_step_success(0, "벡터 스토어 재구축 완료")
        except Exception as e:
            logger.log_step_failure(0, f"벡터 스토어 재구축 실패: {str(e)}")
            logger.log_detailed(f"Vector store rebuild error: {e}", "ERROR")

    # --- Step 1: RAG로 컨텍스트 강화 (조건부 실행) ---
    if use_rag:
        logger.log_step_start(1, "RAG 컨텍스트 강화")
        
        # 인덱스 존재 여부 먼저 확인
        vector_store_path_obj = Path(vector_store_path)
        if not (vector_store_path_obj / "docstore.json").exists():
            logger.log_step_failure(1, "벡터 인덱스를 찾을 수 없습니다")
            print("\n⚠️  RAG 인덱스가 존재하지 않습니다!")
            print("📋 해결 방법: .env 파일에서 REBUILD_VECTOR_STORE=True로 설정하고 다시 실행해주세요.")
            print(f"📁 지식 베이스 경로: {knowledge_base_path}")
            print(f"📁 벡터 스토어 경로: {vector_store_path}")
            print("\n💡 지식 베이스에 .md 파일이 있는지 확인하고, 벡터 스토어를 빌드해주세요.\n")
            # RAG 없이 계속 진행
            logger.log_step_success(1, "RAG 인덱스 없음 - RAG 없이 분석 진행")
        else:
            retriever = RAGRetriever(
                knowledge_base_path=knowledge_base_path, 
                vector_store_path=vector_store_path,
                rebuild=False  # 재구축은 Step 0에서 이미 처리됨
            )
            try:
                retriever.load()
                rag_context = retriever.retrieve_context(request)
                context.add_rag_result(rag_context)
                logger.log_rag_context(rag_context)
                logger.log_step_success(1, "지식 베이스에서 관련 정보 검색 완료")
            except Exception as e:
                logger.log_step_failure(1, str(e))
                logger.log_detailed(f"RAG Error: {e}", "ERROR")
    else:
        logger.log_step_start(1, "RAG 건너뛰기")
        logger.log_step_success(1, "환경 설정에 따라 RAG 단계 생략")

    # --- Step 2: 데이터 로딩 및 초기 탐색 ---
    logger.log_step_start(2, "데이터 로딩 및 탐색")
    try:
        if input_file_path.suffix == '.csv':
            df = pd.read_csv(input_file_path)
        elif input_file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(input_file_path)
        elif input_file_path.suffix == '.parquet':
            df = pd.read_parquet(input_file_path)
        else:
            raise ValueError(f"Unsupported file type: {input_file_path.suffix}")
        
        logger.log_detailed(f"Data shape: {df.shape}")
        logger.log_detailed(f"Columns: {list(df.columns)}")
        logger.log_step_success(2, f"데이터 로딩 완료 ({df.shape[0]}행, {df.shape[1]}열)")
    except FileNotFoundError:
        error_msg = f"파일을 찾을 수 없습니다: {input_file_path}"
        logger.log_step_failure(2, error_msg)
        logger.log_detailed(error_msg, "ERROR")
        sys.exit(1)
    except Exception as e:
        error_msg = f"데이터 로딩 중 오류 발생: {e}"
        logger.log_step_failure(2, error_msg)
        logger.log_detailed(error_msg, "ERROR")
        sys.exit(1)

    # --- Step 3: 통계 분석 계획 수립 ---
    logger.log_step_start(3, "분석 계획 수립")
    try:
        schema = {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
        null_values = df.isnull().sum().to_dict()
        sample_data = df.head().to_string()
        context.set_data_info(schema=schema, null_values=null_values, sample_data=sample_data)

        plan = agent.generate_analysis_plan(context)
        context.set_analysis_plan(plan)
        
        logger.log_detailed("Generated Analysis Plan:")
        for i, step in enumerate(plan, 1):
            logger.log_detailed(f"{i}. {step}")
        
        logger.log_step_success(3, f"분석 계획 수립 완료 ({len(plan)}단계)")
    except Exception as e:
        logger.log_step_failure(3, str(e))
        logger.log_detailed(f"Analysis planning error: {e}", "ERROR")
        sys.exit(1)
    
    # --- Step 4: 계획 기반 실행 및 자가 수정 루프 ---
    logger.log_step_start(4, "분석 계획 실행")
    try:
        failed_steps = 0
        for i, step in enumerate(context.analysis_plan):
            step_num = i + 1
            logger.log_detailed(f"\nExecuting Step {step_num}: {step}")
            
            code = agent.generate_code_for_step(context, step)
            logger.log_generated_code(step_num, code)

            result, success = executor.run(code, global_vars={'df': df})
            logger.log_execution_result(step_num, result, success)
            
            if success:
                context.add_to_history({'role': 'assistant', 'code': code})
                context.add_to_history({'role': 'system', 'result': result})
            else:
                logger.log_detailed(f"Step {step_num} failed, attempting self-correction...")
                context.add_to_history({'role': 'assistant', 'code': code})
                context.add_to_history({'role': 'system', 'error': result})
                
                corrected_code = agent.self_correct_code(context, step, code, result)
                logger.log_detailed(f"Corrected code generated for step {step_num}")
                
                result, success = executor.run(corrected_code, global_vars={'df': df})
                logger.log_execution_result(step_num, f"CORRECTED: {result}", success)
                
                if success:
                    context.add_to_history({'role': 'assistant', 'code': corrected_code})
                    context.add_to_history({'role': 'system', 'result': result})
                else:
                    failed_steps += 1
                    logger.log_detailed(f"FATAL: Self-correction failed for step {step_num}")
        
        if failed_steps == 0:
            logger.log_step_success(4, f"모든 분석 단계 성공적으로 완료")
        else:
            logger.log_step_success(4, f"분석 완료 (일부 단계 실패: {failed_steps}개)")
            
    except Exception as e:
        logger.log_step_failure(4, str(e))
        logger.log_detailed(f"Analysis execution error: {e}", "ERROR")
        sys.exit(1)
    
    # --- Step 5: 최종 보고서 생성 ---
    logger.log_step_start(5, "최종 보고서 생성")
    try:
        final_report = agent.generate_final_report(context)
        context.set_final_report(final_report)
        logger.log_step_success(5, "보고서 생성 완료")
    except Exception as e:
        logger.log_step_failure(5, str(e))
        logger.log_detailed(f"Report generation error: {e}", "ERROR")
        final_report = "보고서 생성 중 오류가 발생했습니다."
    
    # --- 결과 출력 및 저장 ---
    logger.print_final_report(final_report)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_file_name = f"report-{timestamp}.md"
    report_file_path = report_path / report_file_name
    
    try:
        with open(report_file_path, 'w', encoding='utf-8') as f:
            f.write(final_report)
        logger.log_report_saved(str(report_file_path))
    except Exception as e:
        logger.log_detailed(f"Failed to save report: {e}", "ERROR")

if __name__ == "__main__":
    app() 