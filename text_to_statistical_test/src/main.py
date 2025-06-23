import typer
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os
from dotenv import load_dotenv
import traceback

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
from src.utils.data_profiler import profile_dataframe

app = typer.Typer()

def _update_state_after_prep(df: pd.DataFrame, context: Context, logger, step_info: str) -> pd.DataFrame:
    """[PREP] 단계 성공 후 df와 data_summary를 업데이트하는 헬퍼 함수."""
    logger.log_detailed(f"DataFrame state updated after step: {step_info}")
    new_summary = profile_dataframe(df)
    context.set_data_summary(new_summary)
    logger.log_data_summary(new_summary)
    return df

@app.command()
def analyze(
    file_name: str = typer.Option(..., "--file", help="Name of the data file in 'input/data_files/'"),
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
    input_file_path = base_path / "input" / "data_files" / file_name
    knowledge_base_path = base_path / "resources" / "knowledge_base"

    # vector_store_path 경로 생성 및 설정
    vector_store_path = base_path / "resources" / "rag_index"
    vector_store_path.mkdir(parents=True, exist_ok=True)

    # output 경로 생성 및 설정
    output_path = base_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "reports"
    report_path.mkdir(parents=True, exist_ok=True)
    final_data_path = output_path / "data_files"
    final_data_path.mkdir(parents=True, exist_ok=True)

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
                
                # RAG 쿼리에 파일 이름과 사용자 요청을 함께 사용
                rag_query = f"Data file: {file_name}\nUser request: {request}"
                logger.log_detailed(f"RAG Query: {rag_query}")

                rag_context = retriever.retrieve_context(rag_query)
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
        # 데이터 프로파일링 수행
        data_summary = profile_dataframe(df)
        context.set_data_summary(data_summary)

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
            logger.log_step_separator()
            logger.log_detailed(f"Executing Step {step_num}: {step}")
            
            rationale, code = agent.generate_code_for_step(context, step)
            logger.log_generated_code(step_num, code, rationale)

            result, status, df_after_execution = executor.run(code, global_vars={'df': df.copy()}) # df.copy()로 안전성 확보
            logger.log_execution_result(step_num, result, status != 'ERROR')
            
            context.add_rationale_history(rationale) # Rationale도 기록
            context.add_code_history(code)
            context.add_output_history(result)

            if status == 'SKIPPED':
                context.add_step_to_summary(step, "Skipped")
                continue
            
            if status == 'SUCCESS':
                if step.strip().startswith("[PREP]") and df_after_execution is not None:
                    df = _update_state_after_prep(df_after_execution, context, logger, f"{step_num}")
                context.add_step_to_summary(step, "Success")
            
            elif status == 'ERROR':
                logger.log_detailed(f"Step {step_num} failed, attempting self-correction...")
                
                try:
                    corrected_rationale, corrected_code = agent.self_correct_code(context, step, rationale, code, result)
                    logger.log_detailed(f"Corrected code generated for step {step_num}")
                    logger.log_generated_code(step_num, corrected_code, corrected_rationale, is_corrected=True)
                    
                    result, status, df_after_execution = executor.run(corrected_code, global_vars={'df': df.copy()})
                    logger.log_execution_result(step_num, f"CORRECTED: {result}", status != 'ERROR')
                    
                    context.add_rationale_history(corrected_rationale)
                    context.add_code_history(corrected_code)
                    context.add_output_history(result)

                    if status == 'SUCCESS':
                        if step.strip().startswith("[PREP]") and df_after_execution is not None:
                            df = _update_state_after_prep(df_after_execution, context, logger, f"{step_num} (Corrected)")
                        context.add_step_to_summary(step, "Success (Corrected)")
                    else:
                        raise RuntimeError(f"Self-correction failed. Status: {status}")

                except Exception as e:
                    failed_steps += 1
                    context.add_step_to_summary(step, "Failure (Correction Failed)")
                    logger.log_detailed(f"FATAL: Self-correction failed for step {step_num}. Error: {e}")
        
        if failed_steps == 0:
            logger.log_step_success(4, f"모든 분석 단계 성공적으로 완료")
        else:
            logger.log_step_success(4, f"분석 완료 (일부 단계 실패: {failed_steps}개)")
            
    except Exception as e:
        # --- 향상된 에러 로깅 ---
        tb_str = traceback.format_exc()
        logger.log_step_failure(4, f"An unexpected error occurred: {e}")
        logger.log_detailed(f"Analysis execution error: {e}\n{tb_str}", "ERROR")
        logger.log_detailed(f"Context at the time of error: {context.get_full_context()}", "ERROR")
        sys.exit(1)
    
    # --- Step 5: 최종 보고서 생성 ---
    logger.log_step_start(5, "최종 보고서 생성")
    try:
        final_report = agent.generate_final_report(context, final_data_shape=df.shape)
        context.set_final_report(final_report)
        logger.log_step_success(5, "보고서 생성 완료")
    except Exception as e:
        logger.log_step_failure(5, str(e))
        logger.log_detailed(f"Report generation error: {e}", "ERROR")
        final_report = "보고서 생성 중 오류가 발생했습니다."
    
    # --- 결과 출력 및 저장 ---
    cleaned_report = final_report.strip()
    if cleaned_report.startswith("```markdown"):
        cleaned_report = cleaned_report[11:]
    if cleaned_report.endswith("```"):
        cleaned_report = cleaned_report[:-3]
    
    logger.print_final_report(cleaned_report.strip())
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 보고서 파일 저장
    report_file_name = f"report-{timestamp}.md"
    report_file_path = report_path / report_file_name
    
    try:
        with open(report_file_path, 'w', encoding='utf-8') as f:
            f.write(final_report)
        logger.log_report_saved(str(report_file_path))
    except Exception as e:
        logger.log_detailed(f"Failed to save report: {e}", "ERROR")

    # 최종 데이터프레임 저장
    final_data_filename = f"final_data-{timestamp}.csv"
    final_data_filepath = final_data_path / final_data_filename
    
    try:
        df.to_csv(final_data_filepath, index=False, encoding='utf-8-sig')
        logger.log_final_data_saved(str(final_data_filepath))
    except Exception as e:
        logger.log_detailed(f"Failed to save final data: {e}", "ERROR")

if __name__ == "__main__":
    app() 