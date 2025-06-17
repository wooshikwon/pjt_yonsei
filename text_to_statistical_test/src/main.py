import typer
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os
from dotenv import load_dotenv

from src.components.context import Context
from src.components.rag_retriever import RAGRetriever
from src.components.code_executor import CodeExecutor
from src.agent import Agent

# .env 파일 로드
load_dotenv()

app = typer.Typer()

@app.command()
def analyze(
    file_name: str = typer.Option(..., "--file", help="Name of the data file in 'input_data/data_files/'"),
    request: str = typer.Option(..., "--request", help="Your natural language request for analysis.")
):
    """
    데이터 파일과 사용자 요청을 기반으로 전체 통계 분석 파이프라인을 실행합니다.
    """
    # --- 초기화 ---
    print("--- Initializing System ---")
    
    # 환경변수 읽기
    use_rag = os.getenv("USE_RAG", "True").lower() == "true"
    rebuild_vector_store = os.getenv("REBUILD_VECTOR_STORE", "False").lower() == "true"

    # 경로 설정
    base_path = Path.cwd()
    input_file_path = base_path / "input_data/data_files" / file_name
    knowledge_base_path = str(base_path / "resources/knowledge_base")
    vector_store_path = str(base_path / "resources/rag_index") # 경로 변경
    report_path = base_path / "output_data/reports"
    report_path.mkdir(parents=True, exist_ok=True)

    # 컴포넌트 인스턴스화
    context = Context()
    agent = Agent()
    executor = CodeExecutor()

    context.set_user_input(file_path=str(input_file_path), request=request)
    print(f"User Request: {request}")
    print(f"Data File: {input_file_path}")

    # --- Step 1: RAG로 컨텍스트 강화 (조건부 실행) ---
    if use_rag:
        print("\n--- Step 1: Enhancing Context with RAG ---")
        retriever = RAGRetriever(
            knowledge_base_path=knowledge_base_path, 
            vector_store_path=vector_store_path,
            rebuild=rebuild_vector_store
        )
        retriever.load()
        rag_context = retriever.retrieve_context(request)
        context.add_rag_result(rag_context)
        print(f"RAG Context: {rag_context}")
    else:
        print("\n--- Step 1: Skipping RAG as per .env configuration ---")

    # --- Step 2: 데이터 로딩 및 초기 탐색 ---
    print("\n--- Step 2: Loading and Exploring Data ---")
    try:
        if input_file_path.suffix == '.csv':
            df = pd.read_csv(input_file_path)
        elif input_file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(input_file_path)
        elif input_file_path.suffix == '.parquet':
            df = pd.read_parquet(input_file_path)
        else:
            raise ValueError(f"Unsupported file type: {input_file_path.suffix}")
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        sys.exit(1)

    # --- Step 3: 통계 분석 계획 수립 ---
    print("\n--- Step 3: Generating Analysis Plan ---")
    schema = {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
    null_values = df.isnull().sum().to_dict()
    sample_data = df.head().to_string()
    context.set_data_info(schema=schema, null_values=null_values, sample_data=sample_data)

    plan = agent.generate_analysis_plan(context)
    context.set_analysis_plan(plan)
    print("Generated Analysis Plan:")
    for i, step in enumerate(plan, 1):
        print(f"{i}. {step}")
    
    # --- Step 4 & 5: 계획 기반 실행 및 자가 수정 루프 ---
    print("\n--- Step 4: Executing Analysis Plan ---")
    for i, step in enumerate(context.analysis_plan):
        print(f"\nExecuting Step {i+1}: {step}")
        
        code = agent.generate_code_for_step(context, step)
        print("Generated Code:\n" + code)

        result, success = executor.run(code, global_vars={'df': df})
        
        if success:
            print("Execution Result:\n" + result)
            context.add_to_history({'role': 'assistant', 'code': code})
            context.add_to_history({'role': 'system', 'result': result})
        else:
            print("Execution Failed. Error:\n" + result)
            context.add_to_history({'role': 'assistant', 'code': code})
            context.add_to_history({'role': 'system', 'error': result})
            
            print("\n--- Initiating Self-Correction ---")
            corrected_code = agent.self_correct_code(context, step, code, result)
            print("Corrected Code:\n" + corrected_code)
            
            result, success = executor.run(corrected_code, global_vars={'df': df})
            
            if success:
                print("Execution Result after correction:\n" + result)
                context.add_to_history({'role': 'assistant', 'code': corrected_code})
                context.add_to_history({'role': 'system', 'result': result})
            else:
                print("FATAL: Self-correction failed. Aborting analysis.")
                print("Final Error:\n" + result)
                sys.exit(1)
    
    # --- Step 6: 최종 보고서 생성 ---
    print("\n--- Step 5: Generating Final Report ---")
    final_report = agent.generate_final_report(context)
    context.set_final_report(final_report)
    
    # --- Step 7: 결과 출력 및 저장 ---
    print("\n\n" + "="*20 + " FINAL REPORT " + "="*20)
    print(final_report)
    print("="*54)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_file_name = f"report-{timestamp}.md"
    report_file_path = report_path / report_file_name
    
    with open(report_file_path, 'w', encoding='utf-8') as f:
        f.write(final_report)
        
    print(f"\nReport saved to: {report_file_path}")

if __name__ == "__main__":
    app() 