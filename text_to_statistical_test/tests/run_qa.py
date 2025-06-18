import json
import subprocess
import pathlib
import sys
from tqdm import tqdm

def run_qa_tests():
    """
    tests/qa.json에서 테스트 케이스를 읽어 분석 파이프라인을 실행하고,
    그 결과를 'report' 키에 추가하여 tests/qa_result.json으로 저장합니다.
    """
    # 스크립트가 위치한 디렉토리 (tests/)
    script_dir = pathlib.Path(__file__).parent
    # 프로젝트 루트 디렉토리 (tests/의 부모)
    project_root = script_dir.parent

    qa_file_path = script_dir / "qa.json"
    result_file_path = script_dir / "qa_result.json"
    data_dir = project_root / "input_data" / "data_files"

    if not qa_file_path.exists():
        print(f"오류: QA 파일을 찾을 수 없습니다: {qa_file_path}")
        sys.exit(1)

    with open(qa_file_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    test_cases = qa_data.get("questions", [])
    if not test_cases:
        print("오류: qa.json 파일에 테스트할 질문이 없습니다.")
        sys.exit(1)

    print(f"총 {len(test_cases)}개의 QA 테스트를 시작합니다...")

    for test_case in tqdm(test_cases, desc="QA 테스트 진행 중"):
        number = test_case.get("number")
        dataset = test_case.get("dataset")
        question = test_case.get("question")

        if not dataset or not question:
            test_case["report"] = "SKIPPED: 'dataset' 또는 'question' 필드가 누락되었습니다."
            continue

        file_path = data_dir / dataset
        if not file_path.exists():
            test_case["report"] = f"SKIPPED: 데이터셋 파일을 찾을 수 없습니다: {file_path}"
            continue
            
        command = [
            "poetry", "run", "python", "-m", "src.main",
            "--file", str(file_path),
            "--request", question
        ]

        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding='utf-8',
                check=False
            )

            if process.returncode == 0:
                report = process.stdout
                test_case["report"] = report.strip()
            else:
                error_message = (
                    f"ERROR: 테스트 #{number} 실패 (Exit Code: {process.returncode}).\n\n"
                    f"--- STDERR ---\n{process.stderr.strip()}"
                )
                test_case["report"] = error_message

        except Exception as e:
            error_message = f"CRITICAL ERROR: 테스트 #{number} 실행 중 예외 발생.\n\n{str(e)}"
            test_case["report"] = error_message
    
    with open(result_file_path, 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)

    print(f"\n모든 테스트가 완료되었습니다. 결과가 {result_file_path} 파일에 저장되었습니다.")

if __name__ == "__main__":
    run_qa_tests()
