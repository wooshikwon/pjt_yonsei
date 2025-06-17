import json
import os
import subprocess
import re
from pathlib import Path
import shlex

def load_qa_data(file_path: str) -> dict:
    """QA 데이터를 JSON 파일에서 로드합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_validation_results(data: dict, file_path: str):
    """검증 결과를 새로운 JSON 파일에 저장합니다."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_poetry_python_executable() -> str:
    """Poetry 가상환경의 Python 실행 파일 경로를 반환합니다."""
    try:
        # 'poetry env info -p'를 실행하여 가상환경 경로를 얻음
        env_path_output = subprocess.check_output(
            ["poetry", "env", "info", "-p"],
            text=True,
            encoding='utf-8'
        ).strip()
        # Windows와 Unix-like 시스템 모두 호환되도록 경로 조합
        return os.path.join(env_path_output, 'bin', 'python')
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Poetry 환경을 찾지 못할 경우 시스템의 poetry를 사용
        return "poetry"

def execute_command_in_poetry(command: str, poetry_executable: str) -> str:
    """
    주어진 명령어를 Poetry 가상환경에서 실행하고, 
    성공/실패 여부와 관계없이 전체 출력을 반환합니다.
    """
    # shlex.split()을 사용하여 따옴표로 묶인 인자를 올바르게 처리
    parts = shlex.split(command)
    
    if poetry_executable == "poetry":
        # poetry를 직접 실행
        run_command = ["poetry", "run"] + parts
    else:
        # 가상환경의 python 실행파일을 직접 사용
        run_command = [poetry_executable] + parts[1:]

    try:
        result = subprocess.run(
            run_command,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.returncode != 0:
            return f"Command failed with exit code {result.returncode}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        return result.stdout
    except Exception as e:
        return f"An unexpected error occurred while executing command: {' '.join(run_command)}\n{str(e)}"

def extract_detailed_results(report_content: str) -> str:
    """
    Markdown 보고서에서 '통계 검정 상세 결과' 섹션을 유연하게 추출합니다.
    "3." 앞뒤의 공백이나 마크업 변화에 대응합니다.
    """
    # 유연한 정규표현식
    match = re.search(
        r"###\s*3\.\s*통계 검정 상세 결과\s*\(Detailed Results\)([\s\S]*)",
        report_content,
        re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    
    # 실패 시 전체 내용 반환
    if "Error executing command" in report_content or "An unexpected error occurred" in report_content:
        return report_content
        
    return "상세 결과 섹션을 찾을 수 없습니다."

def main():
    """
    qa.json의 모든 명령어를 Poetry 환경에서 실행하고,
    결과를 파싱하여 validation_result.json에 저장합니다.
    """
    qa_file = Path(__file__).parent / "qa.json"
    result_file = Path(__file__).parent / "validation_result.json"
    
    print("🔍 Poetry 가상환경 경로를 찾는 중...")
    poetry_python = get_poetry_python_executable()
    print(f"✅ Poetry 실행 경로: {poetry_python}")
    
    qa_data = load_qa_data(qa_file)
    
    total_commands = sum(len(items) for items in qa_data.values())
    current_command = 0

    for dataset, items in qa_data.items():
        print(f"\n===== '{dataset}' 데이터셋 검증 시작 =====")
        for item in items:
            current_command += 1
            command = item["command"]
            
            print(f"  -> [{current_command}/{total_commands}] 실행: {item['tag']}")
            
            output = execute_command_in_poetry(command, poetry_python)
            
            detailed_results = extract_detailed_results(output)
            
            item["response"] = detailed_results
            
            print("     ... 결과 추출 완료")

    save_validation_results(qa_data, result_file)
    print(f"\n✅ 검증이 완료되었습니다. 결과가 {result_file}에 저장되었습니다.")

if __name__ == "__main__":
    main() 