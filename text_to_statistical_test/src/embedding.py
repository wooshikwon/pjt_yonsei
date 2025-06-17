import typer
from pathlib import Path
import sys
import os

# 프로젝트 루트를 시스템 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.components.rag_retriever import RAGRetriever
from src.utils.logger import get_logger

app = typer.Typer()

@app.command()
def build(
    force_rebuild: bool = typer.Option(
        False, 
        "--force-rebuild",
        "-f",
        help="기존 인덱스를 강제로 삭제하고 새로 생성합니다."
    )
):
    """
    'resources/knowledge_base'의 문서를 기반으로 FAISS 벡터 인덱스를 생성하거나 업데이트합니다.
    """
    logger = get_logger()
    logger.log_system_info("지식 베이스 빌드를 시작합니다...")
    
    try:
        base_path = Path.cwd()
        knowledge_base_path = str(base_path / "resources/knowledge_base")
        vector_store_path = str(base_path / "resources/rag_index")

        logger.log_detailed(f"지식 베이스 경로: {knowledge_base_path}")
        logger.log_detailed(f"벡터 스토어 경로: {vector_store_path}")

        retriever = RAGRetriever(
            knowledge_base_path=knowledge_base_path,
            vector_store_path=vector_store_path,
            rebuild=force_rebuild
        )

        retriever.load() # load 메서드가 내부적으로 빌드 또는 로드를 처리

        if force_rebuild:
            logger.log_system_info("✅ 지식 베이스를 성공적으로 재구축했습니다.")
        else:
            logger.log_system_info("✅ 지식 베이스가 최신 상태입니다.")

    except Exception as e:
        logger.log_system_info(f"❌ 지식 베이스 빌드 중 오류 발생: {e}")
        logger.log_detailed(str(e), "ERROR")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app() 