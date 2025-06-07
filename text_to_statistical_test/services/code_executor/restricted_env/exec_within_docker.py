# 파일명: services/code_executor/restricted_env/exec_within_docker.py
import docker
import os
import io
import tarfile
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# 사용할 Docker 이미지 (필요한 라이브러리가 설치된 커스텀 이미지)
# 이 이미지는 프로젝트 루트의 Dockerfile로 빌드해야 합니다.
DOCKER_IMAGE_NAME = "ttst-sandbox:latest"

logger = logging.getLogger(__name__)

class ExecWithinDocker:
    """Docker 컨테이너를 사용하여 안전하게 코드를 실행하는 클래스"""

    def __init__(self, session_id: str, df: pd.DataFrame):
        self.session_id = session_id
        self.df = df
        self.container_name = f"ttst-session-{self.session_id}"
        self.client = docker.from_env()
        self.workdir = Path(f"/tmp/{self.session_id}")
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.container = None

    def _start_container(self):
        """세션용 Docker 컨테이너를 시작합니다."""
        try:
            # 기존에 같은 이름의 컨테이너가 있다면 중지하고 삭제
            self.container = self.client.containers.get(self.container_name)
            if self.container.status == 'running':
                self.container.stop()
            self.container.remove()
            logger.warning(f"기존 컨테이너 '{self.container_name}'를 삭제했습니다.")
        except docker.errors.NotFound:
            pass # 컨테이너가 없으면 정상

        try:
            logger.info(f"컨테이너 '{self.container_name}'를 시작합니다...")
            self.container = self.client.containers.run(
                DOCKER_IMAGE_NAME,
                name=self.container_name,
                detach=True,
                tty=True, # 계속 실행 상태로 유지
                working_dir="/app",
                # [TODO] 리소스 제한 추가
                # mem_limit="512m",
                # cpus=1,
            )
            # 데이터프레임을 컨테이너 내부로 복사
            self._copy_df_to_container()
        except docker.errors.ImageNotFound:
            logger.error(f"Docker 이미지 '{DOCKER_IMAGE_NAME}'를 찾을 수 없습니다.")
            logger.error("프로젝트 루트에서 'docker build -t ttst-sandbox:latest .' 명령을 실행하여 이미지를 빌드하세요.")
            raise
        except Exception as e:
            logger.error(f"컨테이너 시작 중 오류 발생: {e}")
            raise

    def _copy_df_to_container(self):
        """데이터프레임을 CSV로 저장하여 컨테이너 내부로 복사합니다."""
        if self.container is None:
            raise RuntimeError("컨테이너가 실행 중이 아닙니다.")

        local_df_path = self.workdir / "data.csv"
        self.df.to_csv(local_df_path, index=False)
        
        logger.info(f"데이터프레임을 '{local_df_path}'에 저장하고 컨테이너 내부 /app/data.csv 로 복사합니다.")
        
        # 파일을 tar 아카이브로 만들어 메모리 상에서 전달
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode='w') as tar:
            tar.add(local_df_path, arcname="data.csv")
        
        tar_stream.seek(0)
        self.container.put_archive("/app/", tar_stream)

    def run(self, code: str) -> Dict[str, Any]:
        """컨테이너 내에서 코드를 실행하고 결과를 반환합니다."""
        self._start_container()
        
        # 실행할 코드를 파일로 만들어 컨테이너에 복사
        local_code_path = self.workdir / "script.py"
        local_code_path.write_text(self._wrap_code(code), encoding='utf-8')
        
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode='w') as tar:
            tar.add(local_code_path, arcname="script.py")
        tar_stream.seek(0)
        self.container.put_archive("/app/", tar_stream)
        
        # 컨테이너에서 스크립트 실행
        logger.info("컨테이너 내부에서 'python script.py'를 실행합니다.")
        exit_code, (stdout_bytes, stderr_bytes) = self.container.exec_run("python script.py")

        stdout = stdout_bytes.decode('utf-8', errors='ignore')
        stderr = stderr_bytes.decode('utf-8', errors='ignore')

        result = {
            "success": exit_code == 0,
            "output": stdout,
            "error": stderr,
            "plots": self._get_artifacts_from_container()
        }
        
        self._stop_container()
        return result

    def _get_artifacts_from_container(self) -> list:
        """컨테이너에서 생성된 이미지 파일을 로컬로 가져오고 경로를 반환합니다."""
        local_artifact_dir = self.workdir / "artifacts"
        local_artifact_dir.mkdir(exist_ok=True)
        
        try:
            # /app/artifacts 디렉토리가 있는지 확인
            self.container.exec_run("ls /app/artifacts")

            bits, stat = self.container.get_archive("/app/artifacts")
            
            # tar 스트림을 로컬에 풀기
            with tarfile.open(fileobj=io.BytesIO(bits.read())) as tar:
                tar.extractall(path=local_artifact_dir)
            
            # 모든 파일 경로 반환
            plot_paths = [str(p) for p in local_artifact_dir.glob("*")]
            logger.info(f"컨테이너에서 {len(plot_paths)}개의 결과물(artifact)을 가져왔습니다.")
            return plot_paths
        except docker.errors.APIError:
            logger.info("컨테이너에 'artifacts' 디렉토리가 없거나 비어있습니다. 결과물을 가져오지 않습니다.")
            return []


    def _wrap_code(self, user_code: str) -> str:
        """사용자 코드를 실행 및 결과 저장을 위한 템플릿으로 감쌉니다."""
        return f"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 결과물(artifact)을 저장할 디렉토리 생성
os.makedirs('/app/artifacts', exist_ok=True)
plt.ioff() # 대화형 모드 끄기

try:
    # 데이터 로드
    df = pd.read_csv('/app/data.csv')
    
    # --- 사용자 코드 시작 ---
{user_code}
    # --- 사용자 코드 종료 ---

    # 생성된 모든 plot을 파일로 저장
    for i, fig in enumerate(plt.get_fignums()):
        plt.figure(fig)
        plt.savefig(f'/app/artifacts/plot_{{i+1}}.png')

except Exception as e:
    import traceback
    traceback.print_exc()

finally:
    plt.close('all')
"""

    def _stop_container(self):
        """컨테이너를 중지하고 삭제합니다."""
        if self.container:
            try:
                self.container.stop()
                self.container.remove()
                logger.info(f"컨테이너 '{self.container_name}'를 중지하고 삭제했습니다.")
            except docker.errors.APIError as e:
                logger.warning(f"컨테이너 정리 중 오류 발생 (이미 삭제되었을 수 있음): {e}")
            finally:
                self.container = None 