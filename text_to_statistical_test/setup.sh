#!/bin/bash
# =============================================================================
# Text-to-Statistical-Test 자동 설정 스크립트
# =============================================================================

set -e  # 오류 시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 함수 정의
print_header() {
    echo -e "${BLUE}"
    echo "============================================================="
    echo "🤖 Text-to-Statistical-Test 설정"
    echo "   RAG 기반 Agentic AI 통계 분석 시스템"
    echo "============================================================="
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}[단계] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

check_requirements() {
    print_step "시스템 요구사항 확인 중..."
    
    # Python 버전 확인
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION 발견"
    else
        print_error "Python 3.11+ 이 필요합니다."
        exit 1
    fi
    
    # Git 확인
    if command -v git &> /dev/null; then
        print_success "Git 사용 가능"
    else
        print_warning "Git이 설치되어 있지 않습니다."
    fi
}

setup_env_file() {
    print_step "환경변수 파일 설정..."
    
    if [ ! -f .env ]; then
        if [ -f env.example ]; then
            cp env.example .env
            print_success ".env 파일이 생성되었습니다."
            echo ""
            print_warning "중요: .env 파일을 열어 다음 API 키를 설정해주세요:"
            echo "  - OPENAI_API_KEY=your_openai_api_key_here"
            echo "  - ANTHROPIC_API_KEY=your_anthropic_api_key_here (선택사항)"
            echo ""
            
            # 사용자에게 API 키 입력 받기
            read -p "지금 API 키를 입력하시겠습니까? (y/n): " setup_keys
            if [[ $setup_keys =~ ^[Yy]$ ]]; then
                read -p "OpenAI API 키를 입력하세요: " openai_key
                if [ ! -z "$openai_key" ]; then
                    sed -i.bak "s/your_openai_api_key_here/$openai_key/" .env
                    print_success "OpenAI API 키가 설정되었습니다."
                fi
                
                read -p "Anthropic API 키를 입력하세요 (선택사항, Enter로 건너뛰기): " anthropic_key
                if [ ! -z "$anthropic_key" ]; then
                    sed -i.bak "s/your_anthropic_api_key_here/$anthropic_key/" .env
                    print_success "Anthropic API 키가 설정되었습니다."
                fi
                
                # 백업 파일 제거
                rm -f .env.bak
            fi
        else
            print_error "env.example 파일을 찾을 수 없습니다."
            exit 1
        fi
    else
        print_success ".env 파일이 이미 존재합니다."
    fi
}

install_method_menu() {
    print_step "설치 방법을 선택하세요:"
    echo "1) Poetry를 사용한 로컬 개발 환경"
    echo "2) Docker를 사용한 컨테이너 환경"
    echo "3) 환경변수만 설정하고 종료"
    echo ""
    read -p "선택 (1-3): " choice
    
    case $choice in
        1)
            install_with_poetry
            ;;
        2)
            install_with_docker
            ;;
        3)
            print_success "환경변수 설정이 완료되었습니다."
            print_usage_instructions
            ;;
        *)
            print_error "잘못된 선택입니다."
            exit 1
            ;;
    esac
}

install_with_poetry() {
    print_step "Poetry를 사용한 설치를 시작합니다..."
    
    # Poetry 설치 확인
    if ! command -v poetry &> /dev/null; then
        print_warning "Poetry가 설치되어 있지 않습니다."
        read -p "Poetry를 설치하시겠습니까? (y/n): " install_poetry
        if [[ $install_poetry =~ ^[Yy]$ ]]; then
            curl -sSL https://install.python-poetry.org | python3 -
            export PATH="$HOME/.local/bin:$PATH"
            print_success "Poetry가 설치되었습니다."
        else
            print_error "Poetry가 필요합니다. https://python-poetry.org/docs/#installation"
            exit 1
        fi
    fi
    
    # 의존성 설치
    print_step "의존성을 설치하는 중..."
    poetry install
    
    print_success "Poetry 설치가 완료되었습니다!"
    echo ""
    echo "실행 방법:"
    echo "  poetry run python main.py --help"
    echo "  poetry run python main.py --interactive"
}

install_with_docker() {
    print_step "Docker를 사용한 설치를 시작합니다..."
    
    # Docker 설치 확인
    if ! command -v docker &> /dev/null; then
        print_error "Docker가 설치되어 있지 않습니다."
        echo "Docker를 설치하세요: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Docker 이미지 빌드
    print_step "Docker 이미지를 빌드하는 중..."
    docker build -t text-to-statistical-test .
    
    print_success "Docker 이미지 빌드가 완료되었습니다!"
    echo ""
    echo "실행 방법:"
    echo "  docker run --env-file .env text-to-statistical-test"
    echo "  docker-compose up"
    
    # Docker Compose 사용 여부 확인
    if command -v docker-compose &> /dev/null; then
        read -p "Docker Compose로 바로 실행하시겠습니까? (y/n): " run_compose
        if [[ $run_compose =~ ^[Yy]$ ]]; then
            docker-compose up
        fi
    fi
}

print_usage_instructions() {
    echo ""
    echo "=========================================="
    echo "🚀 사용 방법"
    echo "=========================================="
    echo ""
    echo "Poetry 사용 시:"
    echo "  poetry run python main.py --help          # 도움말 보기"
    echo "  poetry run python main.py                 # 대화형 모드"
    echo "  poetry run python main.py --file data.csv # 특정 파일 분석"
    echo ""
    echo "Docker 사용 시:"
    echo "  docker run --env-file .env text-to-statistical-test"
    echo "  docker-compose up"
    echo ""
    echo "📚 자세한 사용법은 README.md를 참고하세요."
}

# 메인 실행 흐름
main() {
    print_header
    check_requirements
    setup_env_file
    install_method_menu
    print_usage_instructions
    
    print_success "설정이 완료되었습니다! 🎉"
}

# 스크립트 실행
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 