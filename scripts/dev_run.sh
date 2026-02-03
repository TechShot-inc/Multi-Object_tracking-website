#!/usr/bin/env bash
# Development run script for Multi-Object Tracking Web Application
#
# Usage:
#   ./scripts/dev_run.sh          # Run with defaults
#   ./scripts/dev_run.sh --port 8080
#   ./scripts/dev_run.sh --host 0.0.0.0 --port 5000
#
# Environment variables (optional):
#   MOT_HOST     - Host to bind to (default: 127.0.0.1)
#   MOT_PORT     - Port to bind to (default: 5000)
#   MOT_ENV      - Environment: dev or prod (default: dev)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default configuration
HOST="${MOT_HOST:-127.0.0.1}"
PORT="${MOT_PORT:-5000}"
ENV="${MOT_ENV:-dev}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --prod)
            ENV="prod"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host HOST    Host to bind to (default: 127.0.0.1)"
            echo "  --port PORT    Port to bind to (default: 5000)"
            echo "  --prod         Run in production mode"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  MOT Web - Development Server${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check if we're in a conda environment or venv
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo -e "${GREEN}✓ Conda environment: ${CONDA_DEFAULT_ENV}${NC}"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo -e "${GREEN}✓ Virtual environment: ${VIRTUAL_ENV}${NC}"
else
    echo -e "${YELLOW}⚠ No virtual environment detected${NC}"
    echo -e "  Consider activating your environment first:"
    echo -e "    conda activate mot"
    echo ""
fi

# Check if package is installed
if ! python -c "import mot_web" 2>/dev/null; then
    echo -e "${YELLOW}⚠ mot_web not installed, installing in editable mode...${NC}"
    pip install -e . --quiet
fi

# Create required directories
mkdir -p "$PROJECT_ROOT/var/uploads"
mkdir -p "$PROJECT_ROOT/var/results"

# Export environment variables
export HOST="$HOST"
export PORT="$PORT"
export ENV="$ENV"
export PROJECT_ROOT="$PROJECT_ROOT"
export UPLOAD_DIR="$PROJECT_ROOT/var/uploads"
export RESULTS_DIR="$PROJECT_ROOT/var/results"

echo -e "${GREEN}Configuration:${NC}"
echo -e "  Host:        $HOST"
echo -e "  Port:        $PORT"
echo -e "  Environment: $ENV"
echo -e "  Project:     $PROJECT_ROOT"
echo ""

# Run the application
echo -e "${GREEN}Starting server...${NC}"
echo -e "  URL: ${BLUE}http://${HOST}:${PORT}${NC}"
echo ""

if [[ "$ENV" == "prod" ]]; then
    # Production mode with gunicorn if available
    if command -v gunicorn &> /dev/null; then
        echo -e "${YELLOW}Running with Gunicorn (production)${NC}"
        exec gunicorn \
            --bind "${HOST}:${PORT}" \
            --workers 4 \
            --access-logfile - \
            --error-logfile - \
            "mot_web.app_factory:create_app()"
    else
        echo -e "${YELLOW}Gunicorn not installed, falling back to Flask dev server${NC}"
        exec python -m mot_web
    fi
else
    # Development mode with Flask's built-in server
    exec python -m mot_web
fi
