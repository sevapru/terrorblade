#!/bin/bash

# Script to run GitHub Actions workflows locally
# This allows you to test CI workflows before pushing to GitHub

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if act is installed
check_act() {
    if ! command -v act &> /dev/null; then
        echo -e "${YELLOW}act is not installed. You can install it with:${NC}"
        echo -e "${BLUE}  curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash${NC}"
        echo -e "${BLUE}  Or visit: https://github.com/nektos/act${NC}"
        echo ""
        echo -e "${YELLOW}Alternatively, you can run individual workflow jobs using make commands.${NC}"
        return 1
    fi
    return 0
}

# Function to run workflow with act
run_with_act() {
    local workflow=$1
    local job=$2
    
    echo -e "${BLUE}Running $workflow workflow with act...${NC}"
    
    if [ -n "$job" ]; then
        act -W .github/workflows/$workflow --job $job
    else
        act -W .github/workflows/$workflow
    fi
}

# Function to run workflow equivalents with make
run_with_make() {
    local workflow=$1
    
    case $workflow in
        "ci.yml"|"ci")
            echo -e "${BLUE}Running CI workflow equivalent with make...${NC}"
            make ci-local
            ;;
        "security.yml"|"security")
            echo -e "${BLUE}Running Security workflow equivalent with make...${NC}"
            make security-ci-local
            ;;
        *)
            echo -e "${RED}Unknown workflow: $workflow${NC}"
            echo -e "${YELLOW}Available workflows: ci, security${NC}"
            exit 1
            ;;
    esac
}

# Help function
show_help() {
    echo -e "${GREEN}Local CI Workflow Runner${NC}"
    echo -e "${BLUE}========================${NC}"
    echo ""
    echo -e "${YELLOW}Usage:${NC}"
    echo "  $0 [workflow] [options]"
    echo ""
    echo -e "${YELLOW}Workflows:${NC}"
    echo "  ci          Run CI workflow (lint, test, security)"
    echo "  security    Run security workflow (vulnerability scans)"
    echo "  all         Run all workflows"
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "  --with-act  Use act to run actual GitHub Actions (requires act)"
    echo "  --make-only Use make commands only (default if act not available)"
    echo "  --job JOB   Run specific job (only with --with-act)"
    echo "  --help, -h  Show this help message"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 ci                    # Run CI with make"
    echo "  $0 security --with-act   # Run security with act"
    echo "  $0 ci --with-act --job lint  # Run specific CI job with act"
    echo "  $0 all                   # Run all workflows with make"
}

# Parse command line arguments
WORKFLOW=""
USE_ACT=false
MAKE_ONLY=false
JOB=""

while [[ $# -gt 0 ]]; do
    case $1 in
        ci|security|all)
            WORKFLOW="$1"
            shift
            ;;
        --with-act)
            USE_ACT=true
            shift
            ;;
        --make-only)
            MAKE_ONLY=true
            shift
            ;;
        --job)
            JOB="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Check if workflow is specified
if [ -z "$WORKFLOW" ]; then
    echo -e "${RED}No workflow specified${NC}"
    show_help
    exit 1
fi

# Determine execution method
if [ "$USE_ACT" = true ] && [ "$MAKE_ONLY" = false ]; then
    if check_act; then
        echo -e "${GREEN}Using act to run GitHub Actions locally${NC}"
        if [ "$WORKFLOW" = "all" ]; then
            run_with_act "ci.yml"
            run_with_act "security.yml"
        else
            run_with_act "$WORKFLOW.yml" "$JOB"
        fi
    else
        echo -e "${YELLOW}Falling back to make commands${NC}"
        run_with_make "$WORKFLOW"
    fi
else
    echo -e "${GREEN}Using make commands to run workflow equivalents${NC}"
    if [ "$WORKFLOW" = "all" ]; then
        run_with_make "ci"
        run_with_make "security"
    else
        run_with_make "$WORKFLOW"
    fi
fi

echo -e "${GREEN}Workflow execution completed!${NC}" 