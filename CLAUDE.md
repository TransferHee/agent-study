# CLAUDE.md

이 파일은 이 repository에서 작업할 때 Claude Code(claude.ai/code)에게 제공되는 가이드입니다.

## Project Overview

이 repository는 `langchain`과 `langgraph`를 활용한 AI agent 개발을 학습하기 위한 project입니다.

## Architecture

### 디렉토리 구조

- `langchain_agents/` — langchain 기반 예제 (01~19번, 번호 접두사로 학습 순서 표시)
- `langgraph_agents/` — langgraph 기반 예제 (01번부터 새로 시작, langchain_agents와 독립적인 번호 체계)

### 환경 변수

- `.env`는 project root에만 위치. `python-dotenv`의 `load_dotenv()`는 스크립트 파일 경로 기준으로 상위 디렉토리를 탐색하므로 하위 디렉토리(`langchain_agents/`, `langgraph_agents/`)에서 실행해도 정상 인식됨.