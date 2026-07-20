# Monday LLM API

Jetson Orin Nano 8GB Super Dev Kit 위에서 동작하는 개인 AI 비서 **Monday**용 FastAPI 서버입니다.  
로컬에서 실행 중인 Ollama LLM과 연동하며, ChromaDB를 이용해 간단한 대화 기억 기능을 제공합니다.[web:20][web:24]

## 주요 구성

- 하드웨어: NVIDIA Jetson Orin Nano 8GB Super Dev Kit
- 운영체제: Ubuntu 기반 Jetson 공식 이미지 (예: JetPack 환경)
- LLM 서버: Ollama (로컬에서 실행)
- 임베딩 모델: `nomic-embed-text` (Ollama embedding API 사용)[web:21][web:23]
- 벡터 저장소: ChromaDB `PersistentClient` (로컬 디스크에 대화 기억 저장)[web:20][web:25]
- 웹 서버: FastAPI + Uvicorn

## 주요 기능

- POST 엔드포인트(`/`) 하나로 Monday와 대화
- 시스템 프롬프트에 한국어-only 규칙과 Monday 캐릭터 내장
- `memory.py`를 통한 대화 내용 임베딩/저장 및 유사도 검색
- Jetson Orin Nano 상에서 로컬 LLM 기반 개인 비서 환경 구현

## 요구사항

- 하드웨어
  - Jetson Orin Nano 8GB Super Dev Kit
- 소프트웨어
  - Python 3.10 이상
  - FastAPI, Uvicorn, Requests, Pydantic, ChromaDB
  - Ollama (Jetson 또는 별도 머신에서 실행 가능)
  - Ollama에 `nomic-embed-text` 및 LLM 모델(예: `llama3.2:3b`) 설치[web:21]

```bash
# Ollama 예시
ollama pull nomic-embed-text
ollama run llama3.2:3b
```

## 설치 방법

```bash
git clone https://github.com/<YOUR_GITHUB_ID>/monday-llm-api.git
cd monday-llm-api

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` 예시:

```text
fastapi
uvicorn
requests
pydantic
chromadb
```

## 환경 설정

이 프로젝트는 기본적으로 Jetson Orin Nano에서 로컬로 동작하는 Ollama와 ChromaDB를 사용하는 것을 가정합니다.  
실제 Ollama 서버의 IP 주소와 포트는 환경에 따라 다르므로, 코드에서 직접 수정하거나 환경변수로 설정하세요.

- 예시 임베딩 API URL: `http://localhost:11434/api/embeddings`
- 예시 채팅 API URL: `http://localhost:11434/api/chat`

> ⚠️ 깃허브에 올릴 때는 학교/회사 서버의 실제 IP, 토큰 등 민감한 값은 하드코딩하지 마세요.  
> 이 저장소에서는 `localhost`와 예시 포트만 사용합니다.[web:16][web:19]

## 실행 방법

1. Ollama 서버 및 모델 준비

```bash
ollama serve
ollama run llama3.2:3b
ollama pull nomic-embed-text
```

2. FastAPI 서버 실행

```bash
uvicorn main:app --host 0.0.0.0 --port <예시_포트_번호>
```

3. 엔드포인트 호출

- 메서드: `POST`
- URL: `http://<서버_주소>:<포트>/`  
  (Jetson 로컬 테스트 시 `http://localhost:<포트>/`)

요청 본문 예시:

```json
{
  "message": "오늘 일정 좀 정리해줘"
}
```

응답 예시:

```json
{
  "reply": "Monday가 한국어로 답변한 내용..."
}
```

## 메모리 시스템 (`memory.py`)

이 프로젝트는 ChromaDB를 사용하여 대화 내용을 벡터로 저장하고, 현재 질문과 유사한 과거 대화를 검색합니다.[web:20][web:25]

- `save_memory(user_message: str, ai_reply: str)`:  
  `사용자: ... / Monday: ...` 형식으로 한 턴을 문자열로 만들고, 임베딩을 구해 ChromaDB 컬렉션에 저장
- `recall_memory(query: str, top_k: int = 3) -> str`:  
  현재 질문을 임베딩으로 변환하고, 가장 유사한 과거 대화 `top_k`개를 검색하여 시스템 프롬프트 뒤에 붙일 문자열 생성

구체적인 구현은 `memory.py`를 참고하세요.

## 라이선스

원하는 라이선스를 선택하여 `LICENSE` 파일을 추가하세요.  
(예: MIT License)
