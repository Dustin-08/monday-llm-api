# Monday LLM API

Jetson Orin Nano 8GB Super Dev Kit 위에서 동작하는 개인 AI 비서 **Monday**용 FastAPI 서버입니다.
로컬에서 실행 중인 Ollama LLM과 연동하며, ChromaDB를 이용해 세션별 대화 기억 기능을 제공합니다.

## 주요 구성

- 하드웨어: NVIDIA Jetson Orin Nano 8GB Super Dev Kit
- 운영체제: Ubuntu 기반 Jetson 공식 이미지 (예: JetPack 환경)
- LLM 서버: Ollama (로컬에서 실행)
- 임베딩 모델: `nomic-embed-text` (Ollama embedding API 사용)
- 벡터 저장소: ChromaDB `PersistentClient` (로컬 디스크에 세션별 대화 기억 저장)
- 웹 서버: FastAPI + Uvicorn (내부 HTTP 호출은 httpx 비동기 클라이언트 사용)

## 주요 기능

- POST 엔드포인트(`/`) 하나로 Monday와 대화
- `session_id`로 사용자/대화방을 구분하여 기억이 서로 섞이지 않음
- 시스템 프롬프트에 한국어-only 규칙과 Monday 캐릭터 내장
- `memory.py`를 통한 대화 내용 임베딩/저장, 유사도 검색, 오래된 기억 자동 정리(TTL + 개수 상한)
- 오류 발생 시 상세 예외 내용은 서버 로그에만 남기고, 사용자에게는 참조번호만 반환

## 요구사항

- 하드웨어
  * Jetson Orin Nano 8GB Super Dev Kit
- 소프트웨어
  * Python 3.10 이상
  * `requirements.txt`에 명시된 패키지 (FastAPI, Uvicorn, httpx, Pydantic, ChromaDB)
  * Ollama (Jetson 또는 별도 머신에서 실행 가능)
  * Ollama에 `nomic-embed-text` 및 LLM 모델(예: `llama3.2:3b`) 설치

```bash
# Ollama 예시
ollama pull nomic-embed-text
ollama run llama3.2:3b
```

## 설치 방법

```bash
git clone https://github.com/Dustin-08/monday-llm-api.git
cd monday-llm-api

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## 환경 설정

이 프로젝트는 기본적으로 Jetson Orin Nano에서 로컬로 동작하는 Ollama와 ChromaDB를 사용하는 것을 가정합니다.
실제 Ollama 서버의 IP 주소와 포트는 환경에 따라 다르므로, 코드에서 직접 수정하거나 환경변수로 설정하세요.

- 예시 임베딩 API URL: `http://localhost:11434/api/embeddings`
- 예시 채팅 API URL: `http://localhost:11434/api/chat`

> ⚠️ 깃허브에 올릴 때는 학교/회사 서버의 실제 IP, 토큰 등 민감한 값은 하드코딩하지 마세요.
> 이 저장소에서는 `localhost`와 예시 포트만 사용합니다.

## 실행 방법

1. Ollama 서버 및 모델 준비

```bash
ollama serve
ollama run llama3.2:3b
ollama pull nomic-embed-text
```

2. FastAPI 서버 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 25565
```

3. 엔드포인트 호출

- 메서드: `POST`
- URL: `http://<서버_주소>:25565/`
  (Jetson 로컬 테스트 시 `http://localhost:25565/`)

요청 본문 예시:

```json
{
  "message": "오늘 일정 좀 정리해줘",
  "session_id": "user-001"
}
```

`session_id`를 생략하면 `"default"` 세션이 사용됩니다. 여러 사용자를 구분하려면
클라이언트에서 사용자마다 고유한 `session_id`를 지정해서 보내세요.

응답 예시:

```json
{
  "reply": "Monday가 한국어로 답변한 내용..."
}
```

## 메모리 시스템 (`memory.py`)

ChromaDB를 사용하여 대화 내용을 벡터로 저장하고, 현재 질문과 유사한 과거 대화를
**같은 session_id 안에서만** 검색합니다.

- `save_memory(user_message, ai_reply, session_id)`:
  `사용자: ... / Monday: ...` 형식으로 한 턴을 문자열로 만들고, 임베딩을 구해
  session_id 메타데이터와 함께 ChromaDB 컬렉션에 저장. 저장 후 세션별 개수 상한(`MAX_MEMORIES_PER_SESSION`)과
  TTL(`MEMORY_TTL_SECONDS`)을 넘긴 오래된 기억을 자동 정리
- `recall_memory(query, session_id, top_k=3)`:
  현재 질문을 임베딩으로 변환하고, 같은 세션 내에서 가장 유사한 과거 대화 `top_k`개를 검색하여
  "참고 데이터"임을 명시하는 구분자와 함께 시스템 프롬프트 뒤에 붙일 문자열 생성

구체적인 구현은 `memory.py`를 참고하세요.

## 테스트

```bash
pip install pytest
pytest test_main.py -v
```

Ollama/ChromaDB 없이도 실행 가능한 최소 스모크 테스트(요청 스키마, 프롬프트 구성)를 포함합니다.

## 라이선스

원하는 라이선스를 선택하여 `LICENSE` 파일을 추가하세요.
(예: MIT License)
