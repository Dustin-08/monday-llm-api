from fastapi import FastAPI
from pydantic import BaseModel
import requests
from memory import save_memory, recall_memory  # 대화 내용을 저장/조회하는 사용자 정의 모듈

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(title="Monday LLM API")

# 클라이언트에서 들어오는 요청 바디를 정의하는 Pydantic 모델
class ChatRequest(BaseModel):
    message: str  # 사용자가 보낸 메시지 텍스트

# Ollama 서버와 모델 이름 설정
# 실제 환경에 맞게 URL과 모델 이름을 수정하세요.
# 깃허브에는 로컬 테스트용 값만 예시로 두는 것을 권장합니다.
OLLAMA_URL = "http://localhost:11434/api/chat"  # 예시용 로컬 Ollama API URL
MODEL_NAME = "llama3.2:3b"  # 예시용 모델 이름

@app.post("/")
async def chat_with_llm(request: ChatRequest):
    """
    Monday LLM과 대화하기 위한 기본 엔드포인트.

    1. 사용자의 입력 메시지를 바탕으로 관련 기억을 조회합니다.
    2. 시스템 프롬프트에 기억을 포함하여 Ollama 서버에 전달합니다.
    3. Ollama의 응답을 받아서 다시 사용자의 기억으로 저장합니다.
    4. 최종적으로 사용자에게 응답 텍스트를 반환합니다.
    """

    # 1. 관련 기억 검색
    # 사용자가 이전에 했던 대화 중 현재 메시지와 연관된 내용을 가져옵니다.
    memory_context = recall_memory(request.message)

    # 시스템 프롬프트 정의
    # Monday의 성격, 한국어-only 제약 조건, 과거 기억 활용 등을 명시합니다.
    system_prompt = (
        "You must strictly reply in Korean language only. "
        "너는 사용자의 고성능 개인 AI 비서 'Monday(먼데이)'야. "
        "누가 이름을 물어보거나 비서로서 대답할 때 반드시 자신을 'Monday'라고 소개해줘. "
        "모든 대화는 반드시 자연스럽고 완벽한 '한국어'로만 답변해야 해. "
        "절대로 영어, 한자, 일본어, 힌디어 등 다른 외국어를 문장에 섞어서 쓰지 마. "
        "특히 한국어 조사 뒤에 외국어 단어를 붙이는 행위는 절대 금지한다. "
        "아래에 과거 대화 기록이 있다면, 사용자가 예전에 알려준 정보(이름, 취향, 진행 중인 일 등)를 "
        "기억하고 있는 것처럼 자연스럽게 활용해서 답변해."
        + memory_context  # recall_memory에서 가져온 과거 대화 내용을 뒤에 이어 붙입니다.
    )

    # Ollama에 전달할 payload 구성
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,  # 응답의 창의성 정도 (낮을수록 보수적)
            "top_p": 0.7,        # 샘플링 범위 설정
        },
    }

    try:
        # Ollama 서버로 POST 요청 전송
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()  # HTTP 에러 발생 시 예외 던짐

        # Ollama에서 반환된 JSON을 파싱
        result = response.json()
        ai_response = result["message"]["content"]  # LLM 응답 텍스트

        # 2. 이번 대화를 기억으로 저장
        save_memory(request.message, ai_response)

        # 최종 응답을 JSON 형태로 반환
        return {"reply": ai_response}

    except Exception as e:
        # Ollama 서버 통신 중 에러가 발생한 경우,
        # 사용자에게 간단한 오류 메시지를 전달합니다.
        return {"reply": f"Monday 서버 통신 중 오류가 발생했습니다: {str(e)}"}


if __name__ == "__main__":
    import uvicorn

    # 개발/테스트용 서버 실행.
    # host와 port는 예시 값이며, 실제 운영 환경에 맞게 변경하십시오.
    #
    # 예시:
    #   host="0.0.0.0"  -> 모든 네트워크 인터페이스에서 접속 허용
    #   port=25565      -> 예시 포트 번호 (원하는 값으로 변경 가능)
    #
    # 깃허브에는 구체적인 외부 IP를 적지 않고, host/port를 예시 수준으로 두는 것을 추천합니다.
    uvicorn.run(app, host="0.0.0.0", port=25565)
