import chromadb
import requests
import time

# Ollama 임베딩 API URL과 임베딩 모델 설정
# 실제 환경에 맞게 URL을 수정할 수 있습니다.
# 예시에서는 로컬에서 실행 중인 Ollama 서버를 사용합니다.
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"  # 예시용 임베딩 API URL
EMBED_MODEL = "nomic-embed-text"  # Ollama에서 제공하는 임베딩 전용 모델 이름[web:21][web:23]

# ChromaDB PersistentClient 초기화
# "./monday_memory" 경로 아래에 벡터 저장소가 생성됩니다.
client = chromadb.PersistentClient(path="./monday_memory")
collection = client.get_or_create_collection(name="monday_memory")


def get_embedding(text: str):
    """
    주어진 텍스트를 Ollama 임베딩 API를 통해 벡터로 변환합니다.

    - text: 임베딩을 구할 문자열
    - 반환값: 벡터(embedding) 리스트
    """
    try:
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=10,  # 네트워크 문제가 있을 때 너무 오래 대기하지 않도록 타임아웃 설정
        )
        response.raise_for_status()
        data = response.json()
        # Ollama 임베딩 응답은 통상적으로 "embedding" 키를 포함합니다.[web:21][web:23]
        return data["embedding"]
    except Exception as e:
        # 임베딩을 구하지 못하면 빈 리스트를 반환하거나,
        # 상위 로직에서 적절히 처리할 수 있도록 예외를 다시 던질 수 있습니다.
        # 여기서는 간단히 빈 리스트를 반환하여 메모리 저장/검색을 건너뛰게 합니다.
        print(f"[memory] 임베딩 생성 중 오류 발생: {e}")
        return []


def save_memory(user_message: str, ai_reply: str):
    """
    대화 한 턴을 기억으로 저장합니다.

    - user_message: 사용자가 보낸 메시지
    - ai_reply: Monday가 응답한 내용
    """
    combined = f"사용자: {user_message}\nMonday: {ai_reply}"
    embedding = get_embedding(combined)

    # 임베딩 생성에 실패한 경우(빈 리스트)에는 저장을 건너뜁니다.
    if not embedding:
        print("[memory] 임베딩이 없어서 현재 대화를 저장하지 않습니다.")
        return

    # 간단한 타임스탬프 기반 ID 생성
    doc_id = f"mem_{int(time.time() * 1000)}"

    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[combined],
    )


def recall_memory(query: str, top_k: int = 3) -> str:
    """
    현재 질문과 관련된 과거 기억을 검색합니다.

    - query: 현재 사용자의 질문
    - top_k: 최대 몇 개의 과거 대화를 가져올지
    - 반환값: 시스템 프롬프트에 이어 붙일 문자열 (없으면 빈 문자열)
    """
    # 저장된 문서가 하나도 없으면 빈 문자열 반환
    if collection.count() == 0:
        return ""

    query_embedding = get_embedding(query)
    # 임베딩을 구하지 못했으면 검색을 건너뜁니다.
    if not query_embedding:
        print("[memory] 쿼리 임베딩이 없어서 과거 기억 검색을 건너뜁니다.")
        return ""

    # 컬렉션에서 유사도가 높은 문서 top_k개를 검색
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    # ChromaDB는 documents를 2차원 리스트 형태로 반환합니다.
    docs = results.get("documents", [[]])[0]
    if not docs:
        return ""

    # 시스템 프롬프트 뒤에 자연스럽게 붙일 수 있는 형식으로 문자열 구성
    return "\n\n[참고할 만한 과거 대화 기록]\n" + "\n---\n".join(docs)
