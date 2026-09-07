from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="gpt-5-mini",
    reasoning_effort="medium",
    temperature=0.0,
)

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
# from langgraph.checkpoint.postgres import PostgresSaver
# from psycopg import Connection


@tool
def fetch_aladin_bestseller_top10() -> Union[List[Dict[str, Any]], str]:
    """
    현재 시점의 알라딘 베스트셀러 Top 10 도서 목록을 조회하여 반환한다.
    반환값은 도서 정보가 담긴 딕셔너리의 리스트이거나, 호출 실패 시 에러 메시지(문자열)이다.
    """
    try:
        ttb_key = require_env("ALADIN_TTB_KEY")
        url = "http://www.aladin.co.kr/ttb/api/ItemList.aspx"
        params = {
            "ttbkey": ttb_key,
            "QueryType": "Bestseller",
            "MaxResults": 10,
            "start": 1,
            "SearchTarget": "Book",
            "output": "js",
            "Version": "20131101",
        }

        # API 호출 (타임아웃 10초 설정으로 무한 대기 방지)
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status() # HTTP 4xx, 5xx 에러 발생 시 예외 발생

        data = resp.json()

        # 모델이 핵심 정보에만 집중할 수 있도록, 전체 응답 중 도서 목록 10개만 슬라이싱하여 반환 (토큰 다이어트)
        return data.get("item", [])[:10]

    except Exception as e:
        # 에러가 발생해도 프로그램이 종료되지 않고, 모델에게 실패 원인을 텍스트로 알려줌 (우아한 실패)
        return f"API 호출 중 오류가 발생하여 베스트셀러 정보를 가져오지 못했습니다. 원인: {str(e)}"

@tool
def add(a: int, b: int) -> int:
    """`a`와 `b` 덧셈.

    Args:
        a: First int
        b: Second int
    """
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """`a`와 `b` 곱셈.

    Args:
        a: First int
        b: Second int
    """
    return a * b

@tool
def divide(a: int, b: int) -> float:
    """`a`와 `b` 나눗셈.

    Args:
        a: First int
        b: Second int
    """
    return a / b

tools = [add, multiply, divide, fetch_aladin_bestseller_top10]

# DB_URI = "postgresql://user:password@localhost:5432/agent_db"
# conn = Connection.connect(DB_URI)

# db_checkpointer = PostgresSaver(conn)
# db_checkpointer.setup() # 상태 저장을 위한 내부 테이블 자동 생성

# 1. InMemorySaver 객체를 생성하여 checkpointer 인자로 주입합니다.
agent = create_agent(
    model,
    tools,
    checkpointer=InMemorySaver(),
    # checkpointer=db_checkpointer,
)

cfg = {
    'configurable': {'thread_id': 'my_thread_id'}
}

response = agent.invoke(
    {'messages': [{'role': 'user', 'content': '안녕하세요! 저는 bytedance에서 일하는 jay입니다.'}]},
    config=cfg,
)

print('agent의 최종 응답:', response['messages'][-1].content)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "방금 제가 제 이름을 뭐라고 했죠?"}]},
    config=cfg,
)
print("[에이전트의 답변]:", response["messages"][-1].content)

for i, msg in enumerate(response["messages"], start=1):
    print(f"--- Message {i} ({msg.type}) ---")
    print(msg.content)
    print()