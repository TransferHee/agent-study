from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="gpt-5-mini",
    reasoning_effort="medium",
    temperature=0.0,
)

from langgraph.store.memory import InMemoryStore

# 1. Store 초기화
store = InMemoryStore()

# 2. Namespace(폴더 경로) 설정
user_id = "user_001"
application_context = "personal_assistant"

namespace = (user_id, application_context)

# 3. 첫 번째 데이터 저장
store.put(
    namespace,
    "memory_001",  # 이 데이터의 고유 식별자 (Key)
    {
        "facts": [
            "사용자는 커피보다 차를 선호함",
            "사용자는 매일 아침 6시에 일어남",
        ],
        "language": "Korean",
    },
)

# 4. 두 번째 데이터 저장
store.put(
    namespace,
    "memory_002",  # 이전 데이터가 덮어씌워지지 않도록 새로운 Key 지정
    {
        "facts": [
            "사용자는 그림 회화 작품을 좋아함",
            "빈센트 반 고흐의 작품을 특히 좋아함",
        ]
    },
)

# 특정 Key 조회 (get)
item1 = store.get(namespace, "memory_001")
print("1번 메모리:", item1.value)

item2 = store.get(namespace, "memory_002")
print("2번 메모리:", item2.value)

# Namespace 전체 검색 (search)
print("\n--- 전체 검색 결과 ---")
items = store.search(namespace)
for item in items:
    print(f"Key: {item.key} | Value: {item.value}")

from dataclasses import dataclass
from typing import TypedDict

@dataclass
class Context:
    user_id: str
    app_name: str

class UserInfo(TypedDict):
    personal_information : str
    preference: str

from langchain.agents.middleware import wrap_model_call

@wrap_model_call
def inject_memory(request, handler):
    current_user = request.runtime.context.user_id
    current_app = request.runtime.context.app_name
    memories = request.runtime.store.search((current_user, current_app))

    memory_content = "기록된 정보 없음"
    if memories:
        # 검색된 메모리들을 텍스트로 변환
        extracted_facts = []
        for item in memories:
            if "facts" in item.value:
                extracted_facts.extend(item.value["facts"])
        memory_content = "\n- ".join(extracted_facts)

    system_message = f"사용자 관련 장기 메모리 :\n- {memory_content}"
    request = request.override(system_prompt=system_message)
    return handler(request)

from langchain.agents import create_agent

agent = create_agent(
    model="gpt-5-nano",
    store=store,          # store 연결
    context_schema=Context,
    middleware=[inject_memory]  # 미들웨어 장착
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "나에 대해 알고있는 정보 알려줘"}]},
    context=Context(user_id="user_001", app_name="personal_assistant")
)
print(response["messages"][-1].content)

import uuid
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool, ToolRuntime

tool_store = InMemoryStore()

# --- 조회 도구 ---
@tool
def get_user_info(runtime) -> str:
    """
    현재 사용자의 정보 조회 (시스템 내부용 도구)
    """
    user_id = runtime.context.user_id
    app = runtime.context.app_name

    # 해당 네임스페이스의 모든 메모리 검색
    memories = runtime.store.search((user_id, app))

    if not memories:
        return "기록된 정보 없음"

    results = []
    for item in memories:
        # 저장된 데이터 구조(UserInfo)에 맞춰 필드 확인
        data = item.value
        if "personal_information" in data:
            results.append(f"- 개인정보: {data['personal_information']}")
        if "preference" in data:
            results.append(f"- 선호도: {data['preference']}")

    return "\n".join(results) if results else "데이터 형식 불일치로 읽을 수 없음"

# --- 저장 도구 ---
@tool
def save_user_info(user_info: UserInfo, runtime) -> str:
    """
    사용자의 정보를 저장하거나 업데이트
    """
    # 1. 실행 컨텍스트에서 user_id 가져오기
    user_id = runtime.context.user_id
    app = runtime.context.app_name
    store = runtime.store

    # 2. Store에 데이터 저장 (UUID를 생성하여 계속 누적)
    memory_key = str(uuid.uuid4())
    store.put((user_id, app), memory_key, user_info)

    return f"정보가 안전하게 저장되었습니다. (ID: {memory_key})"

from langchain.agents import create_agent

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_user_info, save_user_info],
    store=tool_store,         # 에이전트에 store 연결
    context_schema=Context
)

response1 = agent.invoke(
    {"messages": [{"role": "user", "content": "내 이름은 이제 'Alice'야. 커피보단 차를 좋아해"}]},
    context=Context(user_id="user_001", app_name="personal_assistant")
)
print(response1["messages"][-1].content)

response2 = agent.invoke(
    {"messages": [{"role": "user", "content": "나에 대해 아는 정보 말해줘"}]},
    context=Context(user_id="user_001", app_name="personal_assistant")
)
print(response2["messages"][-1].content)
