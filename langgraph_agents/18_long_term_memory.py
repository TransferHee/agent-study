import uuid
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

model = init_chat_model("gpt-5-nano")

class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

from langgraph.store.base import BaseStore

def memory_agent_node(state: ChatState, config, store: BaseStore):
    # 1. 설정(config)에서 안전하게 user_id 가져오기 (환각 방지)
    user_id = config["configurable"]["user_id"]

    # 2. Namespace(기억 공간/폴더) 정의: (사용자ID, 카테고리) 튜플 구조
    namespace = (user_id, "profile")

    # 3. [WRITE] 장기 기억 저장 로직 ("기억해" 키워드 트리거)
    last_message = state["messages"][-1]
    if "기억해" in last_message.content:
        print(f"\n💾 [System] '{user_id}'님의 정보를 장기 기억장치에 저장합니다...")

        # uuid를 발급해 이전 데이터에 덮어씌워지지 않고 계속 누적(Append)되게 합니다.
        memory_id = str(uuid.uuid4())
        store.put(
            namespace,
            memory_id,
            {"content": last_message.content}  # 저장할 데이터 (딕셔너리 형태)
        )

    # 4. [READ] 장기 기억 조회 로직 (Push 패턴)
    memories = store.search(namespace)

    # 기억이 존재하면 모두 꺼내서 문자열로 병합합니다.
    if memories:
        memory_text = "\n".join([f"- {m.value['content']}" for m in memories])
        system_msg = f"""
        당신은 사용자의 정보를 기억하는 똑똑한 비서입니다.
        [장기 기억 저장소]
        {memory_text}
        위 기억을 참고하여 답변하세요.
        """
    else:
        system_msg = "당신은 도움이 되는 비서입니다. 아직 사용자에 대해 아는 것이 없습니다."

    # 5. LLM 호출 (시스템 메시지를 대화 내역 맨 앞에 몰래 끼워 넣습니다)
    prompt = [SystemMessage(content=system_msg)] + state["messages"]
    return {"messages": [model.invoke(prompt)]}

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

workflow = StateGraph(ChatState)
workflow.add_node("agent", memory_agent_node)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# 메모리 저장소 생성
checkpointer = InMemorySaver()
store = InMemoryStore()

# 단기 메모리와 장기 메모리를 동시에 주입하여 컴파일
app = workflow.compile(checkpointer=checkpointer, store=store)

from langchain.messages import HumanMessage

# [대화방 1] 정보 저장하기 (thread-1)
config_1 = {"configurable": {"thread_id": "thread-1", "user_id": "user-jay"}}

input1 = {"messages": [HumanMessage(content="내 이름은 Jay이고, 나는 매운 음식을 싫어해. 이거 꼭 기억해.")]}
resp1 = app.invoke(input1, config=config_1)

print(f"응답 1: {resp1['messages'][-1].content}")

# [대화방 2] 다른 스레드에서 기억 꺼내 쓰기 (thread-2)
config_2 = {"configurable": {"thread_id": "thread-2", "user_id": "user-jay"}}

input2 = {"messages": [HumanMessage(content="나 오늘 점심 메뉴 추천해 줘.")]}
resp2 = app.invoke(input2, config=config_2)

print(f"응답 2: {resp2['messages'][-1].content}")

from langchain.tools import tool
from langchain_core.messages import ToolMessage

# 1. 도구(Tool) 정의: AI에게 쥐어줄 '기억 버튼'
@tool
def save_profile(info: str):
    """
    사용자에 대한 중요한 정보(이름, 취미, 특징 등)를 저장할 때 사용합니다.
    단순한 대화나 인사는 저장하지 마세요.
    """
    print(f'저장한 정보 : {info}')
    # 이 함수는 실제로는 실행되지 않고, 스키마(껍데기)만 LLM에게 전달
    # 실제 저장 로직은 아래 'save_node'에서 store 객체를 이용해 수행
    return "saved"

tools = [save_profile]
model_with_tools = model.bind_tools(tools)

# [Node 1] 뇌 (Agent): 판단하고 도구 호출 결정
def agent_node(state: ChatState, config, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "profile")

    # 1. 저장된 기억을 조회해서 컨텍스트로 주입
    memories = store.search(namespace)
    if memories:
        info = "\n".join([f"- {m.value['data']}" for m in memories])
        system_msg = f"당신은 사용자의 기억을 담당하는 비서입니다.\n[기억된 정보]\n{info}"
    else:
        system_msg = "당신은 사용자의 기억을 담당하는 비서입니다."

    # 2. 도구를 쓸 줄 아는 모델 호출 (AI가 스스로 save_profile 호출을 결정)
    return {"messages": [model_with_tools.invoke([SystemMessage(content=system_msg)] + state["messages"])]}

# [Node 2] 손발 (Action): 에이전트의 지시를 받아 실제 Store에 기록
def save_node(state: ChatState, config, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "profile")

    last_message = state["messages"][-1]
    tool_outputs = []

    # 에이전트가 요청한 도구 호출(tool_calls) 내역을 뒤져서 실제 DB에 저장
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "save_profile":
            info_to_save = tool_call["args"]["info"]
            print(f"\n💾 [System] AI의 자율적 판단으로 정보를 저장합니다: '{info_to_save}'")

            # 무작위 UUID를 키로 사용하여 항목을 계속 누적(Append) 합니다.
            memory_id = str(uuid.uuid4())
            store.put(namespace, memory_id, {"data": info_to_save})

            # 도구 실행 결과를 LLM에게 돌려주기 위한 시스템 메시지 생성
            tool_outputs.append(
                ToolMessage(
                    content=f"정보 저장 완료: {info_to_save}",
                    tool_call_id=tool_call["id"]
                )
            )

    return {"messages": tool_outputs}

workflow = StateGraph(ChatState)
workflow.add_node("agent", agent_node)
workflow.add_node("save_node", save_node)

workflow.add_edge(START, "agent")

# 조건부 라우팅: 도구를 호출했으면 save_node로, 아니면 END
def should_continue(state: ChatState):
    if state["messages"][-1].tool_calls:
        return "save_node"
    return END

workflow.add_conditional_edges("agent", should_continue, ["save_node", END])

# 저장이 끝나면 다시 에이전트로 돌아가 "저장 완료했습니다"라고 자연스럽게 대화를 이어가게 함
workflow.add_edge("save_node", "agent") 

# 컴파일
app = workflow.compile(checkpointer=InMemorySaver(), store=InMemoryStore())

config = {"configurable": {"thread_id": "1", "user_id": "user-jay"}}

# 자연스러운 대화 입력
input1 = {"messages": [HumanMessage(content="안녕, 나는 샌프란시스코에 사는 Jay라고 해.")]}
resp1 = app.invoke(input1, config=config)

print(f"응답 1: {resp1['messages'][-1].content}")

input2 = {"messages": [HumanMessage(content="내 이름이 뭐고 어디 산다고 했지?")]}
resp2 = app.invoke(input2, config=config)
print(resp2["messages"][-1].content)
