from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

# 1. 모델 준비
model = init_chat_model("gpt-5-nano")

# 2. State 정의 (메시지 누적 방식)
class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

from langgraph.graph import StateGraph, START, END

# 3. 노드 정의
def chatbot_node(state: ChatState):
    return {"messages": [model.invoke(state["messages"])]}

# 4. 그래프 조립
workflow = StateGraph(ChatState)
workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

from langgraph.checkpoint.memory import InMemorySaver

# 메모리 저장소 객체 생성
memory = InMemorySaver()

# 체크포인터와 함께 컴파일 (메모리 기능이 켜진 에이전트 탄생)
app = workflow.compile(checkpointer=memory)

from langchain.messages import HumanMessage

# 1번 대화방(스레드) 설정
config_1 = {"configurable": {"thread_id": "1"}}

# 첫 번째 질문
input_msg1 = {"messages": [HumanMessage(content="안녕? 내 이름은 Jay야.")]}
response1 = app.invoke(input_msg1, config=config_1)
print(response1['messages'][-1].content)

# 두 번째 질문 (같은 스레드 유지)
input_msg2 = {"messages": [HumanMessage(content="내 이름이 뭐라고?")]}
response2 = app.invoke(input_msg2, config=config_1)
print(response2['messages'][-1].content)

# 2번 대화방(스레드)으로 변경
config_2 = {"configurable": {"thread_id": "2"}}

input_msg3 = {"messages": [HumanMessage(content="내 이름이 뭐라고?")]}
response3 = app.invoke(input_msg3, config=config_2)
print(response3['messages'][-1].content)

current_state = app.get_state(config_1)
print(current_state)

# 과거부터 현재까지의 스냅샷 목록 가져오기
history = list(app.get_state_history(config_1))

# 보기 쉽게 반복문으로 출력
for i, snapshot in enumerate(history):
    print(f"\n[Snapshot {i}]")
    print(f" - 시간: {snapshot.created_at}")

    msgs = snapshot.values.get("messages", [])
    if msgs:
        last_msg = msgs[-1]
        sender = "AI" if last_msg.type == "ai" else "User"
        print(f" - 마지막 대화: [{sender}] {last_msg.content[:20]}...")
    else:
        print(" - (대화 시작 전 초기 상태)")

    print(f" - 다음 행선지(Next): {snapshot.next}")

