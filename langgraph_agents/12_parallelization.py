from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

# 1. 모델 초기화
model = init_chat_model("gpt-5-nano")

# 2. State(상태) 정의
class WriterState(TypedDict):
    topic: str          # 사용자가 입력할 주제
    poem: str           # 시 (Worker A의 결과물)
    story: str          # 소설 (Worker B의 결과물)
    joke: str           # 농담 (Worker C의 결과물)
    final_report: str   # 편집장이 취합할 최종 편집본

# 3. Node 정의
# [Worker A] 시인 노드
def write_poem(state: WriterState):
    topic = state["topic"]
    print(f"   [Worker A] '{topic}' 주제로 시(Poem) 작성 시작...")
    msg = model.invoke(f"'{topic}'에 대한 아름다운 시를 짧게 써줘.")
    return {"poem": msg.content}

# [Worker B] 소설가 노드
def write_story(state: WriterState):
    topic = state["topic"]
    print(f"   [Worker B] '{topic}' 주제로 소설(Story) 작성 시작...")
    msg = model.invoke(f"'{topic}'에 대한 감동적인 짧은 이야기를 써줘.")
    return {"story": msg.content}

# [Worker C] 개그맨 노드
def write_joke(state: WriterState):
    topic = state["topic"]
    print(f"   [Worker C] '{topic}' 주제로 농담(Joke) 작성 시작...")
    msg = model.invoke(f"'{topic}'에 대한 재미있는 아재개그를 하나 해줘.")
    return {"joke": msg.content}

# [Aggregator] 편집장 노드 (모든 결과를 취합)
def aggregator(state: WriterState):
    print("\n--- [Aggregator] 모든 원고 도착! 최종 편집 중 ---")

    # State에 안전하게 저장된 각 결과물을 꺼내서 보기 좋게 합침
    final_text = f"""
[주제: {state['topic']} 종합 선물세트]

1. 시 (Poem)
----------------
{state['poem']}

2. 소설 (Story)
----------------
{state['story']}

3. 농담 (Joke)
----------------
{state['joke']}
    """
    return {"final_report": final_text}

# 4. Make Graph
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(WriterState)

# 노드 추가
workflow.add_node("write_poem", write_poem)
workflow.add_node("write_story", write_story)
workflow.add_node("write_joke", write_joke)
workflow.add_node("aggregator", aggregator)

# 1. 펼치기 (Fan-out): START에서 3개의 노드로 동시에 뻗어 나가기
workflow.add_edge(START, "write_poem")
workflow.add_edge(START, "write_story")
workflow.add_edge(START, "write_joke")

# 2. 모으기 (Fan-in): 3개의 노드가 모두 하나의 노드로 모이기
workflow.add_edge("write_poem", "aggregator")
workflow.add_edge("write_story", "aggregator")
workflow.add_edge("write_joke", "aggregator")

# 종료 연결 및 컴파일
workflow.add_edge("aggregator", END)
app = workflow.compile()

import time

start_time = time.time()
inputs = {"topic": "직장인의 월요일"}

# 에이전트 실행
result = app.invoke(inputs)

end_time = time.time()
print(f"총 소요 시간: {end_time - start_time:.2f}초")
print(result["final_report"])
