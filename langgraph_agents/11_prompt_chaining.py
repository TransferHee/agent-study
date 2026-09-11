from dotenv import load_dotenv
load_dotenv()

# 1. Init model
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-mini", temperature=1.0)

# 2. Define State
from typing import TypedDict

class JokeState(TypedDict):
    topic: str          # 초기 주제
    draft_joke: str     # 1단계: 초안
    improved_joke: str  # 2단계: 수정안
    final_joke: str     # 3단계: 최종 결과

# 3. Define Node
# [Step 1] 초안 생성 노드
def generate_joke(state: JokeState):
    topic = state["topic"]
    print(f"\n--- [1단계] '{topic}' 주제로 농담 초안 생성 중 ---")

    # 체인이 망가지지 않도록 '다른 말은 하지 말고'라는 제약을 줍니다.
    prompt = f"'{topic}'에 대한 짧고 재미있는 농담을 하나 만들어줘. 절대 인사말이나 부연 설명 없이 농담 하나만 텍스트로 출력해."
    msg = model.invoke(prompt)

    # 생성된 초안을 draft_joke 필드에 저장합니다.
    return {"draft_joke": msg.content}

# [Step 2] 윤색/수정 노드
def improve_joke(state: JokeState):
    # 1단계에서 작성해둔 초안을 꺼내옵니다.
    original_joke = state["draft_joke"]
    print(f"\n--- [2단계] 더 웃기게 수정 중 (아재개그 스타일) ---")

    prompt = f"""
    다음 농담을 보고, 더 썰렁하고 재미있는 '아재개그' 스타일로 개선해줘. 
    주의: 절대 인사말이나 부연 설명, 옵션을 주지 말고 완성된 농담 딱 하나만 출력해.
    원문: {original_joke}
    """
    msg = model.invoke(prompt)

    # 수정된 농담을 improved_joke 필드에 저장합니다.
    return {"improved_joke": msg.content}

# [Step 3] 최종 포장 노드
def polish_joke(state: JokeState):
    # 2단계에서 작성해둔 수정안을 꺼내옵니다.
    improved_joke = state["improved_joke"]
    print(f"\n--- [3단계] 이모지 추가 및 마무리 ---")

    prompt = f"""
    다음 농담에 적절한 이모지를 듬뿍 넣어서 SNS에 올리기 좋게 꾸며줘.
    주의: "여기 있습니다", "다음과 같이 준비했습니다" 같은 인사말이나 해시태그 목록을 절대 포함하지 마. 오직 이모지가 추가된 농담 본문만 출력해.
    농담: {improved_joke}
    """
    msg = model.invoke(prompt)

    # 최종 결과를 final_joke 필드에 저장합니다.
    return {"final_joke": msg.content}

# 4. Make Graph
from langgraph.graph import StateGraph, START, END

# 앞서 정의한 공책(State) 규격으로 그래프를 초기화합니다.
workflow = StateGraph(JokeState)

# 노드 등록 (이름표, 실제 함수)
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_node("polish_joke", polish_joke)

# 엣지 연결: 조건부 분기 없이 일직선으로 연결합니다 (Linear Chain).
# 흐름: START -> 1단계 -> 2단계 -> 3단계 -> END
workflow.add_edge(START, "generate_joke")
workflow.add_edge("generate_joke", "improve_joke")
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)

# 설계도를 실제 실행 가능한 체인으로 컴파일합니다.
chain = workflow.compile()

# 5. Test
inputs = {"topic": "고양이"}
result = chain.invoke(inputs)

print(result)