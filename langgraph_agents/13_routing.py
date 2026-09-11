from dotenv import load_dotenv
load_dotenv()

# 1. 모델 정의 및 출력 정의
from langchain.chat_models import init_chat_model
from typing import Literal
from pydantic import BaseModel, Field

model = init_chat_model("gpt-5-mini", temperature=0)

# 라우터가 반드시 지켜야 할 출력 규격(Schema)을 정의합니다.
class RouteDecision(BaseModel):
    category: Literal["billing", "technical", "shipping", "general"] = Field(
        description="고객 문의 내용을 분석하여 적절한 담당 부서를 선택하세요."
    )

# 일반 모델에 규격을 씌워, 응답이 항상 RouteDecision 객체로 나오도록 만듭니다.
router_llm = model.with_structured_output(RouteDecision)

# 2. Define State
from typing import TypedDict

class SupportState(TypedDict):
    query: str          # 고객의 초기 질문
    category: str       # 라우터가 분류한 카테고리 (분기점의 핵심 키)
    response: str       # 최종적으로 전문가가 작성한 답변

# 3. Define Node
# [Manager] 라우터 노드
def router_node(state: SupportState):
    query = state["query"]
    print(f"\n--- [Router] 문의 분석 중: '{query}' ---")

    # 구조화된 LLM이 문맥을 읽고 정확히 4가지 중 하나의 카테고리를 결정합니다.
    decision = router_llm.invoke(query)

    print(f"   -> 분류 결과: {decision.category.upper()} 부서로 연결합니다.")

    # 판단 결과를 State의 category 칸에 덮어씁니다.
    return {"category": decision.category}

# [Expert A] 결제/환불 담당
def billing_expert(state: SupportState):
    print("--- [Billing Expert] 결제 전문가가 답변 작성 중 ---")
    prompt = f"당신은 결제/환불 전문가입니다. 다음 문의에 대해 정중하게 3문장 이내로 핵심만 답변하세요: {state['query']}"
    msg = model.invoke(prompt)
    return {"response": msg.content}

# [Expert B] 기술 지원 담당
def technical_expert(state: SupportState):
    print("--- [Tech Expert] 기술 지원 엔지니어가 분석 중 ---")
    prompt = f"당신은 IT 엔지니어입니다. 다음 기술 문제에 대해 해결책을 3문장 이내로 제시하세요: {state['query']}"
    msg = model.invoke(prompt)
    return {"response": msg.content}

# [Expert C] 배송 담당
def shipping_expert(state: SupportState):
    print("--- [Shipping Expert] 물류 담당자가 배송 조회 중 ---")
    prompt = f"당신은 배송 관리자입니다. 다음 배송 문의에 대해 3문장 이내로 명확히 답변하세요: {state['query']}"
    msg = model.invoke(prompt)
    return {"response": msg.content}

# [Expert D] 일반 상담
def general_expert(state: SupportState):
    print("--- [General] 일반 상담원이 답변 중 ---")
    msg = model.invoke(f"다음 문의에 친절하게 3문장 이내로 답변하세요: {state['query']}")
    return {"response": msg.content}

# 4. Make Graph
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(SupportState)

# 노드 등록
workflow.add_node("router_node", router_node)
workflow.add_node("billing_expert", billing_expert)
workflow.add_node("technical_expert", technical_expert)
workflow.add_node("shipping_expert", shipping_expert)
workflow.add_node("general_expert", general_expert)

# 시작점에서는 항상 라우터 노드로 먼저 진입합니다.
workflow.add_edge(START, "router_node")

# 조건부 엣지를 위한 라우팅 함수
def route_to_expert(state: SupportState):
    # 라우터 노드가 적어둔 카테고리 값을 읽습니다.
    category = state["category"]

    if category == "billing":
        return "billing_expert"
    elif category == "technical":
        return "technical_expert"
    elif category == "shipping":
        return "shipping_expert"
    else:
        return "general_expert"

# 조건부 엣지 연결
workflow.add_conditional_edges(
    "router_node",       # 분기가 시작되는 출발지 노드
    route_to_expert,     # 방향을 결정할 파이썬 함수
    {
        # 함수의 반환값 : 실제 이동할 노드 이름
        "billing_expert": "billing_expert",
        "technical_expert": "technical_expert",
        "shipping_expert": "shipping_expert",
        "general_expert": "general_expert"
    }
)

# 어떤 전문가 노드를 거치든, 답변 작성이 끝나면 그래프를 종료(END)합니다.
workflow.add_edge("billing_expert", END)
workflow.add_edge("technical_expert", END)
workflow.add_edge("shipping_expert", END)
workflow.add_edge("general_expert", END)

app = workflow.compile()

# 테스트 1: 결제 문의
print(app.invoke({"query": "지난달 요금이 두 번 빠져나갔어요. 확인해주세요."}))

# 테스트 2: 기술 문의
print(app.invoke({"query": "API 연결할 때 404 에러가 자꾸 떠요."}))

# 테스트 3: 배송 문의
print(app.invoke({"query": "주문한 노트북 언제 도착하나요?"}))
