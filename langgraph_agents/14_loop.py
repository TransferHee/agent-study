from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict, Literal
from pydantic import BaseModel, Field

# 1. 모델 정의: 창의적인 문구 작성을 위해 temperature를 약간 높입니다.
model = init_chat_model("gpt-5-nano", temperature=0.7)

# 2. State 정의 (덮어쓰기 방식)
class AdState(TypedDict):
    product_name: str       # 상품명 (불변)
    ad_copy: str            # 생성된 광고 문구 (최신본으로 덮어쓰기)
    feedback: str           # 팀장의 피드백 내용 (최신본으로 덮어쓰기)
    status: str             # PASS / FAIL
    iteration_count: int    # 무한 루프 방지용 카운터

# 3. 평가 결과 구조화 (Pydantic)
class EvaluationResult(BaseModel):
    status: Literal["pass", "fail"] = Field(description="기준 충족 여부")
    feedback: str = Field(description="탈락 시 구체적인 수정 지시사항")

# 팀장용 LLM은 반드시 정해진 양식(JSON)으로만 대답하도록 강제합니다.
evaluator_llm = model.with_structured_output(EvaluationResult)

# [Generator] 신입 카피라이터 노드
def copywriter_node(state: AdState):
    product = state["product_name"]
    feedback = state.get("feedback")
    count = state.get("iteration_count", 0)

    print(f"\n--- [Copywriter] 광고 문구 작성 중 (시도: {count + 1}) ---")

    if not feedback:
        # 첫 시도는 의도적으로 건조하게 작성하도록 유도하여 팀장의 Fail을 유발해 봅니다.
        prompt = f"'{product}'의 기능 위주로 인스타그램 홍보 문구를 건조하게 작성해줘. 홍보 문구만 답변하고 반드시 20자 이하로 작성하시오."
    else:
        # 두 번째 시도부터는 팀장의 피드백을 철저히 반영합니다.
        print(f"   ㄴ [팀장님 지시 반영]: {feedback}")
        prompt = f"""
        '{product}' 인스타그램 홍보 문구를 다시 작성해.

        <반드시 지켜야 할 수정 사항>
        {feedback}
        </반드시 지켜야 할 수정 사항>

        <작성 시 반드시 지켜야할 사항>
        홍보 문구만 답변하고 절대 50자 이하로 작성하시오.
        """

    msg = model.invoke(prompt)

    # 작성된 문구와 함께, 무한 루프 방지를 위해 시도 횟수를 1 증가시킵니다.
    return {"ad_copy": msg.content, "iteration_count": count + 1}

# [Evaluator] 깐깐한 마케팅 팀장 노드
def manager_node(state: AdState):
    ad_copy = state["ad_copy"]
    print(f"\n--- [Manager] 문구 검수 중 ---")
    print(f"   ㄴ 제출된 문구: {ad_copy}")

    prompt = f"""
    당신은 깐깐한 마케팅 팀장입니다. 신입 사원이 쓴 다음 광고 문구를 평가하세요:
    "{ad_copy}"

    <평가 기준>
    1. (정량) 해시태그(#)가 3개 이상 있어야 합니다.
    2. (정량) '할인' 또는 '특가'라는 단어가 포함되어야 합니다.
    3. (정성 - 중요!) 문구가 너무 설명문 같거나 딱딱하면 안 됩니다. 소비자의 감성을 자극하는 '활기차고 매력적인 톤'이어야 합니다.
    </평가 기준>

    위 3가지 기준 중 하나라도 부족하면 fail을 주세요.
    특히 3번(톤앤매너)이 부족하다면 "좀 더 감성적으로 쓰세요" 같이 100자 이내로 조언하세요.
    """

    # 구조화된 출력(JSON)으로 평가 결과를 받습니다.
    result = evaluator_llm.invoke(prompt)

    print(f"   ㄴ 판정: {result.status.upper()}")
    if result.status == "fail":
        print(f"   ㄴ 지적 사항: {result.feedback}")

    # 평가 결과만 State에 덮어씁니다. (팀장이 직접 글을 고치지 않음!)
    return {"status": result.status, "feedback": result.feedback}

from langgraph.graph import StateGraph, START, END

# 1. 루프 라우팅 로직 (무한 루프 방지 포함)
def route_submission(state: AdState):
    status = state["status"]
    count = state["iteration_count"]

    # 1순위: 합격하면 루프를 당당히 탈출합니다.
    if status == "pass":
        print("\n[System] 팀장님 승인 완료! 인스타그램에 업로드합니다.")
        return END

    # 2순위: 무한 루프를 막기 위한 핵심 탈출 조건 (안전장치)
    if count >= 3:
        print("\n[System] 3번이나 수정했는데도 불합격입니다... 강제 종료합니다.")
        return END

    # 3순위: 불합격이고 기회가 남았다면 다시 신입 사원에게 돌려보냅니다.
    return "copywriter_node"

# 2. 워크플로우 조립
workflow = StateGraph(AdState)

workflow.add_node("copywriter_node", copywriter_node)
workflow.add_node("manager_node", manager_node)

# 시작하면 무조건 카피라이터가 먼저 작성
workflow.add_edge(START, "copywriter_node")
# 작성이 끝나면 무조건 팀장에게 검수 받음
workflow.add_edge("copywriter_node", "manager_node")

# 팀장의 검수 결과에 따라 루프를 돌거나 종료(조건부 엣지)
workflow.add_conditional_edges(
    "manager_node",
    route_submission,
    {"copywriter_node": "copywriter_node", END: END}
)

app = workflow.compile()

inputs = {"product_name": "자율주행 자동차"}
result = app.invoke(inputs)
print(result)