from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-mini", temperature=0.2)

from typing import List, Annotated
import operator
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# 1. 데이터 모델 정의 (Plan & Section)
# 팀장(Orchestrator)이 짤 계획표의 양식입니다.
class Section(BaseModel):
    name: str = Field(description="목차(섹션)의 제목")
    description: str = Field(description="이 섹션에서 다뤄야 할 핵심 내용")

class Plan(BaseModel):
    sections: List[Section] = Field(description="보고서 작성을 위한 목차 리스트")

# 구조화된 출력으로 팀장 LLM이 반드시 Plan 양식을 지키도록 강제합니다.
planner_llm = model.with_structured_output(Plan)

# 2. 공용 State 정의 (팀장과 편집자가 보는 전체 상황판)
class ReportState(TypedDict):
    topic: str                      # 주제
    sections: List[Section]         # 팀장이 짠 계획(목차)

    # [핵심] 여러 Worker가 동시에 결과를 써도 덮어쓰지 않고 '합쳐지도록(Append)' 설정
    # operator.add는 리스트끼리 더하기(+) 연산을 수행합니다.
    completed_sections: Annotated[List[str], operator.add]

    final_report: str               # 최종 결과물

# 3. Worker용 State 정의 (알바생이 받는 개별 작업 지시서)
class WorkerState(TypedDict):
    section: Section

# [Orchestrator] 팀장 노드: 계획 수립
def orchestrator_node(state: ReportState):
    topic = state["topic"]
    print(f"\n--- [Orchestrator] '{topic}' 보고서 계획 수립 중 ---")

    # 보고서 목차를 생성 (비용과 속도를 위해 최대 3개로 제한합니다)
    plan = planner_llm.invoke(f"'{topic}'에 대한 보고서 목차를 짜줘. 3개 섹션 이내로 구성해.")

    print(f"   ㄴ 생성된 계획: {[s.name for s in plan.sections]}")
    return {"sections": plan.sections}

# [Worker] 팀원 노드: 섹션 집필 (이 노드는 목차 개수만큼 복제되어 동시에 실행됩니다)
def worker_node(state: WorkerState):
    # 알바생은 전체 흐름(ReportState)을 모른 채, 자신에게 할당된 section 정보만 받습니다.
    section = state["section"]
    print(f"   --- [Worker] 집필 중: {section.name} ---")

    prompt = f"""
    다음 섹션에 대한 내용을 짧게 작성해줘.
    제목: {section.name}
    내용 가이드: {section.description}
    """
    msg = model.invoke(prompt)

    # 제목과 내용을 포맷팅해서 반환합니다. 
    # 반환된 리스트는 operator.add에 의해 공용 State의 completed_sections에 누적됩니다.
    content = f"## {section.name}\n{msg.content}\n"
    return {"completed_sections": [content]}

# [Synthesizer] 편집자 노드: 취합
def aggregator_node(state: ReportState):
    print("\n--- [Aggregator] 모든 원고 취합 및 최종 편집 ---")

    # 리스트에 모인 조각글들을 하나로 합칩니다.
    completed = state["completed_sections"]
    final_report = "\n".join(completed)

    return {"final_report": final_report}

# [Synthesizer] 편집자 노드: 단순 취합이 아니라 '화학적 결합'을 수행
def synthesizer_node(state: ReportState):
    topic = state["topic"]
    completed_docs = state["completed_sections"]

    print(f"\n--- [Synthesizer] 원고 {len(completed_docs)}건 도착. 최종 편집 시작 ---")

    # 1. 일단 텍스트 덩어리로 병합
    raw_content = "\n\n".join(completed_docs)

    # 2. LLM에게 '전문 편집자' 역할 부여
    prompt = f"""
    당신은 전문 리포트 편집자입니다.
    다음은 '{topic}'에 대해 여러 작가가 나누어 쓴 원고들입니다.

    이 초안들을 바탕으로 **하나의 자연스럽고 전문적인 보고서**로 다시 작성해주세요.

    [지시사항]
    1. 각 섹션의 연결이 매끄러워야 합니다.
    2. 전체를 아우르는 '서론'과 '결론'을 추가해주세요.
    3. 마크다운(Markdown) 형식을 사용하여 가독성을 높여주세요.

    [원고 내용]
    {raw_content}
    """

    # 최종 생성을 위한 LLM 호출
    msg = model.invoke(prompt)

    return {"final_report": msg.content}

from langgraph.types import Send

# 동적 라우팅 로직 (Map: 1 -> N)
def assign_workers(state: ReportState):
    sections = state["sections"]

    # [핵심] Send API 사용
    return [Send("worker_node", {"section": s}) for s in sections]

from langgraph.graph import StateGraph, START, END

workflow = StateGraph(ReportState)

workflow.add_node("orchestrator_node", orchestrator_node)
workflow.add_node("worker_node", worker_node)
workflow.add_node("synthesizer_node", synthesizer_node)

# 1. 시작 -> 팀장
workflow.add_edge(START, "orchestrator_node")

# 2. 팀장 -> (Map) -> 팀원들 (동적 분배)
workflow.add_conditional_edges(
    "orchestrator_node",
    assign_workers,
    ["worker_node"] # 도착할 노드 이름을 명시적으로 적어줍니다.
)

# 3. 팀원들 -> (Reduce) -> 편집자
# 모든 worker_node의 실행이 끝나면 알아서 공유 상태를 들고 synthesizer_node로 모입니다.
workflow.add_edge("worker_node", "synthesizer_node")

# 4. 편집자 -> 종료
workflow.add_edge("synthesizer_node", END)

app = workflow.compile()

inputs = {"topic": "생성형 AI의 미래"}
result = app.invoke(inputs)

print(result['final_report'])