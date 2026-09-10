from typing import TypedDict, Literal

# 1. LLM이 분류해 낼 '날것(Raw)' 형태의 딕셔너리 구조를 먼저 정의합니다.
class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str

# 2. 에이전트 전체가 공유할 State(공책) 양식을 정의합니다.
class EmailAgentState(TypedDict):
    # [입력 데이터] 다시 복구할 수 없는 원본 데이터
    email_content: str
    sender_email: str

    # [LLM 판단 결과] 여러 노드에서 조건문으로 쓸 깔끔한 딕셔너리 객체 (프롬프트 문장이 아님!)
    classification: EmailClassification | None

    # [비싼 연산 결과] 매번 다시 검색하면 비용이 드는 API/DB 호출 결과
    search_results: list[str] | None 
    customer_history: dict | None  

    # [최종 결과물] 루프를 돌며 수정되고 누적될 최종 답장과 대화 기록
    draft_response: str | None
    messages: list[str] | None
