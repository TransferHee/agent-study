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

'''
원칙 1: 꼭 필요한 것만 저장하라
State에 새로운 필드를 추가할 때는 "이 정보가 다음 작업자(노드)에게 반드시 필요한가?"를 스스로에게 질문해야 합니다.

원칙 2: 가공하지 않은 '날것(Raw Data)'을 저장하라
가장 많이 하는 실수 중 하나는 State에 사람이 읽기 좋은 형태의 '문장'을 저장하는 것입니다. State에는 반드시 딕셔너리, 리스트, 숫자 등 컴퓨터가 바로 처리할 수 있는 형태(Raw Data)를 유지해야 합니다.
'''