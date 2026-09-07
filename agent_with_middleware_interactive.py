from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="gpt-5-mini",
    reasoning_effort="medium",
    temperature=0.0,
)

from langchain.tools import tool
from typing import Dict, List

@tool
def send_email_tool(to: str, subject: str, body: str) -> str:
    """지정한 주소로 이메일을 보내는 도구(프로토타입)."""
    return f"Email sent. to={to}, subject={subject}"

@tool
def read_email_tool(limit: int = 3) -> List[Dict[str, str]]:
    """최근 받은 이메일을 조회하는 도구(프로토타입)."""
    return [{"from": "hr@example.com", "subject": "정책 안내", "body": "..."}][:limit]

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, LLMToolEmulator
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

checkpointer = InMemorySaver()

agent = create_agent(
    model,
    tools=[send_email_tool, read_email_tool],
    checkpointer=checkpointer,
    middleware=[
        LLMToolEmulator(model="gpt-5-nano"),
        HumanInTheLoopMiddleware(
            interrupt_on={
                # 이메일 전송은 부작용이 크므로 승인/수정/거절 옵션을 활성화
                "send_email_tool": {"allowed_decisions": ["approve", "edit", "reject"]},
                # 이메일 읽기는 단순 조회이므로 중단 없이 바로 실행 허용
                "read_email_tool": False,
            }
        ),
    ],
)

cfg = {"configurable": {"thread_id": "HIL-interactive"}}


def ask_human_decision(action_request: dict, review_config: dict) -> dict:
    """터미널에서 사용자에게 승인 여부를 물어보고 Decision dict를 반환합니다."""
    print("\n--- 사람의 승인이 필요한 도구 호출 ---")
    print(f"Tool: {action_request['name']}")
    print(f"Args: {action_request['args']}")
    print(f"설명: {action_request.get('description', '')}")
    allowed = review_config["allowed_decisions"]

    while True:
        choice = input(f"결정을 입력하세요 {allowed}: ").strip().lower()
        if choice in allowed:
            break
        print(f"'{choice}'는 허용되지 않는 결정입니다. 다시 입력하세요.")

    if choice == "approve":
        return {"type": "approve"}

    if choice == "reject":
        message = input("거절 사유(엔터 시 기본 메시지 사용): ").strip()
        decision: dict = {"type": "reject"}
        if message:
            decision["message"] = message
        return decision

    # choice == "edit"
    print(f"현재 args: {action_request['args']}")
    to = input("to (엔터 시 유지): ").strip() or action_request["args"].get("to", "")
    subject = input("subject (엔터 시 유지): ").strip() or action_request["args"].get("subject", "")
    body = input("body (엔터 시 유지): ").strip() or action_request["args"].get("body", "")
    return {
        "type": "edit",
        "edited_action": {
            "name": action_request["name"],
            "args": {"to": to, "subject": subject, "body": body},
        },
    }


def run_with_human_in_the_loop(user_input: str, config: dict) -> None:
    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]}, config)

    # interrupt가 걸려있는 동안 사람의 결정을 받아 재개(resume)합니다.
    while "__interrupt__" in result:
        hitl_request = result["__interrupt__"][0].value

        decisions = [
            ask_human_decision(action_request, review_config)
            for action_request, review_config in zip(
                hitl_request["action_requests"], hitl_request["review_configs"]
            )
        ]

        result = agent.invoke(Command(resume={"decisions": decisions}), config)

    print("\n[최종 응답]:", result["messages"][-1].content)


if __name__ == "__main__":
    run_with_human_in_the_loop(
        "온 메일 다 확인한 뒤 요약해 보고해. 그 다음 답장 작성해 회신 보내줘. 마지막으로 어떻게 보냈는지 보고해.",
        cfg,
    )
