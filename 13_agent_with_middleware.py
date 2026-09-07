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
from langchain.agents.middleware import LLMToolEmulator, TodoListMiddleware

# agent = create_agent(
#     model,
#     tools=[send_email_tool, read_email_tool],
#     middleware=[
#         LLMToolEmulator(model="gpt-5-nano"),
#         TodoListMiddleware(), # TodoList 미들웨어를 추가합니다.
#     ],
# )

# result = agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "온 메일 다 확인한 뒤 요약해 보고해. 그 다음 답장 작성해 회신 보내줘. 마지막으로 어떻게 보냈는지 보고해.",
#             }
#         ]
#     }
# )
# print(result["messages"][-1].content)

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, LLMToolEmulator
from langgraph.checkpoint.memory import InMemorySaver

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

cfg = {"configurable": {"thread_id": "HIL-a"}}

# 안전한 도구(메일 조회) 호출 테스트
response = agent.invoke({"messages": [{"role": "user", "content": "무슨 메일 왔는지 확인해줘"}]}, cfg)
print(response["messages"][-1].content)
