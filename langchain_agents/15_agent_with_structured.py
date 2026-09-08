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

from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

from pydantic import BaseModel, Field
from typing import Literal

class EmailAnalysis(BaseModel):
    """이메일 내용을 분석한 결과 구조."""

    intent: Literal["complaint", "inquiry", "confirmation", "other"] = Field(
        description="이메일의 주요 의도 (complaint=불만, inquiry=문의, confirmation=확인, other=기타)"
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="고객의 감정 상태"
    )
    summary: str = Field(
        description="이메일의 핵심 내용을 1~2문장으로 짧게 요약"
    )
    next_action: str = Field(
        description="담당자가 취해야 할 다음 단계 (예: 환불 부서 이관, 매뉴얼 링크 발송 등)"
    )

from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-nano")
tools = [send_email_tool, read_email_tool]

agent = create_agent(
    model=model,
    tools=tools,
    # 핵심: 최종 산출물의 규격을 EmailAnalysis 스키마로 강제합니다.
    response_format=ToolStrategy(EmailAnalysis),
    middleware=[
        # 이메일 도구가 실제 백엔드에 없더라도 가상으로 동작하게 해주는 실습용 미들웨어
        LLMToolEmulator(model="gpt-5-nano"),
    ],
)

# 사용자 지시사항 전달 및 에이전트 실행
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "최근 온 메일을 읽고, 고객의 의도와 감정, 요약, 그리고 필요한 다음 조치를 분석해줘.",
            }
        ]
    }
)

# messages 배열을 뒤지는 대신, structured_response 키로 바로 접근합니다.
analysis = response["structured_response"]
data_for_db = analysis.model_dump()
print(analysis)
print(data_for_db)