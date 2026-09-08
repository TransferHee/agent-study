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

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

import re

def detect_api_key(content: str) -> list[dict[str, str | int]]:
    """'sk-'로 시작하는 32자리 문자열을 API 키로 간주하고 탐지합니다."""
    matches = []
    pattern = r"sk-[a-zA-Z0-9]{32}"
    for match in re.finditer(pattern, content):
        matches.append({
            "text": match.group(0),
            "start": match.start(), # 정보 시작점
            "end": match.end(),     # 정보 끝점
        })
    return matches

api_key_blocker = PIIMiddleware(
    pii_type="api_key",
    detector=detect_api_key,
    strategy="mask",
    apply_to_input=True,
)

agent = create_agent(
    model,
    tools=[send_email_tool, read_email_tool],
    middleware=[
        # 이메일은 완전히 가림 (redact)
        PIIMiddleware(pii_type="email", strategy="redact", apply_to_input=True),
        # 신용카드 번호는 마스킹 처리 (mask)
        PIIMiddleware(pii_type="credit_card", strategy="mask", apply_to_input=True),
        api_key_blocker,
    ],
)

prompt = "안녕하세요. 이메일은 user123@example.com 입니다. 제 카드번호는 1234123443214321 입니다. api_key = sk-qweqweqweqweqweqweqweqweqweqweqw"
response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

# 에이전트가 실제로 넘겨받은(마스킹된) 메시지 확인
print(response["messages"][0].content)
