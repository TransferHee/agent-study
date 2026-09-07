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
from dataclasses import dataclass

@dataclass
class Context:
    user_name: str

from langchain.agents.middleware import before_model, after_model

from typing import Callable
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.messages import HumanMessage, SystemMessage

@wrap_model_call
def inject_user_name(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    print(f"request : \n{request}")

    # 1. request 보따리 안의 책상(runtime)에서 신분증 정보 추출
    user_name = request.runtime.context.user_name

    if user_name:
        system_message = f"사용자의 이름은 {user_name}입니다"
        # 2. request.override()를 사용해 시스템 프롬프트 덮어쓰기 (모델에게 직접 떠먹여 주기)
        request = request.override(system_prompt=system_message)

    # 3. 조작된 request를 handler를 통해 모델로 전송
    return handler(request)

@wrap_model_call
def dynamic_model_selector(request, handler):
    # 최근 사용자의 입력 메시지 추출
    last_msg = request.messages[-1].content if request.messages else ""
    msg_len = len(last_msg)

    # 길이에 따라 모델 선택
    if msg_len < 10:
        model_name = "gpt-5-nano"
    elif msg_len < 30:
        model_name = "gpt-5-mini"
    else:
        model_name = "gpt-5"

    # request.model을 새로운 모델로 교체
    new_model = ChatOpenAI(model_name=model_name)
    new_request = request.override(model=new_model)

    # 수정된 요청으로 LLM 호출
    return handler(new_request)
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-5-mini",
    tools=[],
    middleware=[inject_user_name],
    context_schema=Context,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "제 이름이 뭐죠?"}]},
    context=Context(user_name="Jay Pak"),
)
print(result["messages"][-1].content) 