from typing import Callable
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="gpt-5-mini",
    reasoning_effort="medium",
    temperature=0.0,
)

@dataclass
class Context:
    user_name: str

from langchain.agents.middleware import before_model, after_model
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.messages import HumanMessage, SystemMessage, AIMessage

@before_model(can_jump_to=['end'])
def validate_input(state, runtime):
    last_messages = state['messages'][-1]
    if '개인정보' in last_messages.content:
        print('개인정보 탐지됨 - 응답 차단!')
        return {
            'messages': [AIMessage(content='이 요청은 처리할 수 없습니다.')],
            'jump_to': 'end' # 모델 호출 중단 후 에이전트 종류
        }
    print('정상 입력. 모델 호출 진행')
    return None

@wrap_model_call
def inject_user_name(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    print('request\n', request)

    user_name = request.runtime.context.user_name

    if user_name:
        system_message = f'사용자의 이름은 {user_name} 입니다'
        request = request.override(system_prompt=system_message)

    return handler(request)
    
@wrap_model_call
def dynamic_model_selector(request, handler):
    # 최근 사용자의 입력 메시지 추출
    last_message = request.messages[-1].content if request.messages else ''
    msg_len = len(last_message)

    if msg_len < 10:
        model_name = "gpt-5-nano"
    elif msg_len < 100:
        model_name = "gpt-5-mini"
    else:
        model_name = "gpt-5"

    print(f'메세지 길이: {msg_len}, 모델 선택: {model_name}')
    new_model = ChatOpenAI(
        model_name=model_name,
        reasoning_effort="medium",
        temperature=0.0,
    )
    request = request.override(model=new_model)
    return handler(request)


from langchain.agents import create_agent

agent = create_agent(
    model,
    tools=[],
    middleware=[
        validate_input,
        inject_user_name,
        dynamic_model_selector,
    ],
    context_schema=Context,
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "제 이름이 뭐죠? 알려주세요"}]},
    context=Context(user_name='Hudson'),
)

print(response)
print(response['messages'][-1].content)