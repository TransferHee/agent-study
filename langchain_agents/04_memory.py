from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="gpt-5-mini",
    reasoning_effort="medium",
    temperature=0.0,
)

from langchain.messages import HumanMessage, AIMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="제 이름은 jay이고 나이는 20살입니다."),
    AIMessage(content="저는 jay님을 도와드릴 수 있습니다. 무엇을 도와드릴까요?"),
    HumanMessage(content="저는 처음 오셨습니다. 저에 대해 알려주세요."),
]

from langchain_core.messages.utils import trim_messages, count_tokens_approximately

trimmed_messages = trim_messages(
    messages,
    strategy="last", # 오래된 대화를 버리고 최근 대화를 유지
    token_counter=count_tokens_approximately,
    max_tokens=2000,
    include_system=True, # 시스템 메시지는 잘려나가지 않도록 고정(핀)
    start_on="human", # 잘라냈을 때 첫 시작이 무조건 사람의 질문이 되도록 보장
    end_on=("human", "tool"),
)

response = model.invoke(trimmed_messages)

print(response.content)