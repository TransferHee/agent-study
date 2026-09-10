from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

# 1. 뇌(LLM) 준비
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

# 2. 손발(도구) 정의
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.
    Args:
        a: First int
        b: Second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.
    Args:
        a: First int
        b: Second int
    """
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.
    Args:
        a: First int
        b: Second int
    """
    return a / b

# 도구들을 리스트로 묶기
tools = [add, multiply, divide]

model_with_tools = model.bind_tools(tools)

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    # 대화 기록은 리스트 형태로 '누적'되도록 add_messages 리듀서 적용
    messages: Annotated[list[AnyMessage], add_messages]
    # 모델 호출 횟수 기록용
    llm_calls: int

from langchain.messages import SystemMessage

def llm_call(state: State):
    """LLM이 현재 상태를 보고 답변하거나, 도구 사용을 요청하는 작업자"""

    # 시스템 메시지를 맨 앞에 추가하고 기존 대화 기록을 뒤에 붙여서 LLM에게 전송
    response = model_with_tools.invoke(
        [
            SystemMessage(
                content="당신은 사칙연산을 완벽하게 해내는 유능한 Agent입니다."
            )
        ] + state["messages"]
    )

    # 작업이 끝나면, 새로 생성된 메시지 1개와 카운터 1 증가분을 공책에 업데이트
    return {
        "messages": [response],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

from langchain.messages import ToolMessage

# 도구 이름을 키로, 실제 함수를 값으로 가지는 딕셔너리 준비
tools_by_name = {tool.name: tool for tool in tools}

def tool_node(state: State):
    """LLM이 도구 사용을 요청했을 때, 실제로 함수를 실행하는 작업자"""

    result = []
    # 공책의 맨 마지막 메시지(LLM의 요청장)에서 tool_calls 리스트 꺼내기
    last_message = state["messages"][-1]

    for tool_call in last_message.tool_calls:
        # 1. 도구 이름으로 실제 파이썬 함수 찾기
        tool = tools_by_name[tool_call["name"]]

        # 2. 인자(args)를 넣고 함수 실행
        tool_result = tool.invoke(tool_call["args"])

        # 3. 실행 결과를 ToolMessage로 포장 (tool_call_id를 반드시 맞춰주어야 모델이 인식함)
        result.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))

    # 도구 실행 결과들을 공책에 업데이트
    return {"messages": result}

from langgraph.graph import StateGraph, START

# 1. 빈 작업장(그래프) 도면 펼치기 (공책 양식을 알려줌)
agent_builder = StateGraph(State)

# 2. add_node: 도면에 작업자(노드) 배치하기
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# 3. add_edge: 고정된 연결선(Edge) 긋기
# 시작(START)하자마자 무조건 llm_call 작업자에게 최초로 공책을 넘깁니다.
agent_builder.add_edge(START, "llm_call")
# 도구 실행이 끝나면 결과값을 얻었으니 무조건 다시 llm_call 작업자에게 공책을 넘겨 최종 답변 생성
agent_builder.add_edge("tool_node", "llm_call")

from langgraph.graph import END

def should_continue(state: State):
    """
    LLM의 응답을 보고 다음 단계로 어디를 갈지 결정하는 라우팅 함수
    """
    last_message = state["messages"][-1]

    # LLM이 도구 호출(tool_calls) 요청장을 작성했다면? -> 도구 작업자(tool_node)에게 가라고 문자열 반환
    if last_message.tool_calls:
        return "tool_node"

    # 도구 호출이 없고 최종 답변을 작성했다면? -> 작업 종료(END) 객체 반환
    return END

# 4. add_conditional_edges: 라우팅 기능을 하는 선 설치
agent_builder.add_conditional_edges(
    "llm_call",         # 1) 출발지: llm_call 작업이 끝나면 무조건 이 라우터 함수가 발동합니다.
    should_continue,    # 2) 라우팅 함수: 위에서 만든 함수가 공책을 보고 어디로 갈지 판단합니다.
    ["tool_node", END]  # 3) 도착지 목록: 갈 수 있는 목적지들을 리스트나 딕셔너리로 명시합니다.
)

agent = agent_builder.compile()

from langchain.messages import HumanMessage

# 단순한 질문 테스트
messages = [HumanMessage(content="3과 4를 더해줘")]
response1 = agent.invoke({"messages": messages})

print("첫 번째 응답:", response1["messages"][-1].content)

'''
1) OpenAI (GPT 계열)의 경우

도구를 호출할 때(내용이 없을 때): content='' (빈 문자열)
최종 답변을 낼 때: content='3과 4를 더하면 7입니다.' (단순 문자열)
2) Google Gemini 계열의 경우

도구를 호출할 때(내용이 없을 때): content=[] (빈 리스트)
최종 답변을 낼 때: content=[{'type': 'text', 'text': '3과 4를 더하면 7입니다.'}] (리스트 안의 딕셔너리)
왜 이런 차이가 발생할까요? 
그 이유는 Gemini는 텍스트뿐만 아니라 이미지, 오디오 등 다양한 멀티모달(Multi-modal) 출력을 하나의 규격으로 안전하게 담기 위해, 
텍스트만 있는 상황에서도 기본적으로 '리스트 형태'로 데이터를 반환하는 경우가 많습니다. 
반면, OpenAI는 순수 텍스트 결과일 때 우리가 익숙한 일반 문자열(String) 형태로 반환하죠.
'''