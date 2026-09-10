from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

# 1. 모델 선언
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

# 2. Tool 정의
from langchain.tools import tool

@tool
def search_manual(query : str) -> str:
    """
    고객의 질문에 답변하기 위해 참고할만한 규정이나 메뉴얼을 검색할 때 사용하는 도구.
    """
    if '비밀번호' in query :
        return '비밀번호 변경은 마이페이지 - 보안 설정에 있음'
    elif '배송' in query :
        return '00택배에서 3일 내 배송 예정임'
    else :
        return '해당 내용 관련 매뉴얼은 찾을 수 없습니다.'

tools = [search_manual]
tools_by_name = {tool.name: tool for tool in tools}

# 도구를 사용할 수 있는 권한을 가진 실무자용 모델
model_with_tools = model.bind_tools(tools)

# 3. State 정의
from typing_extensions import TypedDict, Annotated
from langchain.messages import AnyMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_step: str 

# 4. Node 정의
# [Node 1] 분류 노드 (Manager)
# LLM에게 이메일을 읽고 어떤 트랙으로 보낼지 결정
def classify_node(state: AgentState):
    print("\n--- [1] 분류 단계 (LLM 판단) ---")
    last_message = state["messages"][-1]

    # 분류를 위한 프롬프트 (System Prompt)
    prompt = """
    당신은 고객 센터 관리자입니다. 고객의 이메일을 분석해서 다음 단계를 결정하세요.

    1. 단순 문의나 정보 요청이라면 -> 'consultant' 반환
    2. 환불 요청, 불만 제기, 화난 고객이라면 -> 'escalate' 반환

    답변은 오직 단어 하나만 하세요.
    """

    # LLM 호출
    response = model.invoke([SystemMessage(content=prompt), last_message])
    raw_content = response.content

    if isinstance(raw_content, list):
        # Gemini 등: 리스트로 오는 경우 텍스트 블록만 추출해서 합침
        decision = "".join([block['text'] for block in raw_content if block.get('type') == 'text'])
    else:
        # OpenAI 등: 문자열로 오는 경우 그대로 사용
        decision = str(raw_content)

    # 공백 제거 및 소문자 변환
    decision = decision.strip().lower()
    print(f"   -> LLM 판단 결과: {decision}")

    # 다음 단계(next_step)를 상태에 저장
    if "escalate" in decision:
        return {"next_step": "escalate"}
    else:
        return {"next_step": "consultant"}

# [Node 2] 상담 AI 노드 (Worker)
# 도구를 사용해서 답변을 생성하는 실제 에이전트
def consultant_node(state: AgentState):
    print("\n--- [2-A] 상담 AI 답변 생성 중 ---")
    response = model_with_tools.invoke(state['messages'])
    return {'messages' : [response]}

# [Node 3] 상담원 이관 노드
def escalate_node(state: AgentState):
    return {'messages' : [AIMessage(content='해당 메일은 전문 상담원에게 이관되었습니다.')]}

# [Node 4] Tool Node
def tool_node(state: AgentState):
    print("\n--- [Tool Node] 도구 직접 실행 ---")
    result = []
    # 가장 최근 메시지에서 도구 호출 요청 꺼내기
    last_message = state["messages"][-1]

    for tool_call in last_message.tool_calls:
        # 1. 도구 이름으로 실제 함수 찾기
        tool = tools_by_name[tool_call["name"]]

        # 2. 함수를 실행하여 결과를 얻기
        print(f"   -> 실행 중: {tool_call['name']}")
        tool_result = tool.invoke(tool_call["args"])

        # 3. 결과를 ToolMessage 형태로 포장 (tool_call_id 필수!)
        result.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))

    # 실행 결과를 대화 기록에 추가
    return {"messages": result}

# 5. Edge 정의
# 라우터 1: 분류 결과에 따른 분기
def route_after_classify(state: AgentState):
    if state['next_step'] == 'escalate' :
        return 'escalate'
    else :
        return 'consultant'

# 라우터 2: 도구 사용 여부 판단
def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    # LLM이 도구 호출(tool_calls)을 포함한 응답을 보낸 경우
    if last_message.tool_calls:
        return "tool_node"

    # 도구 호출이 없으면 종료
    return END

from langgraph.graph import StateGraph, START, END

# 5. 그래프 초기화 및 컴파일
agent_builder = StateGraph(AgentState)

# 노드 배치
agent_builder.add_node("classify_node", classify_node)
agent_builder.add_node("consultant_node", consultant_node)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("escalate_node", escalate_node)

# 엣지 연결
agent_builder.add_edge(START, "classify_node")

agent_builder.add_conditional_edges(
    'classify_node',
    route_after_classify,
    {
        'escalate' : 'escalate_node',
        'consultant' : 'consultant_node'
    }
)
agent_builder.add_conditional_edges(
    'consultant_node',
    should_continue,
    ['tool_node', END]
)

# 핵심: 도구 사용이 끝나면 무조건 다시 상담 AI에게 돌아가서 최종 답변을 만들게 합니다 (피드백 루프).
agent_builder.add_edge("tool_node", "consultant_node")
agent_builder.add_edge("escalate_node", END)

# 최종 컴파일
agent = agent_builder.compile()

# 6. 테스트
from langchain.messages import HumanMessage

inputs = {"messages": [HumanMessage(content="비밀번호 변경은 어디서 해?")]}
response = agent.invoke(inputs)

# print(response)
print(response["messages"][-1].content[-1]['text'])