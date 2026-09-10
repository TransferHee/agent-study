from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict

# 1. 모델 선언
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

# 2. State 정의
# add_messages 없이 단순 자료형만 선언하면, 값이 들어올 때마다 덮어쓰기(Override) 됩니다.
class AgentState(TypedDict):
    email_content: str   # 고객이 보낸 원문 이메일
    category: str        # inquiry(문의) / complaint(불만)
    next_step: str       # 다음 행동 (search_manual / escalate_to_human)
    response: str        # 중간/최종 결과물

# 3. Node 정의
def read_email(state: AgentState):
    return {'email_content': state['email_content']}

def classify_intent(state: AgentState):
    print('\n[2] Analysis email content...')

    email = state['email_content']

    if '환불' in email or '빨리' in email:
        category = 'complaint'
        next_step = 'escalate_to_human' # 상담원 연결
    else:
        category = 'inquiry'
        next_step = 'search_manual' # 메뉴얼 검색
    
    return {'category': category, 'next_step': next_step}

def search_manual(state: AgentState):
    print('3-A 진입... 메뉴얼을 검색합니다.')
    return

def escalate_to_human(state : AgentState):
    print('3-B 진입... 상담원 이관합니다.')
    return {'response' : '불편을 드려 죄송합니다.. 상담원에게 이관하였으니 잠시 대기해 주시기 바랍니다.'}

def write_reply(state : AgentState):
    email = state['email_content']
    response = model.invoke(email)
    return {'response' : response}

# 3. conditional edge
def route_email(state : AgentState):
    if state['next_step'] == 'escalate_to_human' :
        return 'escalate_to_human'
    else :
        return 'search_manual'

# 4. 그래프 생성 및 컴파일
from langgraph.graph import StateGraph, START, END

agent_builder = StateGraph(AgentState)

# 노드 배치
agent_builder.add_node("read_email", read_email)
agent_builder.add_node("classify_intent", classify_intent)
agent_builder.add_node("search_manual", search_manual)
agent_builder.add_node("escalate_to_human", escalate_to_human)
agent_builder.add_node("write_reply", write_reply)

# 엣지 연결
agent_builder.add_edge(START, "read_email")
agent_builder.add_edge("read_email", "classify_intent")

agent_builder.add_conditional_edges(
    'classify_intent',
    route_email,
    ['escalate_to_human', 'search_manual']
)

agent_builder.add_edge("search_manual", "write_reply")
agent_builder.add_edge("write_reply", END) # 답변 작성이 끝나면 프로세스 종료
# escalate_to_human은 목적지가 없으므로, 따로 엣지를 잇지 않아도 암묵적으로 END로 향합니다.

# 컴파일 (설계도를 작동 가능한 애플리케이션으로 빌드)
agent = agent_builder.compile()

# 5. 테스트
# 5-1. 단순 문의 테스트 (매뉴얼 검색 트랙)
inputs = {"email_content": "비밀번호 변경 방법을 알려주세요."}
response = agent.invoke(inputs)
print(response)
print()

# 5-2. 불만 이메일 테스트 (상담원 이관 트랙)
inputs = {"email_content": "당장 환불해줘!"}
response = agent.invoke(inputs)
print(response)
print()
