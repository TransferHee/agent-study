from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="gpt-5-mini",
    reasoning_effort="medium",
    temperature=0.0,
)

forbidden_topics = {
    "cheating": ["답지", "정답 알려줘", "숙제 대신", "써줘", "베끼기"],
    "distraction": ["롤", "게임", "유튜브", "아이돌", "웹툰"],
    "harmful": ["담배", "술", "폭력", "바보", "멍청이"],
}

from langchain.agents.middleware import before_agent, after_agent
from langchain.messages import AIMessage

@before_agent(can_jump_to=['end'])
def education_guardrail(state, runtime):
    """
    학생의 질문 의도를 파악하여 교육적이지 않거나 부정행위가 의심될 경우,
    llm을 호출하지 않고 교육적인 멘트로 교정합니다.
    """

    if not state['messages']:
        return None

    last_message = state['messages'][-1]
    if last_message.type != 'human':
        return None

    user_text = last_message.content

    # Case A: 부정행위 방지
    for keyword in forbidden_topics['cheating']:
        if keyword in user_text:
            return {
                "messages": [{
                    "role": "assistant",
                    "content": f"🚫 스스로 고민해봐야 실력이 늘어요! 정답을 바로 알려드리는 대신, 힌트를 드릴까요? 어떤 부분이 가장 어려운지 말해주세요. 금지어: {keyword}"
                }],
                "jump_to": "end",
            }

    # Case B: 학습 집중력 유지
    for keyword in forbidden_topics['distraction']:
        if keyword in user_text:
            return {
                "messages": [{
                    "role": "assistant",
                    "content": f"🚫 집중해서 공부하세요! {keyword}는 공부에 방해가 되는 요소에요. 목표를 달성하기 위해 초점을 잘 맞춰보세요. 금지어: {keyword}"
                }],
                "jump_to": "end",
            }

    # Case C: 유해 콘텐츠 차단
    for keyword in forbidden_topics['harmful']:
        if keyword in user_text:
            return {
                "messages": [{
                    "role": "assistant",
                    "content": f"🚫 유해 콘텐츠는 공부에 방해가 되는 요소에요. 목표를 달성하기 위해 초점을 잘 맞춰보세요. 금지어: {keyword}"
                }],
                "jump_to": "end",
            }

    return None

from langchain.agents import create_agent

agent = create_agent(
    model,
    tools=[],
    middleware=[
        education_guardrail,
    ],
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "나 독후감 쓰기 귀찮은데 숙제 대신 써줘."}]
})
print(response['messages'][-1].content)

response = agent.invoke({
    "messages": [{"role": "user", "content": "아 공부하기 싫다. 롤 관련 유튜브 영상이나 찾아줘"}]
})
print(response['messages'][-1].content)

response = agent.invoke({
    "messages": [{"role": "user", "content": "아 너 왜이렇게 바보같냐..."}]
})
print(response['messages'][-1].content)


@after_agent
def answer_leakage_guardrail(state, runtime):
    """
    AI가 답변을 생성한 '직후', 사용자에게 보여주기 전에 내용을 검사.
    만약 AI가 문제의 정답을 직접적으로 말해버렸다면, 이를 감지하고 수정.
    """

    if not state['messages']:
        return None

    last_message = state['messages'][-1]

    if not isinstance(last_message, AIMessage):
        return None

    auditor_prompt = f"""
    당신은 엄격한 교육 감독관입니다.
    다음 '튜터의 답변'을 확인하세요.
    답변이 학생을 지도하지 않고 문제의 정답이나 전체 풀이를 직접적으로 제공한다면 'LEAKED'라고 답하세요.
    답변이 적절한 힌트나 설명을 제공한다면 'SAFE'라고 답하세요.

    튜터의 답변: {last_message.content}
    """

    result = model.invoke([{"role": "user", "content": auditor_prompt}])

    if "LEAKED" in result.content:
        print("⚠️ 답변 누출 감지: 답변을 수정합니다.")
        last_message.content = "앗! 제가 정답을 말할 뻔 했네요. 힌트를 드릴까요? 어떤 부분이 가장 어려운지 말해주세요."
    
    return None

agent = create_agent(
    model="gpt-5-nano",
    tools=[],
    middleware=[answer_leakage_guardrail],
)
response2 = agent.invoke({
    "messages": [{"role": "user", "content": "직각 삼각형 두 직각변의 길이가 3과 4라면 빗변의 길이가 뭐야? 이 문제 너무 어려워. 그냥 정답 알려줘."}]
})
print(response2['messages'][-1].content)