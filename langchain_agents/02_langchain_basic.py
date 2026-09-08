from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

load_dotenv()

llm = ChatOpenAI(
    model="gpt-5-mini",
    reasoning_effort="medium",
    temperature=0.0,
)

response = llm.invoke("한국어로 짧게 자기소개를 해줘.")

print(response)

for chunk in llm.stream("AI Agent란 무엇인지 100자 이상으로 설명해 주세요"):
    # chunk는 AIMessageChunk 객체이므로 .content로 텍스트를 추출합니다.
    print(chunk.content, end="", flush=True)

inputs = [
    "과적합(Overfitting)이 뭔가요? 한 줄로 요약해줘.",
    "앵무새의 털 색상이 화려한 이유를 한 줄로 요약해줘.",
    "AI Agent의 핵심 특징 한 가지는?"
]

# 한 번의 호출로 3개의 질문을 병렬 처리
responses = llm.batch(inputs, temperature=1.0, config={'max_concurrency': 5})

for i, response in enumerate(responses):
    print(f"[{i+1}번 답변] {response.content}")
