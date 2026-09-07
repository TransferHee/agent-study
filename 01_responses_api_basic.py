from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.responses.create(
    model="gpt-5-mini",
    reasoning={"effort": "medium"},
    input="한국어로 짧게 자기소개를 해줘.",
)

print(response.output_text)
