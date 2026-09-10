from dotenv import load_dotenv
load_dotenv()

from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents="랭체인과 랭그래프의 관계를 설명해줘"
)

print(response.text)
