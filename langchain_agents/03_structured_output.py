from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="gpt-5-mini",
    reasoning_effort="medium",
    temperature=0.0,
)

from pydantic import BaseModel, Field
from typing import Optional, Literal

class Movie(BaseModel):
    """상세한 영화 정보."""
    title: str = Field(description="영화의 제목 (예: 인셉션)")
    year: Optional[int] = Field(default=None, description="개봉 연도. 정보를 알 수 없다면 None.")
    genre: Literal["액션", "로맨스", "SF", "코미디", "기타"] = Field(description="영화의 장르")
    director: str = Field(description="영화 감독 이름")
    rating: float = Field(description="영화 평점 (10점 만점 기준)")

model_with_structure = model.with_structured_output(Movie)

# 2. 자연어 프롬프트로 호출
response = model_with_structure.invoke("영화 인셉션에 대해 설명해 주세요")

# 3. 결과 확인
print(response, type(response))

import json

json_schema = {
    "title": "Movie",
    "description": "A movie with details",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "The title of the movie"
        },
        "year": {
            "type": "integer",
            "description": "The year the movie was released"
        },
        "director": {
            "type": "string",
            "description": "The director of the movie"
        },
        "rating": {
            "type": "number",
            "description": "The movie's rating out of 10"
        }
    },
    "required": ["title", "director", "rating"]
}

model_with_json_schema = model.with_structured_output(json_schema)

response = model_with_json_schema.invoke("영화 인셉션에 대해 설명해 주세요")

print(response, type(response))

print(json.dumps(response, indent=2, ensure_ascii=False))