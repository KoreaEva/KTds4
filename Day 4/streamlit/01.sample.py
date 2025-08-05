import openai
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.azure_endpoint = os.getenv("AZURE_ENDPOINT")
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_version = os.getenv("OPENAI_API_VERSION")

# OpenAI API 호출 예시
response = openai.chat.completions.create(
    model="dev-gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "이순신 장군이 누구야?"}
    ]
)

# 응답 출력
print(response.choices[0].message.content)