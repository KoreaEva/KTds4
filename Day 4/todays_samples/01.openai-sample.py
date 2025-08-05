import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.azure_endpoint = os.getenv('AZURE_ENDPOINT')
openai.api_type = os.getenv('OPENAI_API_TYPE')
openai.api_version = os.getenv('OPENAI_API_VERSION')

while True:

    question = input("질문을 입력하세요: ")
    if question.lower() == 'exit':
        break

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question}
    ]

    response = openai.chat.completions.create(
                    model="dev-gpt-4.1-mini",  
                    messages=messages
                )

    print(response.choices[0].message.content)