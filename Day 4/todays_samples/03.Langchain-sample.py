import openai
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import os

# Load environment variables from .env file
load_dotenv()

print("Azure OpenAI API Key:", os.getenv('OPENAI_API_KEY'))
print("Azure OpenAI Endpoint:", os.getenv('AZURE_ENDPOINT'))
print("Azure OpenAI API Type:", os.getenv('OPENAI_API_TYPE'))
print("Azure OpenAI API Version:", os.getenv('OPENAI_API_VERSION'))


model = AzureChatOpenAI(model="dev-gpt-4.1-mini")

response = model.invoke("삼성전자의 파운드리 사업에 대해서 알려줘")
print(response.content)