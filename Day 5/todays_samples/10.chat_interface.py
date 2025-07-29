import openai
from openai import OpenAI
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
DEPLOYMENT_NAME = "dev-gpt-4.1-mini"

print("OpenAI API Key:", openai.api_key)
print("Azure OpenAI Endpoint:", openai.azure_endpoint)
print("OpenAI API Type:", openai.api_type)
print("OpenAI API Version:", openai.api_version)


def get_openai_client(messages):
    try:
        response = openai.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.7,
        )

        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error connecting to OpenAI: {e}")
        return None
    

# Streamlit UI 설정
st.title("Chat Interface with OpenAI")
st.write("OpenAI 모델에게 무엇이든 물어보세요")

# 채팅 메시지의 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 채팅 메시지의 표시
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# 사용자 입력 받기
if prompt := st.chat_input("질문을 입력하세요:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # OpenAI에 메시지 전송 및 응답 받기
    response = get_openai_client(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)