import openai
import os
from dotenv import load_dotenv
import streamlit as st

# 환경 변수 로드
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.azure_endpoint = os.getenv("AZURE_ENDPOINT")
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# OpenAI API 클라이언트 설정
def get_openai_client(messages):
    # OpenAI API 호출 예시
    try:
        response = openai.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.4,
        )
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"OpenAI API 호출 중 오류 발생: {e}")
        return f"Error: {e}"

# Streamlit UI 설정
st.title("Azrue OpenAI Chat Interface")
st.write("Azure OpenAI API를 사용한 모델과 대화해 보세요 ^o^") 

# 채팅 기록의 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []

# 채팅 메시지 표시    
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# 사용자 입력 처리
if user_input := st.chat_input("메시지를 입력하세요"):
    # 사용자 메시지 저장 및 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # 모델 응답 생성 및 저장
    with st.spinner("응답을 기다리는 중..."):
        assistant_response = get_openai_client(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    st.chat_message("assistant").write(assistant_response)