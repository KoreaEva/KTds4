import streamlit as st
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

st.title("AI Poem Generator")

subject = st.text_input("주제를 입력하세요:")
content = st.text_area("시를 작성할 내용을 입력하세요: ")

button_click = st.button("Generate Poem")
if button_click:
    with st.spinner("Generating poem..."):
    
        messages = [
            {"role": "system", "content": "You are a AI poem."},
            {"role": "user", "content": f"주제: {subject}\n내용: {content}"},
            {"role": "user", "content": "시를 작성해줘"}
        ]

        # Create a chat completion request
        response = openai.chat.completions.create(
                        model="dev-gpt-4.1-mini",
                        temperature=1,  
                        messages=messages
                    )
        
        st.write(response.choices[0].message.content)
        st.success("Poem generated successfully!")