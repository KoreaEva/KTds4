import os
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools import TavilySearchResults
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

from langchain.agents import AgentExecutor

@tool
def web_search(query: str) -> str:   
    """Search the web using Tavily."""

    tavilyRetriever = TavilySearchResults(
        max_results=5,  # 반환할 결과의 수
        search_depth="advanced",  # 검색 깊이: basic 또는 advanced
        include_answer=True,  # 결과에 직접적인 답변 포함
        include_raw_content=True,  # 페이지의 원시 콘텐츠 포함
        include_images=True,  # 결과에 이미지 포함
        # include_domains=[...],  # 특정 도메인으로 검색 제한
        # exclude_domains=[...],  # 특정 도메인 제외
        # name="...",  # 기본 도구 이름 덮어쓰기
        # description="...",  # 기본 도구 설명 덮어쓰기
        # args_schema=...,  # 기본 args_schema 덮어쓰기
    )

    
    return tavilyRetriever.invoke(query)

@tool
def hotel_search(query: str) -> str:
    """Search for Hotel information in PDF files."""
    retriever = AzureAISearchRetriever(
        content_key="chunk",      # 인덱스 내 컨텐츠 필드명
        top_k=3,                   # 반환할 검색 결과 개수
        index_name=os.getenv("AZURE_AI_SEARCH_INDEX_NAME"),  # Azure Search 인덱스 이름
    )

    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),  # 환경 변수명 확인
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context provided.
        Context: {context}
        Question: {question}"""
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 체인 실행
    result = chain.invoke("recommend a hotel in london")
    return result

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


tools = [web_search, hotel_search]  # 사용할 도구 목록

llm = AzureChatOpenAI(model="dev-gpt-4.1-mini",temperature=0)

# 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 사용자의 요청을 처리하는 AI Assistant입니다."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 에이전트 생성 (도구 호출)
agent = create_tool_calling_agent(llm, tools, prompt)

# 에이전트 실행기 생성
agent_executor = AgentExecutor(
    agent=agent,      # 도구 호출 에이전트
    tools=tools,      # 도구 목록
    verbose=True,     # 상세 로그 출력
    )

# 에이전트 실행
response = agent_executor.invoke(
    #{"input": "서울의 날씨는 어떤가요?"}
    {"input": "런던의 호텔을 추천해줘"}
)




