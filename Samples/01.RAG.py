import os
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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
print(result)
