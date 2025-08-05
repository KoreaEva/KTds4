import os
import openai
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# === Environment Variables ===
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_SERVICE_ENDPOINT", "https://winkey-openai-002.openai.azure.com")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "https://winkey-ai-search-001.search.windows.net")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY", "YOUR_SEARCH_API_KEY")
AZURE_INDEX_NAME = os.getenv("AZURE_INDEX_NAME", "pdf-index")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini")  # deployment or engine name for OpenAI

# Configure the openai package
openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
# For Azure OpenAI, you might need to set a version header or use a specific API version.
# Refer to: https://learn.microsoft.com/azure/cognitive-services/openai/reference for details.

# === Retrieve Documents Using Azure Cognitive Search ===
def retrieve_documents(query):
    credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_INDEX_NAME, credential=credential)
    results = search_client.search(query)
    documents = [doc for doc in results]
    return documents

# === Generate Answer Using OpenAI Library ===
def generate_answer(query, documents):
    # Combine document content into a context string
    context = "\n".join(doc.get("content", "") for doc in documents)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    response = openai.Completion.create(
        engine=AZURE_DEPLOYMENT_NAME,
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    answer = response.choices[0].text.strip() if response.choices else ""
    return answer

# === RAG Pipeline: Retrieve Documents and Generate Answer ===
def rag_pipeline(query):
    documents = retrieve_documents(query)
    if not documents:
        return "No documents found."
    answer = generate_answer(query, documents)
    return answer

if __name__ == "__main__":
    query = input("Enter your query: ")
    result = rag_pipeline(query)
    print("Answer:", result)