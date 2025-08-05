import os
from dotenv import load_dotenv
from openai import AzureOpenAI

chat_client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION")
)

prompt = [
    {
        "role": "system",
        "content": "You are a travel assistant that provides information on travel service available from Margie's Travel."
    },
]

while True:
    input_text = input("Enter your question (or type 'exit' to quit): ")

    if input_text.lower() == 'exit':
        print("Exiting the chat.")
        break
    elif input_text.strip() == "":
        print("Please enter a valid question.")
        continue

    prompt.append({"role": "user", "content": input_text})

    rag_params = {
        "data_sources": [
            {
                # he following params are used to search the index
                "type": "azure_search",
                "parameters": {
                    "endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
                    "index_name": os.getenv("AZURE_SEARCH_INDEX_NAME"),
                    "authentication": {
                        "type": "api_key",
                        "key": os.getenv("AZURE_SEARCH_API_KEY"),
                    },
                    # The following params are used to vectorize the query
                    "query_type": "vector",
                    "embedding_dependency": {
                        "type": "deployment_name",
                        "deployment_name": os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
                    },
                }
            }
        ],
    }

    response = chat_client.chat.completions.create(
        model="dev-gpt-4.1-mini",
        messages=prompt,
        extra_body=rag_params
    )

    completion = response.choices[0].message.content
    print(f"AI: {completion}")

    prompt.append({"role": "assistant", "content": completion})
