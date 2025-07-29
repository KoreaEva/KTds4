import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER")

try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER)
    blob_client = container_client.get_blob_client("lee.jpg")

    # File upload logic can be added here
    with open(r"C:\Users\young\Documents\KTds실습\lee.jpg", "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    print("File uploaded successfully.")
except Exception as e:
    print(f"An error occurred: {e}")
