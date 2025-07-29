import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import streamlit as st

load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER")

st.title("File Upload to Azure Blob Storage")
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER)
        blob_client = container_client.get_blob_client(uploaded_file.name)

        # Upload the file to Azure Blob Storage
        blob_client.upload_blob(uploaded_file, overwrite=True)

        st.success("File uploaded successfully.")
    except Exception as e:
        st.error(f"An error occurred: {e}")