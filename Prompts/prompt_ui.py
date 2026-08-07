from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub import whoami
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import streamlit as st
import os

# Load environment variables
load_dotenv()

# token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
# print(whoami(token=token))


st.title("Research Page")

# Create Hugging Face endpoint
llm = HuggingFaceEndpoint(
    repo_id=os.getenv("HUGGINGFACE_MODEL_REPO"),
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
)

# Wrap with chat model
chat_model = ChatHuggingFace(llm=llm)

# User input
query = st.text_input("Enter your query:")

# Button
if st.button("Generate Response"):
    if query:
        response = chat_model.invoke(query)
        st.write(response.content)
    else:
        st.warning("Please enter a query.")