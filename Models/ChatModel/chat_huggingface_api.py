from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id=os.getenv("HUGGINGFACE_MODEL_REPO"),
    task="text-generation"
)


chat_model=ChatHuggingFace(llm=llm)

print(chat_model)

result=chat_model.invoke("what is the capital of india")
print(result.content) 