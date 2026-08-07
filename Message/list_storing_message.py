from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id=os.getenv("HUGGINGFACE_MODEL_REPO"),
    task="text-generation",
    max_new_tokens=100,
    temperature=0.7,
)

# Wrap with chat model
chat_model = ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about LangChain')
]

result = chat_model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)
