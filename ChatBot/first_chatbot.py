from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id=os.getenv("HUGGINGFACE_MODEL_REPO"),
    task="text-generation",
    max_new_tokens=100,
    temperature=0.5
)

chat_model=ChatHuggingFace(llm=llm)

chat_history=[
    SystemMessage(content="You are helpful assistent.")
]

while True:
    user_input=input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input=="exit":
        break
    response=chat_model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("AI: ", response.content)

    
for x in chat_history:
    print(x.content)
