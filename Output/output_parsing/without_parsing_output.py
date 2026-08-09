from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

template1=PromptTemplate(
    template="Write a detailed report on {topic}.",
    input_variable=["topic"]
)

template2=PromptTemplate(
    template="Write a 5 line summary from {text}",
    input_variable=["text"]
)

prompt1=template1.invoke({"topic": input("Enter the topic: ")})

llm=HuggingFaceEndpoint(
    repo_id=os.getenv("HUGGINGFACE_MODEL_REPO"),
    task="text-generation",
    max_new_tokens=300,
    temperature=0.8
)

chat_model=ChatHuggingFace(llm=llm)

result1=chat_model.invoke(prompt1)

prompt2=template2.invoke({"text": result1.content})

result2=chat_model.invoke(prompt2)
print(result2.content)



