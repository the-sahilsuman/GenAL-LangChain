from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os

load_dotenv()

llm=HuggingFacePipeline.from_model_id(
    repo_id=os.getenv("HUGGINGFACE_MODEL_REPO"),
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=200
    )
)

chat_model=ChatHuggingFace(llm=llm)

result=chat_model.invoke(input("Enter your message: "))
print(result.content)
