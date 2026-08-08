from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os

load_dotenv()

class Review(TypedDict):
    summary: Annotated[str, "a breif about the topic"]
    sentiment: Annotated[str, "return the topic sentiment"]


llm=HuggingFaceEndpoint(
    repo_id=os.getenv("HUGGINGFACE_MODEL_REPO"),
    task="text-generation",
    max_new_tokens=100,
    temperature=0.7
)

chat_model=ChatHuggingFace(llm=llm)

structured_model=chat_model.with_structured_output(Review)

response=structured_model.invoke("""
    I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, its an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

    The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

    However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

    """)

print(response)
print(type(response))
print(response["summary"])
# print(response["sentiment"])

