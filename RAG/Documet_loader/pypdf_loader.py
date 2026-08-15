from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

parser=StrOutputParser()

prompt=PromptTemplate(
    template="write a 5 line short note of this {doc}",
    input_variables=["doc"]
)

llm=HuggingFaceEndpoint(
    repo_id=os.getenv("HUGGINGFACE_MODEL_REPO"),
    task="text-generation",
    max_new_tokens=300,
    temperature=0.8
)

model=ChatHuggingFace(llm=llm)

loader=PyPDFLoader("handbook_graph.pdf")

doc=loader.load()

chain= prompt | model | parser


# print(chain.invoke({"doc":doc[0].page_content}))

print(doc)
print(type(doc))
print(len(doc))
print(doc[0].page_content)