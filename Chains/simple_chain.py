from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

parser=StrOutputParser()

template=PromptTemplate(
    template=("Write 5 line about {topic}."),
    input_variables=["topic"]
    # partial_variables={"format_instructions":parser.get_format_in structions()}
)

llm=HuggingFaceEndpoint(
    repo_id=os.getenv("HUGGINGFACE_MODEL_REPO"),
    task="text-generation",
    max_new_tokens=400,
    temperature=0.8
)

chat_model=ChatHuggingFace(llm=llm)

chain= template | chat_model | parser
print(chain.invoke({"topic":"India"}))
chain.get_graph().print_ascii()
