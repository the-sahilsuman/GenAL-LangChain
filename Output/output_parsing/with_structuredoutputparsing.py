from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
import os

load_dotenv()

schema=[
    ResponseSchema(name="fact1"),
    ResponseSchema(name="fact2"),
    ResponseSchema(name="fact3"),
    ResponseSchema(name="fact4"),
    ResponseSchema(name="fact5")
]

parser = StructuredOutputParser.from_response_schema(schema)

template = PromptTemplate(
    template="Give the 5 line about {topic}. \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

llm = HuggingFaceEndpoint(
    repo_id=os.getenv("HUGGINGFACE_MODEL_REPO"),
    task="text-generation",
    max_new_tokens=300,
    temperature=0.8
)

chat_model = ChatHuggingFace(llm=llm)

chain = template | chat_model | parser
result = chain.invoke({"topic": "Black Hole"})
print(result)
print(type(result))
