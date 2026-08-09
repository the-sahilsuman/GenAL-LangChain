from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, EmailStr
from dotenv import load_dotenv
import os

load_dotenv()


class Person(BaseModel):
    name: str = Field(description="Name of person")
    age: int = Field(gt=20, description="age")
    email: EmailStr = Field(description="Email of person")
    city: str = Field(description="address")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template=(
        "Give name, age, email, city of a fictinal person. \n"
        "{format_instruction}"
    ),
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

llm = HuggingFaceEndpoint(
    repo_id=os.getenv("HUGGINGFACE_MODEL_REPO"),
    task="text-generation",
    max_new_tokens=500,
    temperature=0.5
)

chat_model = ChatHuggingFace(llm=llm)

chain = template | chat_model | parser
result = chain.invoke({})
print(result)
print(type(result))
