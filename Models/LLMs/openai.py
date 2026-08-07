from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm=OpenAI(model="OPENAI_LLM_MODEL")

result=llm.invoke("input")

print(result)
 