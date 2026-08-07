from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_model=ChatOpenAI(model="CHAT_OPENAI_API", temperature=0, max_completion_token=200)

result=chat_model.invoke("Input")

print(result)
print(result.content)
