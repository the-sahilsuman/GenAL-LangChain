from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chatmode=ChatGoogleGenerativeAI(model=GOOGLE_GEN_MODEL)

result=chat_model.invoke("input")
print(result) 