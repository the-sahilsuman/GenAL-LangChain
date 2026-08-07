from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

chat_model=ChatAnthropic(model="ANTHROPIC_CHATMODEL", temperature=1, max_completion_token=400)

result=chat_model.invoke(" Input ")

print(result)
