from langchain_openai import OpenAIEmbeddings
from dutenv import load_dotenv

load_dotenv()

embedding=OpenAIEmbeddings(model="OPENAI_EMBEDDING_MODEL")

result=embedding.embed.query("INPUT")

print(result)
