from langchain_openai import OpenAIEmbeddings
from dutenv import load_dotenv

load_dotenv()

embedding=OpenAIEmbeddings(model="OPENAI_EMBEDDING_MODEL")

documents=[
    "LINE1",
    "LINE2"
]

result=embedding.embed.documents(dicument)

print(result)
