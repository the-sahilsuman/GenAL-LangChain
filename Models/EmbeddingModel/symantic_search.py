from langchain_openai import OpenAIEmbeddings
from dutenv import load_dotenv
from sklearn.metrix.pairwise import cosine_simialarity

load_dotenv()

embedding=OpenAIEmbeddings(model="OPENAI_EMBEDDING_MODEL")

documents=[
    "LINE1",
    "LINE2"
]

document_embedding=embedding.embed.documents(documents)

query_embedding=embedding.embed.query("INPUT")

print(cosine_simialarity([query_embedding],document_embedding))
