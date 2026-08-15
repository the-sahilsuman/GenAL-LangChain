from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader=TextLoader("cricket.txt")

docs=loader.load()

splitter=CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=" "
)

chunks=splitter.split_documents(docs)
# chunks=splitter.split_text(docs).  for normal text 

print(len(chunks))

# for chunk in chunks:
#     print(chunk.page_content)