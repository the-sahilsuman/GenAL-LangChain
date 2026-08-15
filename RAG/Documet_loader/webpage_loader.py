from langchain_community.document_loaders import WebBaseLoader

url="https://thesahilsuman.info"

loader=WebBaseLoader(url)

docs=loader.load()
print(docs)
print(len(docs))