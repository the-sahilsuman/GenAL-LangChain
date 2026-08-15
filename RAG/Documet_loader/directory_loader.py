from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader=DirectoryLoader(
    path="all_pdfs",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs=loader.load()
# print(docs)
print(len(docs))