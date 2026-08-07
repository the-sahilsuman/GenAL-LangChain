from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ("system", "you are helpful {domain} expert."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

chat_history = []

with open("chat_history.txt") as f:
    chat_history.extend(f.readlines())

prompt = chat_template.invoke({
    "domain": "Software",
    "chat_history": chat_history,
    "query": "what is langchain."
})

print(prompt)
