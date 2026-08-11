from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()

template1=PromptTemplate(
    template="Write a detailed report on {topic}.",
    input_variable=["topic"]
)

template2=PromptTemplate(
    template="Write a 5 line summary from {text}",
    input_variable=["text"]
)


llm=HuggingFaceEndpoint(
    repo_id=os.getenv("HUGGINGFACE_MODEL_REPO"),
    task="text-generation",
    max_new_tokens=300,
    temperature=0.8
)


chat_model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

# chain= template1 | chat_model | parser | template2 | chat_model | parser

parallel_chain=RunnableParallel({
    "Report": RunnablePassthrough(),
    "Summary":RunnableSequence(template2,chat_model,parser)
    })

chain=RunnableSequence(template1,chat_model,parser,parallel_chain)

response=chain.invoke({"topic": input("Enter topic: ")})

print(response)

chain.get_graph().print_ascii()