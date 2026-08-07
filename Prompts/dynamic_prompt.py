from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate, load_prompt 
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id=os.getenv("HUGGINGFACE_MODEL_REPO"),
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)

chat_model=ChatHuggingFace(llm=llm)

st.header("Research Page")

topic_input = st.selectbox( "Select Research topic Name", ["India", "SriLanka", "America", "Iran"] )

# templete=PromptTemplate(
#     template='''
#         tell about {topic}
#     ''',
#     input_variables=["topic"],
#     validation_template=True
# )

template=load_prompt("template_tpoic.json")

prompt=template.invoke({
    "topic":topic_input
}) 


if st.button("Summerize"):
    if prompt:
        response=chat_model.invoke(prompt)
        st.write(f"final Query is: {prompt.text}")
        st.write(response.content)
    else:
        st.write("give the prompt first.")