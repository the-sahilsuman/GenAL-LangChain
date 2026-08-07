from langchain_core.prompts import PromptTemplate

templete=PromptTemplate(
    template='''
        tell about {topic}
    ''',
    input_variables=["topic"],
    validation_template=True
)

templete.save("template_tpoic.json")