import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a detailed report on the topic: {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Make a short summary or 3 points from the following text:\n{text}",
    input_variables=["text"]
)

model = ChatOpenAI(
    model_name="llama3-8b-8192",  # confirm this is the right model name from Groq dashboard
    openai_api_key=os.getenv("GROQ_API_KEY"),  # Groq API key from your .env
    openai_api_base="https://api.groq.com/openai/v1",  # Important: Groq API base URL
    temperature=0.7,  # Optional tuning parameter
    max_tokens=500  # Optional: control response length
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "Artificial Intelligence"})
print(result)

chain.get_graph().print_ascii()
