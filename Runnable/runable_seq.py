from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

from dotenv import load_dotenv

load_dotenv()


prompt = PromptTemplate(
    template= 'write a joke about {topic}',
    input_variables = ['topic']
)

model = GoogleGenerativeAI(model="gemini-1.5-flash-002")

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template= 'explain the following:  {text}',
    input_variables = ['text']
)

chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)

x = chain.invoke({'topic': 'death note'})

print(x)

