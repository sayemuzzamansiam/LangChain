from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

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


joke_chain = RunnableSequence(prompt, model, parser) # this will generate a joke 

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(), # this will pass the joke 
    'explanation': RunnableSequence(prompt2, model, parser) # this will explain the joke 
})

chain = joke_chain | parallel_chain # we can replace RunnableSequence with | | this kinda pipe like system -LCEL
x =chain.invoke({'topic': 'death note'})

print(x)