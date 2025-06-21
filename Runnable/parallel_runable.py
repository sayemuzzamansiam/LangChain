# here multiple runnables can be run in parallel

from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence

from dotenv import load_dotenv

load_dotenv()


prompt = PromptTemplate(
    template= 'write a tweet about {topic}',
    input_variables = ['topic']
)
prompt2 = PromptTemplate(
    template= 'write a linkedIn {topic}',
    input_variables = ['topic']
)


model = GoogleGenerativeAI(model="gemini-1.5-flash-002")

parser = StrOutputParser()



chain = RunnableParallel(
    {
        'tweet': RunnableSequence(prompt, model, parser),
        'linkedin': RunnableSequence(prompt2, model, parser)
    }
)

x = chain.invoke({'topic': 'death note'})

print(x['tweet'])
print(x['linkedin'])

