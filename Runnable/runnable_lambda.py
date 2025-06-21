# from langchain.schema.runnable import RunnableLambda


def word_count(text):
   return len(text.split())

# word_count_runnable = RunnableLambda(word_count)

# x =word_count_runnable.invoke('Hi there how are you?')
# print(x)



from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableLambda,RunnableParallel,RunnablePassthrough


from dotenv import load_dotenv

load_dotenv()



prompt = PromptTemplate(
    template= 'write a joke about {topic}',
    input_variables = ['topic']
)

model = GoogleGenerativeAI(model="gemini-1.5-flash-002")

parser = StrOutputParser()

joke_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),  # this will pass the joke
    'word_count': RunnableLambda(word_count)  # this will explain the joke
})

chain = RunnableSequence(joke_chain, parallel_chain)
x = chain.invoke({'topic': 'death note'})


final_result = """ {} \n word count - {}""".format(x['joke'], x['word_count'])

print(final_result)