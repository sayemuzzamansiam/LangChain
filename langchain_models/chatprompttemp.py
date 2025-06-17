from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage


chat_temp = ChatPromptTemplate([
    ('system', 'you are a helpful {domain} expert'),
    ('human','explain in simple terms, what is  {topic}')
    
])

prompt = chat_temp.invoke({'domain':'anime', 'topic':'death note'})
print(prompt)