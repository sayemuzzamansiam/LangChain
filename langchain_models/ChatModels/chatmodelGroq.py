from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

import os

load_dotenv()

llm = ChatOpenAI(
    model="llama3-70b-8192",
    openai_api_key = os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)

response= llm.invoke("what is capital of bangladesh?") # langchain's invoke model
#print(response) # this will provide answer with meta-data 
#print(response.content)# here we are filtering out the content so we only get the actual answer not the meta data.


#===========================================
# Dealing with arguments
#===========================================

llm2 = ChatOpenAI(
    model="llama3-70b-8192",
    openai_api_key = os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=1.1, # lower value more deterministic answer. high value more creative answer 
    max_completion_tokens= 20, # how many token the output should take in answering.
)
result = llm2.invoke("tell me a joke on education") # for joke we made the temp high.
print(result.content)



