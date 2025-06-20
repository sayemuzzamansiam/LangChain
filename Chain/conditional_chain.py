import os

from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal



from dotenv import load_dotenv

load_dotenv()


model = ChatOpenAI(
    model_name="llama3-8b-8192",  # confirm this is the right model name from Groq dashboard
    openai_api_key=os.getenv("GROQ_API_KEY"),  # Groq API key from your .env
    openai_api_base="https://api.groq.com/openai/v1",  # Important: Groq API base URL
    temperature=0.7,  # Optional tuning parameter
    max_tokens=500  # Optional: control response length
)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description= "Give the feedback sentiment ")
    

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback text into positive or negative.\n"
        "{feedback}\n\n"
        "Output ONLY a JSON object matching the format below, with no extra text or explanation:\n"
        "{format_instructions}"
    ),
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)


classification_chain = prompt1 | model | parser2

# now the branch chain

prompt2 = PromptTemplate(
    template='Write an appropriate response to this Positive feedback \n {feedback}',
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain= RunnableBranch(
    (lambda x:x.sentiment =='positive', prompt2 | model | parser),
    (lambda x:x.sentiment =='negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classification_chain | branch_chain

x = chain.invoke({"feedback": "I love the new features in this product, they are amazing!"})

print(x)

chain.get_graph().print_ascii()