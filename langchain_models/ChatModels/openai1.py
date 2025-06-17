from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize the ChatOpenAI model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # Use a model that is available
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=1.1,
    max_completion_tokens=20
)

# Invoke the model with a prompt
result = llm.invoke("Tell me a joke about education")

# Print the response content
print(result.content)
