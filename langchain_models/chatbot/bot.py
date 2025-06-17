from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from dotenv import load_dotenv
import os

load_dotenv()

model = GoogleGenerativeAI(model="gemini-1.5-flash-002")

memory = [
    SystemMessage(content="you are a helpful ai")
    
    ] # to keep chat history

while True:
    user_input = input("You: ")
    memory.append(HumanMessage(content=user_input))
    if user_input.lower() == 'exit':
        break
    result = model.invoke(memory)

    memory.append(AIMessage(result))
    print("AI: ", result)

print(memory)