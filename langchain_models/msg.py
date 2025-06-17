from langchain_google_genai import GoogleGenerativeAI

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model="gemini-1.5-flash-002")


messages = [
    SystemMessage(content='You are a smart ai'),
    HumanMessage(content="tell me about LangChain")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result))

print(messages)