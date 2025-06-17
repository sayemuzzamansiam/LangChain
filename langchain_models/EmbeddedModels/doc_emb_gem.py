from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from dotenv import load_dotenv
import os 

load_dotenv()


embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    dimensions= 25
    
    )


documents= [
    "Dhaka is the Capital of Bangladesh",
    "Rome is the Capital of Italy",
    "Free Palestine"
]

result = embedding.embed_documents(documents)
print(str(result)) 
print(f"\nVector length: {len(result)}")