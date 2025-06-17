from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from dotenv import load_dotenv
import os 

load_dotenv()


embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    dimensions= 25
    
    )

result = embedding.embed_query("Dhaka is the capital of Bangladesh")
print(str(result)) 
print(f"\nVector length: {len(result)}")