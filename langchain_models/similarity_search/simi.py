
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np


load_dotenv()

embedding= GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    dimensions = 300
)

documents = [
    "Shakib al Hasan is an Bangladeshi cricketer known for his aggressive batting and leadership.",
    "Mashrafi is a former Bangladeshi captain famous for his calm demeanor.",
    "Tamim Iqbal is known for his elegant batting and record-breaking double centuries.",
    "Mustafiz is an Bangladeshi fast bowler known for his unorthodox action and yorkers."
]

query = "Tell me about Mustafiz"

doc_em = embedding.embed_documents(documents) # embedding the doc

query_em = embedding.embed_query(query) # embedding the query 

scores = cosine_similarity([query_em],doc_em)[0] # finding similarity between doc and query to get the answer 

print(scores)

print(sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]) # assigning index to get the most similar text from the embedded value.

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print("User Query: ", query)
print(documents[index])
print("similarity score is: ", score)