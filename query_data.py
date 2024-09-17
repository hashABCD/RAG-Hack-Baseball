import os
from Embeddings import get_embedding_function
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
import psycopg2
import numpy as np
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

#set up pg vector DB
load_dotenv(override=True)
DBUSER = os.environ["DBUSER"]
DBPASS = os.environ["DBPASS"]
DBHOST = os.environ["DBHOST"]
DBNAME = os.environ["DBNAME"]
DATABASE_URI = f"postgresql://{DBUSER}:{DBPASS}@{DBHOST}/{DBNAME}"

# Use SSL if not connecting to localhost
DBSSL = "disable"
if DBHOST != "localhost":
    DBSSL = "require"

# Connect to PostgreSQL database in Timescale using connection string
conn = psycopg2.connect(
    host=DBHOST,                   
    database=DBNAME,           
    user=DBUSER,                   
    password=DBPASS,             
    port=5432                     
)
cur = conn.cursor()

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def get_similar_docs(query_embedding, conn):
    embedding_array = np.array(query_embedding)
    # Register pgvector extension
    register_vector(conn)
    cur = conn.cursor()
    # Get the top 3 most similar documents using the KNN <=> operator
    cur.execute("SELECT embedding FROM embeddings ORDER BY embedding <=> %s LIMIT 3", (embedding_array,))
    top3_docs = cur.fetchall()
    return top3_docs

def get_embeddings(user_input):
    embed_query = get_embedding_function().embed_query(user_input)
    return embed_query


if __name__ == "__main__":
    user_input = "how many players are in baseball team?"
    embed_query = get_embeddings(user_input)
    res = get_similar_docs(embed_query, conn)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=res, question=embed_query)
    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)
    print(response_text)