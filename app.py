import os
import pandas as pd
import mysql.connector
import streamlit as st
import chromadb
import requests
from dotenv import load_dotenv

# Load API keys and DB credentials
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

# Initialize ChromaDB for storing embeddings
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("real_estate_properties")

# Function to load dataset
def load_data():
    df = pd.read_csv("real_estate_data.csv")
    df = df.where(pd.notna(df), None)  # Replace NaN with None for MySQL compatibility
    return df

# Function to insert data into MySQL
def insert_property_data():
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )
    cursor = conn.cursor()

    # Create properties table if not exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS properties (
        id INT AUTO_INCREMENT PRIMARY KEY,
        brokered_by TEXT,
        status TEXT,
        price FLOAT,
        bed INT,
        bath INT,
        acre_lot FLOAT,
        street TEXT,
        city TEXT,
        state TEXT,
        zip_code INT,
        house_size FLOAT,
        prev_sold_date TEXT
    )
    """)

    # Load and insert data
    df = load_data()
    for _, row in df.iterrows():
        cursor.execute("""
        INSERT INTO properties (brokered_by, status, price, bed, bath, acre_lot, street, city, state, zip_code, house_size, prev_sold_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            row['brokered_by'], row['status'], row['price'], row['bed'], row['bath'], 
            row['acre_lot'], row['street'], row['city'], row['state'], row['zip_code'], 
            row['house_size'], row['prev_sold_date']
        ))

    conn.commit()
    cursor.close()
    conn.close()
    st.success("✅ Data successfully inserted into MySQL!")

# Function to get property embeddings using Hugging Face API
def get_embedding(text):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    response = requests.post(
        "https://api-inference.huggingface.co/models/meta-llama/Llama-3-1B",
        headers=headers,
        json={"inputs": text}
    )
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Function to store property embeddings in ChromaDB
def store_embeddings():
    df = load_data()
    for _, row in df.iterrows():
        text = f"{row['city']}, {row['state']}, {row['price']} USD, {row['bed']} Beds, {row['bath']} Baths"
        embedding = get_embedding(text)
        if embedding:
            collection.add(
                ids=[str(row['street'])],
                embeddings=[embedding],
                metadatas=[{"city": row["city"], "state": row["state"], "price": row["price"]}]
            )
    st.success("✅ Property embeddings stored in ChromaDB!")

# Function to query properties using Llama 3.1B
def query_llama(prompt):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    response = requests.post(
        "https://api-inference.huggingface.co/models/meta-llama/Llama-3-1B",
        headers=headers,
        json={"inputs": prompt}
    )
    if response.status_code == 200:
        return response.json()["generated_text"]
    else:
        return "⚠️ Error fetching response from Llama 3.1B!"

# Streamlit UI
st.title("🏡 Property Search Chatbot")

# Buttons to insert data and store embeddings
if st.button("Insert Property Data into MySQL"):
    insert_property_data()

if st.button("Store Property Embeddings in ChromaDB"):
    store_embeddings()

# Chatbot interface
st.subheader("💬 Chat with Llama 3.1B")
user_query = st.text_input("Ask about real estate properties:")

if st.button("Search"):
    if user_query:
        response = query_llama(user_query)
        st.write("🤖 **Llama 3.1B:**", response)
    else:
        st.warning("⚠️ Please enter a query.")

# Display stored properties
st.subheader("📌 Available Properties")
df = load_data()
st.write(df.head(10))  # Show top 10 properties
