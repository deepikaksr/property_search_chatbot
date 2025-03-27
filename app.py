import os
import pandas as pd
import streamlit as st
import chromadb
import requests
import re
from dotenv import load_dotenv
import numpy as np
import mysql.connector
from sqlalchemy import create_engine

# Load environment variables
load_dotenv()

# ChromaDB Initialization
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("real_estate_properties")

# MySQL database connection
def connect_db():
    return mysql.connector.connect(
        host=os.getenv('MYSQL_HOST'),
        user=os.getenv('MYSQL_USER'),
        password=os.getenv('MYSQL_PASSWORD'),
        database=os.getenv('MYSQL_DB')
    )

# Load CSV Data to MySQL
def load_data_to_mysql(df):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS properties (
        id INT AUTO_INCREMENT PRIMARY KEY,
        street VARCHAR(255),
        city VARCHAR(100),
        state VARCHAR(100),
        price FLOAT,
        bed INT,
        bath INT,
        house_size FLOAT
    )""")
    
    insert_query = "INSERT INTO properties (street, city, state, price, bed, bath, house_size) VALUES (%s,%s,%s,%s,%s,%s,%s)"
    for _, row in df.iterrows():
        cursor.execute(insert_query, (row['street'], row['city'], row['state'], row['price'], row['bed'], row['bath'], row['house_size']))

    conn.commit()
    cursor.close()
    conn.close()
    st.success("✅ Data loaded into MySQL database successfully!")

# Load Data from MySQL
def load_data():
    engine = create_engine(
        f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@{os.getenv('MYSQL_HOST')}/{os.getenv('MYSQL_DB')}"
    )
    df = pd.read_sql("SELECT * FROM properties", engine)
    return df

# Gemini API Embedding Function
def get_embedding(text):
    api_key = os.getenv("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-exp-03-07:embedContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {"content": {"parts": [{"text": text}]}}

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        embedding = response.json()['embedding']['values']
        return embedding
    else:
        st.error(f"Error fetching embeddings: {response.status_code}")
        return [0.0]*768

# Store Embeddings in ChromaDB
def store_embeddings(df):
    texts = df.apply(lambda row: f"{row['street']}, {row['city']}, {row['state']}, ${row['price']}, {row['bed']} bed, {row['bath']} bath, {row['house_size']} sqft", axis=1).tolist()
    embeddings = [get_embedding(text) for text in texts]
    collection.upsert(
        ids=[str(idx) for idx in df.index],
        embeddings=embeddings,
        metadatas=df.to_dict('records')
    )
    st.success("✅ Embeddings stored successfully!")

# Parse User Query
def parse_query(query):
    filters = {}
    price_match = re.findall(r'(?:under|below)\s*\$?(\d+[\,\d]*)', query)
    beds_match = re.findall(r'(\d+)\s*bed', query)
    baths_match = re.findall(r'(\d+)\s*bath', query)
    location_match = re.findall(r'in\s+(\w+),?\s?(\w+)?', query)

    if price_match:
        filters['max_price'] = float(price_match[0].replace(',', ''))
    if beds_match:
        filters['beds'] = int(beds_match[0])
    if baths_match:
        filters['baths'] = int(baths_match[0])
    if location_match:
        filters['city'] = location_match[0][0]
        if location_match[0][1]:
            filters['state'] = location_match[0][1]

    return filters

# RAG Search Optimization
def optimized_search(query, filters, top_k=5):
    query_embedding = get_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    matched_properties = []
    for meta in results['metadatas'][0]:
        if filters:
            if 'max_price' in filters and meta['price'] > filters['max_price']:
                continue
            if 'beds' in filters and meta['bed'] < filters['beds']:
                continue
            if 'baths' in filters and meta['bath'] < filters['baths']:
                continue
            if 'city' in filters and meta['city'].lower() != filters['city'].lower():
                continue
            if 'state' in filters and meta['state'].lower() != filters['state'].lower():
                continue
        matched_properties.append(meta)

    return matched_properties

# Streamlit UI
st.set_page_config(page_title="AI-Powered Property Search", page_icon="🏡")
st.title("🏡 AI-Powered Property Search")

with st.sidebar:
    st.info("⚠️ Only run these if your data has changed:")
    if st.button("Load CSV to MySQL"):
        csv_df = pd.read_csv("real_estate_data.csv").dropna()
        load_data_to_mysql(csv_df)
    if st.button("Store Embeddings"):
        df = load_data()
        store_embeddings(df)

st.subheader("💬 Property Assistant")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Ask about properties (e.g., '3 bed houses in Austin under $500,000')")

if user_query:
    df = load_data()
    filters = parse_query(user_query)
    results = optimized_search(user_query, filters)

    st.session_state.chat_history.append({"role": "user", "content": user_query})

    if results:
        response = f"I found {len(results)} properties matching your criteria:"
        st.write(response)
        st.dataframe(pd.DataFrame(results)[['street', 'city', 'state', 'price', 'bed', 'bath', 'house_size']])
    else:
        response = "No matching properties found. Can you adjust your filters or preferences?"
        st.warning(response)
