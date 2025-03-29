import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import json
import re
import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions


HIDE_SLIDER_TICKS_CSS = """
<style>
/* Hide the min and max labels on Streamlit sliders */
span[data-baseweb="slider"] [data-testid="stTickBarMin"], 
span[data-baseweb="slider"] [data-testid="stTickBarMax"] {
    display: none !important;
}
</style>
"""

# Load environment variables and configure Gemini API
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

# Initialize session state variables
if "sqlite_connection" not in st.session_state:
    st.session_state.sqlite_connection = sqlite3.connect(":memory:", check_same_thread=False)
if "property_data" not in st.session_state:
    st.session_state.property_data = None

# Initialize ChromaDB with a persistent directory
if "chroma_client" not in st.session_state:
    try:
        if not os.path.exists("./chroma_db"):
            os.makedirs("./chroma_db")
        st.session_state.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection_name = "property_embeddings"
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        collection_names = st.session_state.chroma_client.list_collections()
        if collection_name not in collection_names:
            st.session_state.embedding_collection = st.session_state.chroma_client.create_collection(
                name=collection_name,
                embedding_function=default_ef
            )
        else:
            st.session_state.embedding_collection = st.session_state.chroma_client.get_collection(
                name=collection_name,
                embedding_function=default_ef
            )
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        st.session_state.chroma_client = chromadb.Client()
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        st.session_state.embedding_collection = st.session_state.chroma_client.create_collection(
            name="property_embeddings",
            embedding_function=default_ef
        )

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

def create_property_chunks(df, chunk_size=20):
    chunks, ids, metadatas = [], [], []
    for i in range(0, len(df), chunk_size):
        batch = df.iloc[i:i+chunk_size]
        for idx, row in batch.iterrows():
            property_desc = " ".join([f"{col}: {str(val)}" for col, val in row.items()])
            chunks.append(property_desc)
            ids.append(f"property_{idx}")
            metadata = {
                "price": str(row.get("price", "")),
                "bed": str(row.get("bed", "")),
                "bath": str(row.get("bath", "")),
                "status": str(row.get("status", "")),
                "index": str(idx)
            }
            metadatas.append(metadata)
    return chunks, ids, metadatas

def process_property_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        df = df.dropna(axis=1, how='all')
        numeric_cols = detect_numeric_columns(df)
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        table_name = "property_listings"
        df.to_sql(table_name, st.session_state.sqlite_connection, index=False, if_exists="replace")
        chunks, ids, metadatas = create_property_chunks(df)
        st.session_state.property_data = df
        build_chroma_db(chunks, ids, metadatas)
        return df
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return None

def detect_numeric_columns(df):
    potential_numeric_cols = [
        'price', 'bed', 'bath', 'acre_lot', 'house_size', 'zip_code',
        'bedrooms', 'bathrooms', 'size', 'square_feet', 'sq_ft', 'sqft',
        'area', 'lot_size', 'year', 'year_built', 'built_year', 'age',
        'stories', 'floors', 'rooms', 'garage'
    ]
    numeric_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in potential_numeric_cols):
            numeric_cols.append(col)
        else:
            ratio_numeric = df[col].apply(
                lambda x: isinstance(x, (int, float)) or (
                    isinstance(x, str) and re.match(r'^[$£€]?\d+[,.]?\d*[kKmM]?$', x.strip())
                )
            ).mean()
            if ratio_numeric > 0.7:
                numeric_cols.append(col)
    return numeric_cols

def build_chroma_db(chunks, ids, metadatas):
    try:
        collection_name = "property_embeddings"
        collection_names = st.session_state.chroma_client.list_collections()
        if collection_name in collection_names:
            st.session_state.chroma_client.delete_collection(collection_name)
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        st.session_state.embedding_collection = st.session_state.chroma_client.create_collection(
            name=collection_name,
            embedding_function=default_ef
        )
        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            st.session_state.embedding_collection.add(
                documents=chunks[i:end_idx],
                ids=ids[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
    except Exception as e:
        print(f"Error building ChromaDB: {e}")
        import traceback
        print(traceback.format_exc())

def retrieve_property_info(query, top_n=5):
    try:
        results = st.session_state.embedding_collection.query(
            query_texts=[query],
            n_results=top_n
        )
        if results and 'documents' in results and len(results['documents']) > 0:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            formatted_results = []
            for doc, meta, dist in zip(documents, metadatas, distances):
                similarity = 1 - dist
                formatted_results.append((doc, similarity, meta))
            return formatted_results
        return []
    except Exception as e:
        print(f"Error retrieving from ChromaDB: {e}")
        return []

def generate_property_sql(user_query):
    if st.session_state.property_data is None:
        return ""
    columns = ", ".join(st.session_state.property_data.columns)
    prompt_text = f"""
You are an expert SQL query generator for a real estate database.
Table Name: property_listings
Columns: {columns}

Rules:
1. Always use valid SQLite syntax
2. When user asks to "list" or "show" or "display" properties, use SELECT statements
3. For price-related queries, make sure to handle numeric comparison correctly
4. For location-based queries, use LIKE with wildcards
5. If user asks for "expensive" properties without specifics, use price > 100000
6. If user asks for "affordable" properties without specifics, use price < 80000
7. Apply proper sorting (ORDER BY) for relevant queries
8. Limit results to 20 by default using LIMIT clause
9. For bedroom/bathroom queries, use exact matches (=) or greater than (>=)
10. Use the specific column names from the dataset: brokered_by, status, price, bed, bath, acre_lot, street, city, state, zip_code, house_size, prev_sold_date

Please respond in this exact JSON format:
{{"sql_query": string, "confidence": float, "explanation": string}}

User question:
"{user_query}"
"""
    try:
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        response = gemini_model.generate_content(prompt_text)
        raw_output = response.text.strip()
        cleaned_output = re.sub(r'```json\n(.*?)\n```', r'\1', raw_output, flags=re.DOTALL)
        cleaned_output = re.sub(r'```(.*?)```', r'\1', cleaned_output, flags=re.DOTALL)
        try:
            parsed_json = json.loads(cleaned_output)
            sql_query = parsed_json.get("sql_query", "")
            print(f"Generated SQL: {sql_query}")
            return sql_query
        except json.JSONDecodeError:
            sql_match = re.search(r'SELECT.*?;', cleaned_output, re.DOTALL | re.IGNORECASE)
            if sql_match:
                return sql_match.group(0)
            return ""
    except Exception as e:
        print(f"Error generating SQL: {e}")
        return ""

def generate_property_answer(user_query, sql_result=None, is_dataframe=False):
    if is_dataframe:
        return "Here are the property listings matching your search criteria:"
    retrieved_results = retrieve_property_info(user_query)
    context = "\n".join([f"- {chunk}" for chunk, similarity, meta in retrieved_results])
    prompt_text = f"""
You are a knowledgeable real estate assistant helping users find properties.

User question: "{user_query}"

Database query results: 
{sql_result if sql_result else "No specific properties found matching your criteria."}

Additional property context:
{context}

Please provide a helpful, concise answer about the properties. 
If showing summary statistics, mention key details like count, price ranges, bedrooms, bathrooms, and lot sizes.
If no properties were found, suggest alternative search criteria.
"""
    try:
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        response = gemini_model.generate_content(prompt_text)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating property answer: {e}")
        return "I found some properties matching your criteria, but I'm having trouble summarizing the results."

def execute_property_query(sql_query):
    try:
        result_df = pd.read_sql_query(sql_query, st.session_state.sqlite_connection)
        if len(result_df) == 0:
            return None, "No properties found matching those criteria."
        return result_df, None
    except Exception as e:
        print(f"SQL execution error: {e}")
        if "no such column" in str(e).lower():
            error_detail = str(e)
            column_match = re.search(r"no such column: ([^\s]+)", error_detail, re.IGNORECASE)
            if column_match:
                bad_column = column_match.group(1)
                if st.session_state.property_data is not None:
                    similar_columns = [
                        col for col in st.session_state.property_data.columns
                        if bad_column.lower() in col.lower()
                    ]
                    if similar_columns:
                        return None, f"Column '{bad_column}' not found. Did you mean one of these: {', '.join(similar_columns)}?"
        return None, f"Error executing search: {e}"

def hybrid_property_search(user_query):
    sql_query = generate_property_sql(user_query)
    if sql_query:
        result_df, error_message = execute_property_query(sql_query)
        if error_message:
            return None, error_message
        return result_df, None
    else:
        return None, "Unable to generate a search query for your question."

# Set page config
st.set_page_config(layout="wide", page_title="Property Search Chatbot")

# Inject custom CSS to hide min/max slider labels but keep bubble
st.markdown(HIDE_SLIDER_TICKS_CSS, unsafe_allow_html=True)

# Sidebar: CSV upload
with st.sidebar:
    st.title("Property Data Uploader")
    uploaded_file = st.file_uploader(
        "Upload property CSV file",
        type=["csv"],
        accept_multiple_files=False
    )
    if uploaded_file:
        with st.spinner("Processing property data..."):
            df = process_property_csv(uploaded_file)
        if df is not None:
            st.subheader("Dataset Information")
            cols = df.columns.tolist()
            st.write(f"Columns: {', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}")
            st.subheader("Sample Listings")
            st.dataframe(df.head(3), use_container_width=True)

st.title("🏠 Property Search Chatbot")
st.write("Ask questions about properties or search for listings that match your criteria.")

# Filter-based search
if st.session_state.property_data is not None:
    df = st.session_state.property_data
    st.markdown("## Filter Based Search")
    
    # First row of filters (Price, Bathrooms, Bedrooms, House Size)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if 'price' in df.columns and not df['price'].dropna().empty:
            price_min = int(df['price'].min())
            price_max = int(df['price'].max())
            selected_price_range = st.slider(
                "Price Range", 
                min_value=price_min, 
                max_value=price_max, 
                value=(price_min, price_max)
            )
        else:
            selected_price_range = None

    with col2:
        if 'bath' in df.columns and not df['bath'].dropna().empty:
            bath_min = int(df['bath'].min())
            bath_max = int(df['bath'].max())
            selected_bath_range = st.slider(
                "Bathrooms", 
                min_value=bath_min, 
                max_value=bath_max, 
                value=(bath_min, bath_max)
            )
        else:
            selected_bath_range = None

    with col3:
        if 'bed' in df.columns and not df['bed'].dropna().empty:
            bed_min = int(df['bed'].min())
            bed_max = int(df['bed'].max())
            selected_bed_range = st.slider(
                "Bedrooms", 
                min_value=bed_min, 
                max_value=bed_max, 
                value=(bed_min, bed_max)
            )
        else:
            selected_bed_range = None

    with col4:
        if 'house_size' in df.columns and not df['house_size'].dropna().empty:
            size_min = int(df['house_size'].min())
            size_max = int(df['house_size'].max())
            selected_size_range = st.slider(
                "House Size", 
                min_value=size_min, 
                max_value=size_max, 
                value=(size_min, size_max)
            )
        else:
            selected_size_range = None

    # Second line: City filter
    selected_cities = None
    if 'city' in df.columns and not df['city'].dropna().empty:
        cities = sorted(df['city'].dropna().unique().tolist())
        st.markdown("### Select City")
        selected_cities = st.multiselect(
            label="",  # no label text, just the chips
            options=cities,
            default=cities
        )

    # Apply filters
    filtered_df = df.copy()

    if selected_price_range:
        filtered_df = filtered_df[
            (filtered_df['price'] >= selected_price_range[0]) & 
            (filtered_df['price'] <= selected_price_range[1])
        ]
    if selected_bath_range:
        filtered_df = filtered_df[
            (filtered_df['bath'] >= selected_bath_range[0]) & 
            (filtered_df['bath'] <= selected_bath_range[1])
        ]
    if selected_bed_range:
        filtered_df = filtered_df[
            (filtered_df['bed'] >= selected_bed_range[0]) & 
            (filtered_df['bed'] <= selected_bed_range[1])
        ]
    if selected_size_range:
        filtered_df = filtered_df[
            (filtered_df['house_size'] >= selected_size_range[0]) & 
            (filtered_df['house_size'] <= selected_size_range[1])
        ]
    if selected_cities:
        filtered_df = filtered_df[filtered_df['city'].isin(selected_cities)]

    st.write(f"Found {len(filtered_df)} properties matching the filters.")
    st.dataframe(filtered_df, use_container_width=True)

st.markdown("---")

st.markdown("## User Query Based Search")
st.write("You can also ask a natural language question to search for properties.")

with st.expander("Example Queries"):
    st.markdown("""
    - Show me properties with 3+ bedrooms under $80,000
    - List properties with status "for_sale" sorted by price
    - Find properties with more than 2 bathrooms
    - Display houses with acre_lot greater than 0.1
    - Show me properties with house_size larger than 1000 square feet
    - Find properties brokered by 103378 or 52707
    - Show me the most expensive properties
    - List properties with 4 bedrooms and 2 bathrooms
    """)

user_input = st.chat_input("Search for properties...")

if user_input:
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    if st.session_state.property_data is None:
        final_answer = "Please upload a property CSV file to begin searching."
        st.session_state.chat_messages.append({"role": "assistant", "content": final_answer})
    else:
        result_df, error_message = hybrid_property_search(user_input)
        if error_message:
            final_answer = error_message
            st.session_state.chat_messages.append({"role": "assistant", "content": final_answer})
        else:
            if result_df is not None and len(result_df) > 0:
                final_answer = generate_property_answer(user_input, is_dataframe=True)
                st.session_state.chat_messages.append({
                    "role": "assistant", 
                    "content": final_answer,
                    "has_dataframe": True,
                    "dataframe": result_df
                })
            else:
                final_answer = "No properties found matching your criteria. Try broadening your search."
                st.session_state.chat_messages.append({"role": "assistant", "content": final_answer})

for message in st.session_state.chat_messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        msg_container = st.chat_message("assistant")
        msg_container.write(message["content"])
        if message.get("has_dataframe") and "dataframe" in message:
            df_to_show = message["dataframe"]
            try:
                if 'price' in df_to_show.columns and df_to_show['price'].dtype in ['int64', 'float64']:
                    df_to_show['price'] = df_to_show['price'].apply(lambda x: f"${int(x):,}" if pd.notnull(x) else "")
                msg_container.dataframe(
                    df_to_show, 
                    use_container_width=True,
                    column_config={
                        col: st.column_config.TextColumn(
                            col, width="medium"
                        ) for col in df_to_show.columns 
                        if df_to_show[col].dtype == 'object' 
                        and len(df_to_show[col].astype(str).str.len()) > 0 
                        and df_to_show[col].astype(str).str.len().max() > 20
                    }
                )
                msg_container.caption(f"Showing {len(df_to_show)} properties")
            except Exception as e:
                msg_container.write("Error displaying property table.")
                msg_container.dataframe(df_to_show)
