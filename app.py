import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import json
import re
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini API if available
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

# Initialize session state variables
if "sqlite_connection" not in st.session_state:
    st.session_state.sqlite_connection = sqlite3.connect(":memory:", check_same_thread=False)

if "property_data" not in st.session_state:
    st.session_state.property_data = None

if "all_text_chunks" not in st.session_state:
    st.session_state.all_text_chunks = []

if "tfidf_database" not in st.session_state:
    st.session_state.tfidf_database = {}

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Convert property dataframe to text chunks for RAG
def create_property_chunks(df, chunk_size=20):
    chunks = []
    for i in range(0, len(df), chunk_size):
        batch = df.iloc[i:i+chunk_size]
        # Create a text description for each property in the batch
        for _, row in batch.iterrows():
            property_desc = " ".join([f"{col}: {str(val)}" for col, val in row.items()])
            chunks.append(property_desc)
    return chunks

# Process CSV with property data
def process_property_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        
        # Clean the dataframe
        df = df.dropna(axis=1, how='all')
        
        # Ensure numeric columns are properly typed
        numeric_cols = detect_numeric_columns(df)
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Save to sqlite
        table_name = "property_listings"
        df.to_sql(table_name, st.session_state.sqlite_connection, index=False, if_exists="replace")
        
        # Extract text chunks for RAG
        text_chunks = create_property_chunks(df)
        
        # Store in session state
        st.session_state.property_data = df
        st.session_state.all_text_chunks = text_chunks
        
        return df, text_chunks
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None, []

# Detect which columns should be numeric
def detect_numeric_columns(df):
    # Columns that typically contain numeric values in property data
    potential_numeric_cols = [
        'price', 'bed', 'bath', 'acre_lot', 'house_size', 'zip_code',
        'bedrooms', 'bathrooms', 'size', 'square_feet', 'sq_ft', 'sqft',
        'area', 'lot_size', 'year', 'year_built', 'built_year', 'age',
        'stories', 'floors', 'rooms', 'garage'
    ]
    
    # Check column names
    numeric_cols = []
    for col in df.columns:
        col_lower = col.lower()
        # Check if column name contains any of the potential numeric terms
        if any(term in col_lower for term in potential_numeric_cols):
            numeric_cols.append(col)
        # Also check if column has mostly numeric values
        elif df[col].apply(lambda x: isinstance(x, (int, float)) or 
                         (isinstance(x, str) and re.match(r'^[$£€]?\d+[,.]?\d*[kKmM]?$', x.strip()))).mean() > 0.7:
            numeric_cols.append(col)
    
    return numeric_cols

# Build vector database for property data
def build_vector_db(text_chunks):
    if not text_chunks:
        return
    
    tfidf_vectorizer = TfidfVectorizer()
    embeddings = tfidf_vectorizer.fit_transform(text_chunks)
    st.session_state.tfidf_database = {
        "chunks": text_chunks,
        "embeddings": embeddings,
        "vectorizer": tfidf_vectorizer
    }

# Retrieve relevant property information using semantic search
def retrieve_property_info(query, top_n=5):
    if not st.session_state.tfidf_database:
        return []
    
    tfidf_vectorizer = st.session_state.tfidf_database["vectorizer"]
    embeddings = st.session_state.tfidf_database["embeddings"]
    
    # Create query vector
    query_vector = tfidf_vectorizer.transform([query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_vector, embeddings)[0]
    
    # Get top results
    indices = np.argsort(similarities)[::-1][:top_n]
    results = [(st.session_state.tfidf_database["chunks"][i], similarities[i]) for i in indices]
    
    return results

# Generate SQL query for property database
def generate_property_sql(user_query):
    if st.session_state.property_data is None:
        return ""
    
    # Get column information
    columns = ", ".join(st.session_state.property_data.columns)
    
    # Create a more specific prompt for property search
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
        
        # Clean the JSON output
        cleaned_output = re.sub(r'```json\n(.*?)\n```', r'\1', raw_output, flags=re.DOTALL)
        cleaned_output = re.sub(r'```(.*?)```', r'\1', cleaned_output, flags=re.DOTALL)
        
        # Parse JSON
        try:
            parsed_json = json.loads(cleaned_output)
            sql_query = parsed_json.get("sql_query", "")
            
            # Log the generated SQL
            print(f"Generated SQL: {sql_query}")
            
            return sql_query
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails: try to extract just the SQL
            sql_match = re.search(r'SELECT.*?;', cleaned_output, re.DOTALL | re.IGNORECASE)
            if sql_match:
                return sql_match.group(0)
            return ""
            
    except Exception as e:
        print(f"Error generating SQL: {e}")
        return ""

# Generate natural language answer about properties
def generate_property_answer(user_query, sql_result=None, is_dataframe=False):
    if is_dataframe:
        # For dataframe results, just return a placeholder message
        # The actual display will be handled by Streamlit
        return "Here are the property listings matching your search criteria:"
    
    # Get additional context from vector DB
    retrieved_chunks = retrieve_property_info(user_query)
    context = "\n".join([f"- {chunk}" for chunk, _ in retrieved_chunks])
    
    # Create prompt
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
        return f"I found some properties matching your criteria, but I'm having trouble summarizing the results. You can view the details in the table below."

# Execute SQL query on property database
def execute_property_query(sql_query):
    try:
        # Execute the query
        result_df = pd.read_sql_query(sql_query, st.session_state.sqlite_connection)
        
        # Check if we got results
        if len(result_df) == 0:
            return None, "No properties found matching those criteria."
        
        return result_df, None
    except Exception as e:
        print(f"SQL execution error: {e}")
        # Try to fix common SQL errors
        if "no such column" in str(e).lower():
            error_detail = str(e)
            column_match = re.search(r"no such column: ([^\s]+)", error_detail, re.IGNORECASE)
            if column_match:
                bad_column = column_match.group(1)
                # Suggest similar column
                if st.session_state.property_data is not None:
                    similar_columns = [col for col in st.session_state.property_data.columns 
                                     if bad_column.lower() in col.lower()]
                    if similar_columns:
                        return None, f"Column '{bad_column}' not found. Did you mean one of these: {', '.join(similar_columns)}?"
        
        return None, f"Error executing search: {e}"

# Streamlit UI
st.set_page_config(layout="wide", page_title="Property Search Chatbot")

# Sidebar with file uploader
with st.sidebar:
    st.title("Property Data Uploader")
    uploaded_file = st.file_uploader(
        "Upload property CSV file",
        type=["csv"],
        accept_multiple_files=False
    )
    
    if uploaded_file:
        with st.spinner("Processing property data..."):
            df, text_chunks = process_property_csv(uploaded_file)
            if df is not None:
                st.success(f"Loaded {len(df)} property listings")
                
                # Display dataset info
                st.subheader("Dataset Information")
                cols = df.columns.tolist()
                st.write(f"Columns: {', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}")
                
                # Build vector database
                build_vector_db(text_chunks)
                
                # Show sample of the data
                st.subheader("Sample Listings")
                st.dataframe(df.head(3), use_container_width=True)

# Main chat interface
st.title("🏠 Property Search Chatbot")
st.write("Ask questions about properties or search for listings that match your criteria.")

# Example queries tailored to the dataset
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

# Chat input
user_input = st.chat_input("Search for properties...") 

# Process user input
if user_input:
    # Add user message to chat
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    
    # Check if data is loaded
    if st.session_state.property_data is None:
        final_answer = "Please upload a property CSV file to begin searching."
        st.session_state.chat_messages.append({"role": "assistant", "content": final_answer})
    else:
        # Generate SQL query from user input
        sql_query = generate_property_sql(user_input)
        
        if sql_query:
            # Execute the query
            result_df, error_message = execute_property_query(sql_query)
            
            if error_message:
                # Handle error
                final_answer = error_message
                st.session_state.chat_messages.append({"role": "assistant", "content": final_answer})
            else:
                # Generate answer from results
                if result_df is not None:
                    final_answer = generate_property_answer(user_input, is_dataframe=True)
                    
                    # Add assistant message
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": final_answer,
                        "has_dataframe": True,
                        "dataframe": result_df
                    })
                else:
                    final_answer = "No properties found matching your criteria. Try broadening your search."
                    st.session_state.chat_messages.append({"role": "assistant", "content": final_answer})
        else:
            # If SQL generation failed, use RAG approach
            retrieved_results = retrieve_property_info(user_input, top_n=5)
            final_answer = generate_property_answer(user_input, sql_result=None)
            st.session_state.chat_messages.append({"role": "assistant", "content": final_answer})

# Display chat messages
for message in st.session_state.chat_messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        msg_container = st.chat_message("assistant")
        msg_container.write(message["content"])
        
        # Display property table if available
        if message.get("has_dataframe") and "dataframe" in message:
            df = message["dataframe"]
            # Enhance the display of the dataframe
            try:
                # Format price column
                if 'price' in df.columns and df['price'].dtype in ['int64', 'float64']:
                    df['price'] = df['price'].apply(lambda x: f"${int(x):,}" if pd.notnull(x) else "")
                
                # Display the enhanced dataframe
                msg_container.dataframe(
                    df, 
                    use_container_width=True,
                    column_config={
                        # Customize certain column displays
                        col: st.column_config.TextColumn(
                            col, width="medium"
                        ) for col in df.columns if df[col].dtype == 'object' and len(df[col].astype(str).str.len()) > 0 and df[col].astype(str).str.len().max() > 20
                    }
                )
                
                # Show count of properties
                msg_container.caption(f"Showing {len(df)} properties")
                
            except Exception as e:
                msg_container.error(f"Error displaying property table: {e}")
                msg_container.dataframe(df)  # Fallback to simple display