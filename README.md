# Property Search Chatbot

A Streamlit application that lets you search real estate listings using natural language and interactive filters.

## Features

- Upload CSV files with property listings
- Search properties with natural language questions
- Filter properties by price, bedrooms, bathrooms, and more
- View matching properties in an interactive table

## Project Structure

- **real_estate_data.csv** - Dataset file
- **app.py** - Main backend script
- **README.md** - Project documentation
- **.env** - Environment variables (contains GEMINI_API_KEY)
- **.gitignore** - Git ignore file (includes .env)
- **chroma_db/** - Directory for ChromaDB persistent storage

## Setup
1. Clone the repository:
```
git clone https://github.com/deepikaksr/property-search-chatbot.git
cd property search chatbot
```

2. Install requirements:
```
pip install streamlit pandas sqlite3 python-dotenv google-generativeai chromadb numpy
```

3. Create a `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

4. Run the application:
```
streamlit run app.py
```

## Usage

1. Upload your property CSV data using the sidebar
2. Use the filter sliders to narrow down properties
3. Or type questions like "Show me 3-bedroom houses under $100,000"
4. View the results in the chat interface

## Data Format

Works best with CSV files containing columns like price, bed, bath, status, city, street, house_size, acre_lot, zip_code
