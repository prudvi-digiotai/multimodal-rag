import sys
import pysqlite3

sys.modules["sqlite3"] = pysqlite3

import streamlit as st
import os
from wyge.prebuilt_agents.multimodal_rag import MultiModalRAG
import pandas as pd
from io import BytesIO
from PIL import Image
import base64
import tempfile

st.set_page_config(page_title="MultiModal RAG Demo", layout="wide")

def get_temp_file_path(uploaded_file):
    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    return None

def initialize_session_state():
    if 'rag' not in st.session_state:
        st.session_state['rag'] = None
    if 'processed_file' not in st.session_state:
        st.session_state['processed_file'] = None

def display_result(result):
    if result["ids"].startswith("image_"):
        # Display image
        image_data = base64.b64decode(result["content"])
        st.image(Image.open(BytesIO(image_data)))
        st.write(f"Image Summary: {result['metadata'].get('summary', 'No summary available')}")
        
    elif result["ids"].startswith("table_"):
        # Display table
        st.write("Table Content:")
        try:
            # Split rows and create DataFrame
            rows = [row.strip().split(',') for row in result["content"].split('\n')]
            df = pd.DataFrame(rows)
            st.dataframe(df)
        except:
            st.text(result["content"])
        
    else:
        # Display text
        st.write("Text Content:")
        st.text(result["content"])

def main():
    st.title("MultiModal RAG Demo")
    initialize_session_state()

    # API Key input
    api_key = st.text_input("Enter OpenAI API Key:", type="password")
    if not api_key:
        st.warning("Please enter your OpenAI API key to proceed.")
        return

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file:
        # Get total pages (you might want to add a function to get this)
        col1, col2 = st.columns(2)
        with col1:
            start_page = st.number_input("Start Page", min_value=1, value=1)
        with col2:
            end_page = st.number_input("End Page", min_value=start_page, value=start_page + 10)

        # Process PDF button
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                try:
                    # Initialize RAG
                    st.session_state['rag'] = MultiModalRAG(api_key)
                    
                    # Get temporary file path
                    temp_file_path = get_temp_file_path(uploaded_file)
                    print(1)
                    # Process the PDF
                    st.session_state['rag'].process_pdf(temp_file_path, (start_page, end_page))
                    print(2)
                    st.session_state['processed_file'] = uploaded_file.name
                    
                    # Clean up temporary file
                    os.unlink(temp_file_path)
                    print(3)
                    st.success("PDF processed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    return

        # Query section with new tabs
        if st.session_state['rag'] and st.session_state['processed_file']:
            st.write(f"Currently processed file: {st.session_state['processed_file']}")
            
            # Add tabs for different search modes
            search_mode = st.radio("Select Search Mode:", ["Raw Search", "QA Search"])
            
            query = st.text_input("Enter your query:")
            top_k = st.slider("Number of results to retrieve", min_value=1, max_value=10, value=3)
            
            if query and st.button("Search"):
                with st.spinner("Searching..."):
                    try:
                        if search_mode == "Raw Search":
                            # Original raw search
                            results = st.session_state['rag'].query(query, top_k=top_k)
                            
                            # Display results
                            st.subheader("Search Results")
                            for i, result in enumerate(zip(results["ids"], results["content"], results["metadata"]), 1):
                                with st.expander(f"Result {i}"):
                                    display_result({
                                        "ids": result[0],
                                        "content": result[1],
                                        "metadata": result[2]
                                    })
                        else:
                            # QA search
                            st.subheader("AI Response")
                            response, retrieved_data = st.session_state['rag'].answer_user_query(query, top_k=top_k)
                            st.write(response)
                            
                            # Display images if any were retrieved
                            st.subheader("Retrieved Images")
                            for idx, content, meta in zip(retrieved_data["ids"], retrieved_data["content"], retrieved_data["metadata"]):
                                if idx.startswith("image_"):
                                    with st.expander(f"Image"):
                                        display_result({
                                            "ids": idx,
                                            "content":content,
                                            "metadata": meta
                                        })
                                    # with st.expander(f"Image - {meta.get('summary', 'No summary')}"):
                                    #     image_data = base64.b64decode(content)
                                    #     st.image(Image.open(BytesIO(image_data)))
                            
                    except Exception as e:
                        st.error(f"Error during search: {str(e)}")

if __name__ == "__main__":
    main()
