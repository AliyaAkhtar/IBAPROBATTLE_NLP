import json
import requests
from pypdf import PdfReader
from rank_bm25 import BM25L
import tiktoken
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq
import streamlit as st
import re
import time

# Cache JSON Data for Faster Access
@st.cache_data
def load_json_data(json_file):
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    return [
        f"{item['name']} - {item['faculty']} ({item['days']} at {item['start_time']})"
        for item in data
    ]

# Download & Extract PDF
def download_pdf(url, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):  
            file.write(chunk)

def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join(filter(None, [page.extract_text() for page in reader.pages]))

# Download PDF Once
pdf_url = "https://www.iba.edu.pk/News/pa-2024-25.pdf"
pdf_path = "pa-2024-25.pdf"
download_pdf(pdf_url, pdf_path)
pdf_text = load_pdf_text(pdf_path)

# Create FAISS Index with Optimized Model
@st.cache_resource
def create_faiss_index(texts):
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts, model)

# Load Data into FAISS & BM25L
json_texts = load_json_data("courses_info.json")
all_texts = json_texts + [pdf_text]
faiss_index = create_faiss_index(all_texts)
bm25_index = BM25L([text.split() for text in all_texts])

# Improved Jailbreak Detection
def is_jailbreak_query(query):
    blocked_patterns = [r"\bhack\b", r"\bexploit\b", r"\bypass\b", r"\billegal\b", r"\bcheat\b", r"\bcrack\b", r"\bleak\b"]
    return any(re.search(pattern, query.lower()) for pattern in blocked_patterns)

import numpy as np

def hybrid_search(query, faiss_index, texts, k=3, weight_faiss=0.5, weight_bm25=0.5):
    """Hybrid Search with Ranked FAISS & BM25L Results"""

    # FAISS Search: Get documents with scores
    faiss_results = faiss_index.similarity_search_with_score(query, k=k)
    faiss_docs = [doc[0].page_content for doc in faiss_results]
    faiss_scores = np.array([doc[1] for doc in faiss_results])

    # BM25 Search: Get scores
    bm25_scores = bm25_index.get_scores(query.split())
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
    bm25_docs = [texts[i] for i in top_bm25_indices]
    bm25_top_scores = np.array([bm25_scores[i] for i in top_bm25_indices])

    # Normalize scores between 0-1
    if len(faiss_scores) > 0:
        faiss_scores = (faiss_scores - faiss_scores.min()) / (faiss_scores.max() - faiss_scores.min() + 1e-8)
    if len(bm25_top_scores) > 0:
        bm25_top_scores = (bm25_top_scores - bm25_top_scores.min()) / (bm25_top_scores.max() - bm25_top_scores.min() + 1e-8)

    # Create a combined results dictionary
    combined_results = {}

    # Store FAISS results
    for doc, score in zip(faiss_docs, faiss_scores):
        combined_results[doc] = combined_results.get(doc, 0) + weight_faiss * score

    # Store BM25 results
    for doc, score in zip(bm25_docs, bm25_top_scores):
        combined_results[doc] = combined_results.get(doc, 0) + weight_bm25 * score

    # Sort results by highest score
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)

    # Extract and return only the documents
    ranked_docs = [doc[0] for doc in sorted_results]
    
    return "\n\n".join(ranked_docs[:k]) if ranked_docs else None

# Efficient Text Truncation
def truncate_text(text, max_tokens=1000):
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(enc.encode(text)[:max_tokens])

# Initialize Groq client securely
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# Query Relevance Check
def is_query_relevant(query, context):
    """Check if query words appear in retrieved context"""
    query_words = set(query.lower().split())
    context_words = set(context.lower().split())
    return bool(query_words & context_words)

# Optimized Response Generation
def generate_response(query, index, texts, max_chars=4000):
    start_time = time.time()
    context = hybrid_search(query, index, texts)
    if not context:
        return None, "No relevant context found."

    context = context[:max_chars]  

    if not is_query_relevant(query, context):
        return context, "I cannot answer this question based on the provided information."

    system_prompt = (
        "You are an AI assistant. If the query involves hacking, bypassing security, "
        "or anything illegal, respond with 'I cannot answer this question.'"
    )

    prompt = f"Use the following context to answer the query:\n\n{context}\n\nQuery: {query}\nAnswer:"

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model="Gemma2-9b-It",
            max_tokens=512,  # Reduced token count for faster responses
            temperature=0.2,  # More deterministic responses
        )

        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        return context, f"{chat_completion.choices[0].message.content}\n\n(Response Time: {response_time}s)"

    except Exception as e:
        return context, f"Error: {str(e)}"

# Streamlit UI with Performance Improvements
st.title("🚀 Fast RAG Pipeline with FAISS + BM25L + Groq")

query = st.text_input("Enter your query:")

if st.button("Get Answer"):
    if query:
        with st.spinner("Fetching response..."):
            retrieved_context, response = generate_response(query, faiss_index, all_texts)

        st.write("### Retrieved Context:")
        st.code(retrieved_context if retrieved_context else "No relevant context found.", language="markdown")

        st.write("### Response:")
        st.write(response)
    else:
        st.warning("Please enter a query.")
