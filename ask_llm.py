import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env into environment variables



api_key = os.getenv('gemini_api_key')
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

emb_model = SentenceTransformer('intfloat/e5-large-v2')


index = faiss.read_index("verse_faiss.index")

with open("verse_metadata.json", 'r') as f:
    metadata = json.load(f)




def search_faiss(query, tok_k = 5):

    ## Changes the shape of the array to have 1 row and as many columns as needed
    query_emb = emb_model.encode(query).astype(np.float32).reshape(1, -1)
    D, I = index.search(query_emb, tok_k)

    ## D is numpy array [1, top_k] shape contains dist betw query vector and each of top_k nearest vectors
    ## I is numpy array [1, top_k] shape containing indices
    results = []

    for dist, idx in zip(D[0], I[0]):
        if 0 <= idx < len(metadata):
            results.append(metadata[idx])
        # else:
        #     return []

    return results 


def build_context(verses):

    if not verses:
        return ""

    context = []

    for v in verses:
        if 'canto_no' in v and 'chapter_no' in v and 'verse_id' in v:
            context.append(
                f"Canto {v['canto_no']}, Chapter {v['chapter_no']}, Verse {v['verse_id']}:\n{v['text']}"
            )
        elif 'kanda' in v and 'sarga' in v and 'shloka_id' in v:

            context.append(
                f"Kanda {v['kanda']}, Sarga {v['sarga']}, Shloka {v['shloka_id']}:\n{v['text']}"
            )
    return "\n\n".join(context)


def ask_llm(query, context):

    if context:
        prompt = f"""
Here is some context that may help answer the question:
{context}

Question: {query}

You are an expert on ancient Indian scriptures, especially the Ramayana.
Answer the following question based on your knowledge of the Ramayana and based on the given context:

Answer:"""
    else:
        prompt = f"""
Answer the following question based on your knowledge:

Question: {query}

Answer:"""

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"