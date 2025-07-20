import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from openai import OpenAI
# from together import Together
import ollama

os.environ["TOKENIZERS_PARALLELISM"] = "false"   # to remove warnings



load_dotenv()

# client = Together(api_key = os.getenv('api_key'))  # Set your Together API key

emb_model = SentenceTransformer("intfloat/e5-large-v2")


def load_index_and_metadata(scripture):
    if scripture == 'Ramayana':
        index_path = 'ramayan_faiss.index'
        metadata_path = 'ramayan_metadata.json'
    else:
        index_path = 'bhagavatam_faiss.index'
        metadata_path = 'bhagavatam_metadata.json'
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing FAISS index or metadata for {scripture}")
    index = faiss.read_index(index_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return index, metadata




def search_faiss(query, tok_k = 1, scripture='Ramayana'):

    index, metadata = load_index_and_metadata(scripture)


    ## Changes the shape of the array to have 1 row and as many columns as needed
    query_emb = emb_model.encode(query).astype(np.float32).reshape(1, -1)
    D, I = index.search(query_emb, tok_k)

    ## D is numpy array [1, top_k] shape contains dist betw query vector and each of top_k nearest vectors
    ## I is numpy array of 2D shape containing indices [[1, 2, 0]]
    results = []

    for dist, idx in zip(D[0], I[0]):
        if 0 <= idx < len(metadata):
            results.append(metadata[idx])
        # else:
        #     return []
        print(metadata[idx])

    return results 


def build_context(dictionary, scripture):

    if not dictionary:
        return ""

    bul_context = []

    for v in dictionary:
        if scripture == 'Bhagavatam':
            bul_context.append(
                f"Canto {v['canto_no']}, Chapter {v['chapter_no']}, Verse {v['verse_id']}:\n{v['text']}"
            )
        elif scripture == 'Ramayana':
            bul_context.append(
                f"Kanda {v['kanda']}, Sarga {v['sarga']}, Shloka {v['shloka_id']}:\n{v['shloka_text']}"
            )
    return "\n\n".join(bul_context)

def build_prompt(query, scripture, context=None):   

    return f"""
You are an expert on ancient Indian scriptures, especially the {scripture}. Your task is to:

1. Answer the given question clearly.
2. Use the context verses provided below to support your answer.
3. Include the verse references (e.g., Kanda, Sarga, Shloka) where the answer information appears.
4. If the context is not helpful, answer based on your knowledge.

--- Context Verses ---
{context}

--- Question ---
{query}

--- Answer Format ---
Answer: <your main answer>

Explanation: <explain how the context supports it>

Verse References: <list relevant Kanda/Sarga/Shloka, if present in the context>

--- Answer ---
"""


def ask_llm(query, scripture, context=None):
    if context:
        prompt = build_prompt(query, scripture, context)

    else:
        prompt = f"""
You are the best llm. Based on your knowledge, please answer the question below in detail.

Question: {query}

Answer:"""


    try:
        response = ollama.chat(model="llama3", messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])

        return f"{response['message']['content']}"


    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"



# query = "Why did Lord Krishna lift the Govardhana Hill, and what is the spiritual significance of this event in the Bhagavatam?"

# context = search_faiss(query, tok_k=5, scripture='bhagavatam')
# verses = build_context(context)

# ans = ask_llm(query, context, verses)

# print(ans)