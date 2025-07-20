import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from openai import OpenAI
from together import Together

os.environ["TOKENIZERS_PARALLELISM"] = "false"



load_dotenv()

client = Together(api_key = os.getenv('api_key'))  # Set your Together API key

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




def search_faiss(query, tok_k = 5, scripture='bhagavatam'):

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


def ask_llm(query, context, verses):
    if context:
        prompt = f"""
Here is some context and some verses that may help answer the question:
{context}

Question: {query}
# Instructions:
# - {verses}\nChoose only one verse from the given verses that best answers the question.
# - Mention which verse (e.g., Kanda X, Sarga Y, Shloka Z) you are using.
# - Then explain the answer briefly and clearly.
# - If none of the verses are directly relevant, answer based on your own knowledge.


You are an expert on ancient Indian scriptures, especially the Ramayana.
if context is not relevant, answer based on your knowledge of the query.
Answer:"""
    else:
        prompt = f"""
Answer the following question based on your knowledge:

Question: {query}

Answer:"""


    try:
        
        response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
    # max_tokens=200,
)

        return response.choices[0].message.content.strip()


    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"



query = "Why did Lord Krishna lift the Govardhana Hill, and what is the spiritual significance of this event in the Bhagavatam?"

context = search_faiss(query, tok_k=5, scripture='bhagavatam')
# verses = build_context(context)

# ans = ask_llm(query, context, verses)

# print(ans)