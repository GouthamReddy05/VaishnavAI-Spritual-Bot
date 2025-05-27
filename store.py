import os
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

# 384 is the dimension it is inbuilt if we use all-MiniLM
# This creates a FAISS index for doing fast similarity searches using L2 (Euclidean) distance.
# You store vectors in it, and later you can search for the most similar ones.

def build_index(base_path="Main_f"):
    model = SentenceTransformer('intfloat/e5-large-v2')
    embedding_dim = model.get_sentence_embedding_dimension()
    index1 = faiss.IndexFlatL2(embedding_dim)
    index2 = faiss.IndexFlatL2(embedding_dim)
    metadata1 = []
    metadata2 = []

    canto_f = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and f.startswith('Canto')]

    for canto_no in canto_f:
        canto_path = os.path.join(base_path, canto_no)
        number = canto_no.split('_')[-1] 
        print(f"📁 Processing {canto_no}....")

        chapter_f = [f for f in os.listdir(canto_path) if f.endswith('.json')]

        for chapter_file in chapter_f:
            chapter_no = os.path.split(chapter_file)[0]
            chapter_path = os.path.join(canto_path, chapter_file)
            print(f"  📖 Processing {chapter_file}...")

            with open(chapter_path, 'r') as f:
                data = json.load(f)

            for verse in data:
                verse_id = verse.get('verse_id')
                text = verse.get('text')

                if not verse_id or not text:
                    print(f"    ⚠️ Skipping a verse in {chapter_file} (missing verse_id or text)")
                    continue

                emb = model.encode(text)
                index1.add(np.array([emb], dtype=np.float32))
                metadata1.append({
                    "source" : "bhagavatam",
                    "canto_no": number,
                    "chapter_no": chapter_no,
                    "verse_id": verse_id,
                    "text": text
                })

    print(f"✅ Indexed {len(metadata1)} verses.")

    # Save FAISS index
    faiss.write_index(index1, "bhagavatam_faiss.index")

    # Save metadata to JSON
    with open("bhagavatam_metadata.json", "w") as f:
        json.dump(metadata1, f, indent=2)

    print("✅ FAISS index and metadata saved.")


    with open("Valmiki_Ramayan_Shlokas.json", 'r') as file:
        ramayan = json.load(file)

    # Extract all texts
    texts = [verse['transliteration'] for verse in ramayan if verse.get('transliteration')]

    # Batch encode
    print("Encoding shlokas... (this will take time)")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    # Add to FAISS index and metadata
    i = 0
    for verse in ramayan:
        text = verse.get('transliteration')
        if not text:
            continue

        emb = embeddings[i]
        index2.add(np.array([emb], dtype=np.float32))

        metadata2.append({
            "source" : "ramayan",
            "kanda": verse.get('kanda'),
            "sarga": verse.get('sarga'),
            "shloka_id": verse.get('shloka'),
            "text": text
        })
        i += 1




    print(f"✅ Indexed {len(metadata2)} verses.")

    # Save FAISS index
    faiss.write_index(index2, "ramayan_faiss.index")

    # Save metadata to JSON
    with open("ramayan_metadata.json", "w") as f:
        json.dump(metadata2, f, indent=2)

    print("✅ FAISS index and metadata saved.")



build_index()





## pip install googletrans==4.0.0-rc1

## from googletrans import Translator

# translator = Translator()

# hindi_text = "तपस्स्वाध्यायनिरतं तपस्वी वाग्विदां वरम्"
# translated = translator.translate(hindi_text, src='hi', dest='en')
# print(translated.text)
