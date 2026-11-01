""" MINI RAG Assistant
Addimlar: sened yukle -> chunk-lara bol -> embedding yarat -> LLM ile cavab ver
"""
import glob
import os
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
DOCS_DIR = "docs"           #senedlerin oldugu qovluq
CHUNK_SIZE = 120            #bir chunk-da sozlerin sayi
CHUNK_OVERLAP = 30          #chunk-lar arasi ortaq sozler
TOP_K = 3                   #axtarisda nece chunk qaytarilsin
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = os.getenv("LLM_MODEL","gbt-4o-mini")
# 1. Senedlerin yuklenmesi
def load_documents(paths):
    docs = []
    for path in paths:
        with open(path,"r", encoding="utf-8") as f:
            docs.append({"source": os.path.basename(path),"text": f.read()})
    return docs
# 2. Metnin emali(chunking)
def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    step = size-overlap
    for start in range(0, len(words),step):
        piece = words[start:start + size]
        if piece:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks
def build_chunks(docs):
    all_chunks = []
    for d in docs:
        for c in split_text(d["text"]):
            all_chunks.append({"source": d["source"], "text":c})
    return all_chunks
# 3. Embedding ve FAISS indeksi
def build_index(chunks, model):
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True)
    vectors = np.array(vectors, dtype="float32")
    
    # --- DÜZƏLİŞ: Əgər vektor 1D-dirsə, onu 2D matrisə çeviririk ---
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
        
    index = faiss.IndexFlatIP(vectors.shape[1]) # kosinus oxşarlığı üçün dimensiya ölçüsü
    index.add(vectors)
    return index
def retrieve(question, model, index, chunks, k=TOP_K):
    if not chunks:
        return []
    q = model.encode([question], normalize_embeddings=True)
    q = np.array(q, dtype="float32")
    k = max(1, min(k, len(chunks)))  # k-nın ən azı 1 olmasını təmin edirik
    scores, ids = index.search(q, k)
    return [(chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

# 4. LLM inteqrasiyasi
def generate_answer(question, results):
    context = "\n\n".join([
        f"[{c['source']}] {c['text']}" for c, _ in results
    ])
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return("(API acari yoxdur, LLM cavabi verilmir."
               "Tapilan uygun hisseler yuxarida gosterilib.)")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    prompt = (
        "Asagidaki kontekste esasen suala cavab ver. "
        "Cavab kontekstde yoxdursa, 'Senedlerde bu melumat yoxdur' de."
        f"Kontekst: \n{context}\n\nSual: {question}\ncavab:"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role":"user","content": prompt}],
        temperature = 0,
    )
    return resp.choices[0].message.content

def ask(question, model, index, chunks):
    results = retrieve(question, model, index, chunks)
    print("\n--- Tapilan hisseler ---")
    for c, score in results:
        print(f"[{c['source']}] (benzerlik: {score:.2f}) {c['text'][:150]}...")
    print("\n--- Cavab ---")
    print(generate_answer(question, results))

# 5. Test Suallari
TEST_QUESTIONS = [
    "What is RAG?",
    "What is a chunk and why do we use it?",
    "What is FAISS used for?",
]
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    paths = args or sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt")))
    if not paths:
        print("docs/ qovluguna .txt fayl elave edin.")
        return
    print("Senedler yuklenir:", paths)
    docs = load_documents(paths)
    chunks = build_chunks(docs)
    print(f"{len(chunks)} chunk yaradildi. Model yuklenir...")
    model = SentenceTransformer(EMBED_MODEL)
    index = build_index(chunks, model)

    if "--test" in sys.argv:
        for q in TEST_QUESTIONS:
            print("\n==========\nSual:",q)
            ask(q,model,index,chunks)
        return

    print("Hazirdir! Sual yazin(cixmaq ucun 'exit').")
    while True:
        q= input("\nSual: ").strip()
        if q.lower() in ("exit","quit","q"):
            break
        if q:
            ask(q, model, index, chunks)

if __name__ == "__main__":
    main()