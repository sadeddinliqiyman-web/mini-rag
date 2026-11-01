# Mini RAG Assistant
Python ilə yazılmış sadə RAG(Retrieval-Augmented Generation) tətbiqi. Mətn sənədləri əsasənda suallara cavab verir.
## Texnologiyalar
- Python 3.10+
- sentence-transformers (embedding)
- FAISS (sematik axtarış)
- OpenAI API (cavab yaratmaq üçün, ist'y' bağlı)
## Arxitektura
1. Yükləmə: docs/ qovluğundakı .txt fayllar oxunurş
2. Chunking: mətn 120 sözlükç 30 söz üst-üstə düşən hissələrə bölünür.
3. Embedding + FAISS: hər hissə rəqəm vektoruna çevrilir ve FAISS indeksinə yazılır.
4. Axtarış: sual vektora çevrilir, ən yaxın 3 hissə tapılır.
5. LLM: tapılan hissələr kontekst kimi dil modelinə verilir, cavab yaranır.
6. Yoxlama: python app.py --test bir neçə dəfə sualla sınaq edir.
##Quraşdırma
'''

## İşə salma
'''
set OPENAI_API_KEY=sizin_acar
python app.py
python app.py --test
'''
API açarı olmasa proqram yalnız tapılan ən uyğun hissələri göstərir.
## Qeyd
Öz sənədlərinizi docs\ qovluğuna əlavə edin.