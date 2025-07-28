import pickle
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

def load_vector_store(path='retriever/vector_store.pkl'):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['index'], data['texts'], SentenceTransformer(data['model_name'])

def retrieve_context(query, index, texts, model, k=2):
    query_vector = model.encode([query])
    D, I = index.search(query_vector, k)
    return "\n".join([texts[i] for i in I[0]])

def load_generator(model_name="google/flan-t5-base"):
    return pipeline("text2text-generation", model=model_name)


def generate_answer(query, context, generator):
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    output = generator(prompt, max_new_tokens=150)[0]["generated_text"]
    return output.strip()

