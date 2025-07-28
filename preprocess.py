import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

def load_and_prepare_data():
    files = ['data/Training Dataset.csv', 'data/Testing Dataset.csv', 'data/SampleSubmission.csv']
    dataframes = [pd.read_csv(f) for f in files]
    texts = []

    for df, name in zip(dataframes, ['Train', 'Test', 'Submission']):
        df = df.astype(str)
        text = f"{name} Dataset:\n"
        for col in df.columns:
            text += f"{col}: " + " | ".join(df[col].tolist()) + "\n"
        texts.append(text)

    return texts

def embed_and_store(texts, model_name='all-MiniLM-L6-v2', save_path='retriever/vector_store.pkl'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    with open(save_path, 'wb') as f:
        pickle.dump({'index': index, 'texts': texts, 'model_name': model_name}, f)

if __name__ == "__main__":
    os.makedirs('retriever', exist_ok=True)
    docs = load_and_prepare_data()
    embed_and_store(docs)
