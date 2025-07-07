import numpy as np
import ollama
from typing import List

client = ollama.Client()
embedding_model = "deepseek-r1:8b"


def get_embedding(text: str) -> List[float]:
    response = client.embeddings(model=embedding_model, prompt=text)
    return response["embedding"]

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

def stream_ollama_response(prompt: str, model: str = "deepseek-r1:8b") -> str:
    result = ""
    for chunk in client.generate(model=model, prompt=prompt, stream=True):
        result += chunk.get("response", "")
    return result.strip()
