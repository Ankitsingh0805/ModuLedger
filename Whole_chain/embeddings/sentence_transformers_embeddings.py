from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os 


load_dotenv()

class HFEmbeddings:
    def __init__(self, model: str):
        self.model = SentenceTransformer(model)
        
    def embed(self, text: str) -> list[float]:
        return self.model.encode(text)