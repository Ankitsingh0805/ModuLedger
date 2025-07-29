from Whole_chain.embeddings import OpenAIEmbeddings
from Whole_chain.vector_stores import QdrantVectorStore
from Whole_chain.llms.llms import GroqLLM
from Whole_chain.tools.file_reader import DocumentReader
import os
from dotenv import load_dotenv
from Whole_chain.stt.stt import GroqSTT

def main(if_audio_input: bool = False):
    load_dotenv()
    
    # Initialize the LLM
    llm = GroqLLM("llama3-8b-8192")

    # Initialize the embeddings
    embeddings = OpenAIEmbeddings("text-embedding-3-small")

    # Initialize the vector store
    vector_store = QdrantVectorStore(":memory:")

    # Initialize the document reader
    document_reader = DocumentReader()

    file_path = "data/whole_rag.pdf"
    text = document_reader.read(file_path)

    text_embedding= embeddings.embed(text)
    vector_store.add(text, text_embedding)

    text_query = "What is the insight of the doc?"

    if if_audio_input:
        stt = GroqSTT()
        audio_query = stt.transcribe("data/whole_rag.mp3")
        query_embedding= embeddings.embed(audio_query)
    else:
        query_embedding = embeddings.embed(text_query)
    relevant_texts = vector_store.query(query_embedding, k=3)

    context = "\n".join(relevant_texts)
    prompt = f"Based on the following context, answer the question: {text_query}\n\nContext:\n{context}"
    response = llm.generate(prompt)
    print("Response:", response)
    print(f"Query: {text_query}")

if __name__ == "__main__":
    main(if_audio_input=False)  # Set to True if you want to use audio input
    