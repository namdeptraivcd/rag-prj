from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from src.utils.utils import split_text, read_source

class Vector_store:
    def __init__ (self, web_path):

        embeddings = OpenAIEmbeddings()
        docs = read_source(web_path)
        splited_texts = split_text(docs)
        vector_store = FAISS.from_documents(splited_texts, embeddings)
        return vector_store