import bs4 
from src.config.config import Config
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from src.utils.utils import read_source, split_text
from src.utils.helper_functions import replace_t_with_space
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader

cfg = Config()


class Vector_store:
    def __init__ (self):
        self.vector_store = []
        self.embeddings = cfg.embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap, length_function=len
        ) # Split documents into chunks
        
        # Load datasets
        self.__load_web(cfg.web_data_paths)
        self.__load_pdf(cfg.pdf_data_path)

    def __load_web(self, path):
        # Load webs' documents
        bs4_strainer = bs4.SoupStrainer(class_ =("post-title", "post-header", "post-content"))
        loader = WebBaseLoader(
            web_paths = path,
            bs_kwargs={"parse_only": bs4_strainer},
        )
        documents = loader.load()
        
        texts = self.text_splitter.split_documents(documents)
        cleaned_texts = replace_t_with_space(texts)
        
        # Create vector store
        vector_store = FAISS.from_documents(cleaned_texts, self.embeddings)
        
        self.vector_store.append(vector_store)

    def __load_pdf(self, path):        
        # Load PDF documents
        loader = PyPDFLoader(path)
        documents = loader.load()
        
        texts = self.text_splitter.split_documents(documents)
        cleaned_texts = replace_t_with_space(texts)
        
        # Create vector store
        vector_store = FAISS.from_documents(cleaned_texts, self.embeddings)
        
        self.vector_store.append(vector_store)