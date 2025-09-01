import bs4 
import pandas as pd
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
from langchain_community.document_loaders.csv_loader import CSVLoader

cfg = Config()


class Vector_store:
    def __init__ (self):
        self.vector_store = []
        self.embeddings = cfg.embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap, length_function=len
        ) # Split documents into chunks
        
        # @TODO: fix bug: if we load more than one dataset, the chat will only be able to answer questions related to one dataset
        # Load datasets
        '''self.__load_data("web", cfg.web_data_paths)'''
        self.__load_data("pdf", cfg.pdf_data_path)
        '''self.__load_data("csv", cfg.csv_data_path)'''

    def __load_data(self, data_type, path):
        # @TODO: fix bug: if we change the current website by another one, there will be a bug
        if data_type == "web":
            # Load webs' documents
            bs4_strainer = bs4.SoupStrainer(class_ =("post-title", "post-header", "post-content"))
            loader = WebBaseLoader(
                web_paths = path,
                bs_kwargs={"parse_only": bs4_strainer},
            )
            documents = loader.load()
        
        elif data_type == "pdf":
            # Load PDF documents
            loader = PyPDFLoader(path)
            documents = loader.load()
        
        elif data_type == "csv":
            # Load CSV documents
            loader = CSVLoader(file_path=path)
            documents = loader.load()
            
        else:
            raise NotImplementedError("Data type not supported")
        
        texts = self.text_splitter.split_documents(documents)
        cleaned_texts = replace_t_with_space(texts)
        
        # Create vector store
        vector_store = FAISS.from_documents(cleaned_texts, self.embeddings)
        
        self.vector_store.append(vector_store)