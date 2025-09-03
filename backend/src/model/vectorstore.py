import faiss
import bs4 
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from src.utils.utils import read_source, split_text
from src.utils.helper_functions import replace_t_with_space
from src.utils.hype_embedder import HyPEEmbedder
from src.config.config import Config

cfg = Config()


class Vector_store:
    def __init__ (self):
        self.vector_store = []
        self.embeddings = cfg.embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap, length_function=len
        ) # Split documents into chunks
        self.hype_embedder = HyPEEmbedder()
        
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
        
        print("documents after load:", len(documents)) # Debug
        
        chunks = self.text_splitter.split_documents(documents)
        print("chunks after split:", len(chunks)) # Debug
        chunks = replace_t_with_space(chunks) # Clean chunks
        print("chunks after clean:", len(chunks)) # Debug
        
        # Create vector store
        vector_store = None
        if cfg.enable_hype == False:
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            
        else: 
            # Generate embeddings for the first chunk to get the dimension
            first_chunk = chunks[0]
            question_embeddings = self.hype_embedder.generate_hypothetical_prompt_embeddings(first_chunk.page_content)
            vector_store = FAISS(
                embedding_function=cfg.embeddings,
                index=faiss.IndexFlatL2(len(question_embeddings[0])),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
    
            for _, chunk in enumerate(tqdm(chunks, desc="HyPE embedding")):
                # QUESTION: page_content?
                question_embeddings = self.hype_embedder.generate_hypothetical_prompt_embeddings(chunk.page_content)
                
                # Pair the chunk's content with each generated embedding question
                chunks_with_embedding_questions = [
                    (chunk.page_content, qe) for qe in question_embeddings
                ]
                
                vector_store.add_embeddings(chunks_with_embedding_questions)
        
        self.vector_store.append(vector_store)