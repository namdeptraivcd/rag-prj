import bs4 #1
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.config import Config

cfg = Config()


def read_source(web_paths):
    bs4_strainer = bs4.SoupStrainer(class_ =("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths = web_paths,
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    return docs


def split_text(docs, chunk_size = cfg.chunk_size, chunk_overlap= cfg.chunk_overlap, add_start_index=True):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap= chunk_overlap,
        add_start_index= add_start_index,
    )
    splited_texts = text_splitter.split_documents(docs)
    return splited_texts


