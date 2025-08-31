from src.config.config import Config
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from src.utils.utils import read_source, split_text
cfg = Config()


class Vector_store:
    def __init__ (self):
        embeddings = cfg.embeddings
        client = QdrantClient(":memory:")
        vector_size = len(embeddings.embed_query("sample text"))

        if not client.collection_exists("test"):
            client.create_collection(
                collection_name ="test",
                vectors_config = VectorParams(size= vector_size, distance= Distance.COSINE)
            )

        self.vector_store = QdrantVectorStore(
            client = client,
            collection_name = "test",
            embedding = embeddings)    

    def load_web(self, web_path):
        docs = read_source(web_path)
        splited_texts = split_text(docs)
        self.vector_store.add_documents(splited_texts)
        return self.vector_store    
