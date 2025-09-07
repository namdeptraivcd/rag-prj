import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub


class Config():
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # API keys settings
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
        
        # Milvus API keys settings
        self.milvus_uri = os.getenv("MILVUS_URI", "")
        self.milvus_token = os.getenv("MILVUS_TOKEN", "")
        
        # LLMs and Embedder settings
        '''self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")'''
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.prompt = hub.pull("rlm/rag-prompt")
        
        # Chunk splitter settings
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.embedding_dim = len(self.embeddings.embed_query("test"))
        
        # Graph RAG with Milvus vector database settings
        self.enable_graph_rag = False
        self.graph_rag_tartget_degree = 1 # Degree of graph expansion (for most cases, 1 or 2 are enough)
        self.graph_rag_top_k_entites_or_relations = 3 # Number of entities/relations to retrieve
        self.graph_rag_final_top_k_chunks = 2  # Number of final passages to return
        
        # HyPE settings
        self.enable_hype = False
        
        # HyDE settings
        self.enable_hyde = True
        
        # Query transformations settings
        self.enable_rewrite_query = False
        self.enable_generate_step_back_query = False
        self.enable_decompose_query = False
        
        # Datasets
        self.web_data_paths = ("https://lilianweng.github.io/posts/2023-06-23-agent/",)
        self.pdf_data_path = "data/Understanding_Climate_Change.pdf"
        self.csv_data_path = "data/customers-100.csv"
        self.results_path = "data/experiment_results/evaluation_results"
        
        # Experiment settings
        self.experiment_questions = [
            # Chapter 1
            "What is climate change?",
            "What human activities have contributed most to climate change?",
            "What historical cycles of glacial advance and retreat occurred in the past 650,000 years?",
            "What event marked the beginning of the modern climate era?",
            "Which scientific methods are used to study past climate conditions?",

            # Chapter 2
            "What is the greenhouse effect?",
            "Which gases are the main greenhouse gases?",
            "Why is coal considered the most carbon-intensive fossil fuel?",
            "How does oil consumption contribute to climate change?",
            "Why is natural gas called a 'bridge fuel'?",
            "What role do forests play in climate regulation?",
            "How does deforestation contribute to climate change?",
            "Why are tropical rainforests important for carbon storage?",
            "What is the role of boreal forests in climate regulation?",
            "How does agriculture contribute to climate change?",
            "What greenhouse gas is mainly produced by livestock?",
            "How does rice cultivation generate methane?",
            "Why are synthetic fertilizers harmful to the climate?",
        ]
