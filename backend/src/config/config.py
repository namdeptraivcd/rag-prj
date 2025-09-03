import os 
import getpass
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain import hub


class Config():
    def __init__(self):
        #API key
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = ""

        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = ""
        
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.prompt = hub.pull("rlm/rag-prompt")
        
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # Query transformations
        self.enable_rewrite_query = True
        self.enable_generate_step_back_query = True
        self.enable_decompose_query = False
        
        self.web_data_paths = ("https://lilianweng.github.io/posts/2023-06-23-agent/",)
        self.pdf_data_path = "data/Understanding_Climate_Change.pdf"
        self.csv_data_path = "data/customers-100.csv"
        self.results_path = "data/experiments/evaluation_results"
        
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
