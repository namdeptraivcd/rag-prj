from typing import List
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from src.utils.helper_functions import retrieve_context_per_question
from src.model.vectorstore import Vector_store
from src.utils.query_transformer import QueryTransformer
from src.utils.hype_embedder import HyPEEmbedder
from src.model.graph_rag_processor import GraphRAGProcessor
from src.datasets.load_graph_rag_dataset import load_graph_rag_dataset
from src.config.config import Config

cfg = Config()


def make_retrieve_tool(vector_store):
        @tool(response_format="content_and_artifact")
        def retrieve(query):
            """
            Retrieve relevant documents from the vector store based on the query.
            """
            retrieved_docs = []
            for vs in vector_store:
                for qr in query:
                    tmp_retrieved_doc = vs.similarity_search(qr, k=4) # Limit to 4 docs per store
                    retrieved_docs.extend(tmp_retrieved_doc) 
            
            # If more than 10 docs total, keep only the top 10
            if len(retrieved_docs) > 10:
                retrieved_docs = retrieved_docs[:10]
        
            retrieved_docs_text = "\n\n".join(
                f"Source: {retrieved_doc.metadata}, Content: {retrieved_doc.page_content}"
                for retrieved_doc in retrieved_docs
            )

            return (retrieved_docs_text, retrieved_docs)
        return retrieve


class RAG:
    def __init__(self):
        self.state = MessagesState()
        self.vector_store = Vector_store()
        self.retrieve = make_retrieve_tool(self.vector_store.vector_store)
        self.query_transformer = QueryTransformer()
        self.graph_rag_processor = None
        
        if cfg.enable_graph_rag == True:
            self.__set_up_graph_rag()
        
        if cfg.enable_hype == True:
            self.hype_embedder = HyPEEmbedder()
            
        # Conversation history
        self.state["messages"] = []
        self.system_prompt = (
            "You are a retrieval-augmented assistant. "
            "ALWAYS use the retrieve tool to get context from the provided documents before answering any user question. "
            "Do not answer from your own knowledge unless the tool result is empty."
        )
        self.state["messages"].insert(0, SystemMessage(content=self.system_prompt))
        
    def __set_up_graph_rag(self):
        # Load dataset
        dataset = load_graph_rag_dataset()
        
        self.graph_rag_processor = GraphRAGProcessor(dataset)
    
    def query(self, query: str):
        self.state["messages"].append(HumanMessage(content=query))
        if not cfg.enable_graph_rag:
            self.__normal_query(query)
        else:
            self.__graph_rag_query(query)
    
    def __normal_query(self, query):
        # Retrive documents or diricly answer with llm knowledge
        message, answer_type = self.__retrieve_or_respond()
        self.state["messages"].extend(message)
        self.state["answer_type"] = answer_type
        
        # Generate answer
        self.__generate() 
    
    def __graph_rag_query(self, query) -> List[str]:
        self.state["answer_type"] = "rag"
        
        query_entities = self.__entity_extraction(query)
        retrieved_docs = self.graph_rag_processor.query_graph_rag(
            query=query,
            query_entities=query_entities,
        )
        
        self.__generate(retrieved_docs)
    
    def __entity_extraction(self, query: str) -> list:
        # @TODO: generalize this method
        entities = []
        keywords = ["Euler", "Bernoulli", "Jakob", "Johann", "Daniel", "Basel"]
        for keyword in keywords:
            if keyword.lower() in query.lower():
                entities.append(keyword)
        return entities
        
    def __retrieve_or_respond(self): 
        llm_with_tools = cfg.llm.bind_tools([self.retrieve])
        response = llm_with_tools.invoke(self.state["messages"])
        # If there are tool calls, execute tools and create ToolMessage
        tool_messages = []
        tool_calls = getattr(response, "tool_calls", None)
        
        if tool_calls:
            answer_type = "rag"
            
            for tool_call in tool_calls:
                # Execute tool with query
                retrieved_docs_text = self.retrieve.invoke({"query": self.tranform_query(tool_call["args"]["query"])})
                
                tool_message = ToolMessage(
                    content=retrieved_docs_text,
                    name="retrieve",
                    tool_call_id=tool_call["id"]  
                )
                tool_messages.append(tool_message)
                
        else:
            answer_type = "llm"

        # Return both AIMessage and ToolMessage (if any)
        return [response] + tool_messages, answer_type

    def __generate(self, retrieved_docs: List[str] = None): #Debug retrieved_docs default = None
        retrieved_docs_text = ""
        if cfg.enable_graph_rag:
            retrieved_docs_text = "\n\n".join(retrieved_docs) # Retrieved docs
            
            # Debug 
            '''debug_index = 0
            import os
            file_name = os.path.basename(__file__)
            print(f"\n### Start debug {debug_index} in {file_name}")
            print(retrieved_docs_text)
            print(f"### End debug {debug_index} in {file_name}\n")'''
        
        else:
            tool_messages = []
            for message in reversed(self.state["messages"]):
                if message.type == "tool":
                    tool_messages.append(message)
                else:
                    break
            tool_messages = tool_messages[::-1] # Reverse

            retrieved_docs_text = "\n\n".join(tool_message.content for tool_message in tool_messages) # Retrieved docs

        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{retrieved_docs_text}"
        )

        conversation_messages = []
        for message in self.state["messages"]:
            if message.type == "human" or (message.type == "ai" and not message.tool_calls):
                conversation_messages.append(message)
                
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        response = cfg.llm.invoke(prompt)
        self.state["answer"] = response.content

    def tranform_query(self, original_query):
        transformed_queries = [original_query]
        
        # Apply query transformations
        if cfg.enable_rewrite_query:
            rewritten_query = self.query_transformer.rewrite_query(original_query)
            transformed_queries.append(rewritten_query)
        
        if cfg.enable_generate_step_back_query:
            step_back_query = self.query_transformer.generate_step_back_query(original_query)
            transformed_queries.append(step_back_query)
        
        if cfg.enable_decompose_query:
            sub_queries = self.query_transformer.decompose_query(original_query)
            for sub_query in sub_queries:
                transformed_queries.append(sub_query)
        
        return transformed_queries
        
        