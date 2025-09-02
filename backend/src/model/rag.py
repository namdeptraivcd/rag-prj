from src.config.config import Config
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from src.utils.helper_functions import retrieve_context_per_question
from src.model.vectorstore import Vector_store
from src.utils.query_transformer import QueryTransformer

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

            # Debug 
            '''debug_index = 0
            import os
            file_name = os.path.basename(__file__)
            print(f"\n### Start debug {debug_index} in {file_name}")
            print(f"Title: {retrieved_docs[0]}")
            print(f"### End debug {debug_index} in {file_name}\n")'''

            return (retrieved_docs_text, retrieved_docs)
        return retrieve


class RAG:
    def __init__(self):
        self.prompt = cfg.prompt
        self.llm = cfg.llm
        self.state = MessagesState()
        self.vector_store = Vector_store()
        self.retrieve = make_retrieve_tool(self.vector_store.vector_store)
        self.query_transformer = QueryTransformer()
        
        # Conversation history
        self.state["messages"] = []
        self.system_prompt = (
            "You are a retrieval-augmented assistant. "
            "ALWAYS use the retrieve tool to get context from the provided documents before answering any user question. "
            "Do not answer from your own knowledge unless the tool result is empty."
        )
        self.state["messages"].insert(0, SystemMessage(content=self.system_prompt))
        
    def query_or_respond(self):
        llm_with_tools = self.llm.bind_tools([self.retrieve])
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

    def generate(self):
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
            if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls):
                conversation_messages.append(message)
                
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        response = self.llm.invoke(prompt)
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
        
        