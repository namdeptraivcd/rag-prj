from src.model.state import State, Search
from src.config.config import Config
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

cfg = Config()


def make_retrieve_tool(vector_store):
        @tool(response_format="content_and_artifact")
        def retrieve(query):
            """
            Retrieve relevant documents from the vector store based on the query.
            """
            retrieved_docs = vector_store.similarity_search(query)
            serialized = "\n\n".join(
                f"Source: {doc.metadata}, Content: {doc.page_content}"
                for doc in retrieved_docs
            )
            
            # Debug 
            '''debug_index = 3
            import os
            file_name = os.path.basename(__file__)
            print(f"\n### Start debug {debug_index} in {file_name}")
            for retrieved_doc in retrieved_docs:
                print(retrieved_doc)
            print(f"### End debug {debug_index} in {file_name}\n")'''
        
            return (serialized, retrieved_docs)
        return retrieve


class RAG:
    def __init__(self, vector_store):
        self.prompt = cfg.prompt
        self.llm = cfg.llm
        self.state = MessagesState()
        self.vector_store = vector_store
        self.retrieve = make_retrieve_tool(vector_store)

    '''@tool(response_format = "content_and_artifact")
    def retrieve(self, query):
        """
        Retrieve relevant documents from the vector store based on the query.
        """
        retrieved_docs = self.vector_store.similarity_search(query)
        serialized = "\n\n".join (f"Source: {doc.metadata}, Content: {doc.page_content}" 
                             for doc in retrieved_docs)

        return {"content": serialized, 
                "artifact": retrieved_docs}'''
        
    def query_or_respond (self):
        llm_with_tools = self.llm.bind_tools([self.retrieve])
        response = llm_with_tools.invoke(self.state["messages"])
        # Nếu có tool_call, thực thi tool và tạo ToolMessage
        tool_messages = []
        tool_calls = getattr(response, "tool_calls", None)
        
        if tool_calls:
            # Debug 
            debug_index = 0
            '''import os
            file_name = os.path.basename(__file__)
            print(f"\n### Start debug {debug_index} in {file_name}")
            print("LLM đã gọi tool retrieve")
            print("Thông tin tool_calls:", tool_calls)
            print(f"### End debug {debug_index} in {file_name}\n")'''
            
            for call in tool_calls:
                # Thực thi tool với query
                tool_result = self.retrieve.invoke({"query": call["args"]["query"]})
                
                # Debug 
                '''debug_index = 4
                import os
                file_name = os.path.basename(__file__)
                print(f"\n### Start debug {debug_index} in {file_name}")
                print(type(tool_result))
                print(f"### End debug {debug_index} in {file_name}\n")'''
                
                from langchain_core.messages import ToolMessage
                tool_msg = ToolMessage(
                    content=tool_result,
                    name="retrieve",
                    tool_call_id=call["id"]  
                )
                tool_messages.append(tool_msg)
                
        else:
            # Debug 
            '''debug_index = 1
            import os
            file_name = os.path.basename(__file__)
            print(f"\n### Start debug {debug_index} in {file_name}")
            print("LLM trả lời trực tiếp, không truy xuất document")
            print(f"### End debug {debug_index} in {file_name}\n")'''

        # Trả về cả AIMessage và ToolMessage (nếu có)
        return [response] + tool_messages

    def generate(self):
        recent_tool_messages = []
        for message in reversed(self.state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        
        # Debug 
        '''debug_index = 5
        import os
        file_name = os.path.basename(__file__)
        print(f"\n### Start debug {debug_index} in {file_name}")
        print(docs_content)
        print(f"### End debug {debug_index} in {file_name}\n")'''

        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )

        conversation_messages = []
        for message in self.state["messages"]:
            if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls):
                conversation_messages.append(message)
                
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        response = self.llm.invoke(prompt)
        self.state["answer"] = response.content

