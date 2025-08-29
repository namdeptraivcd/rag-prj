from src.model.rag.state.state import State, Search
from src.config.config import Config
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool



config = Config()

class RAG:
    def __init__(self, vector_store):
        self.prompt = config.prompt
        self.llm = config.llm
        self.state = MessagesState()
        self.vector_store = vector_store

    @tool(response_format = "content_and_artifact")
    def retrieve(self, query):
        """
        Retrieve relevant documents from the vector store based on the query.
        """
        retrieved_docs = self.vector_store.similarity_search(query)
        serialized = "\n\n".join (f"Source: {doc.metadata}, Content: {doc.page_content}" 
                             for doc in retrieved_docs)
        return {"content": serialized, 
                "artifact":retrieved_docs}
        
    
    def query_or_respond (self):
        llm_with_tools = self.llm.bind_tools([self.retrieve])
        respone = llm_with_tools.invoke(self.state["messages"])
        # Nếu có tool_call, thực thi tool và tạo ToolMessage
        tool_messages = []
        tool_calls = getattr(respone, "tool_calls", None)
        if tool_calls:
            print("LLM đã gọi tool retrieve!")
            print("Thông tin tool_calls:", tool_calls)
            for call in tool_calls:
                # Thực thi tool với query
                tool_result = self.retrieve(call["args"]["query"]) #Debug
                from langchain_core.messages import ToolMessage
                tool_msg = ToolMessage(
                    content=tool_result["content"],
                    name="retrieve",
                    tool_call_id=call["id"]  # Phải đúng tool_call_id
                )
                tool_messages.append(tool_msg)
        else:
            print("LLM trả lời trực tiếp, không truy xuất document.")

        # Trả về cả AIMessage và ToolMessage (nếu có)
        return {"messages": [respone] + tool_messages}
        



    def generate(self):
        recent_tool_messages = []
        for message in reversed(self.state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        docs_content = "\n\n".join(doc.page_content for doc in tool_messages)
        system_message_content = (
    
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
        conversation_messages =[
            message
            for message in self.state["messages"]
            if message.type in("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages


        respone = self.llm.invoke(prompt)
        self.state["answer"] = respone.content

