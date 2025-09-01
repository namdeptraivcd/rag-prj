from src.config.config import Config
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from src.utils.helper_functions import retrieve_context_per_question
from src.model.vectorstore import Vector_store

cfg = Config()


def make_retrieve_tool(vector_store):
        @tool(response_format="content_and_artifact")
        def retrieve(query):
            """
            Retrieve relevant documents from the vector store based on the query.
            """
            retrieved_docs = []
            for vs in vector_store:
                retrieved_docs.extend(vs.similarity_search(query))
            serialized = "\n\n".join(
                f"Source: {doc.metadata}, Content: {doc.page_content}"
                for doc in retrieved_docs
            )
        
            return (serialized, retrieved_docs)
        return retrieve


class RAG:
    def __init__(self):
        self.prompt = cfg.prompt
        self.llm = cfg.llm
        self.state = MessagesState()
        self.vector_store = Vector_store()
        self.retrieve = make_retrieve_tool(self.vector_store.vector_store)
        
        # Conversation history
        self.state["messages"] = []
        system_prompt = (
            "You are a retrieval-augmented assistant. "
            "ALWAYS use the retrieve tool to get context from the provided documents before answering any user question. "
            "Do not answer from your own knowledge unless the tool result is empty."
        )
        self.state["messages"].insert(0, SystemMessage(content=system_prompt))
        
    def query_or_respond(self):
        llm_with_tools = self.llm.bind_tools([self.retrieve])
        response = llm_with_tools.invoke(self.state["messages"])
        # If there are tool calls, execute tools and create ToolMessage
        tool_messages = []
        tool_calls = getattr(response, "tool_calls", None)
        
        if tool_calls:
            answer_type = "rag"
            
            for call in tool_calls:
                # Execute tool with query
                tool_result = self.retrieve.invoke({"query": call["args"]["query"]})
                
                from langchain_core.messages import ToolMessage
                tool_msg = ToolMessage(
                    content=tool_result,
                    name="retrieve",
                    tool_call_id=call["id"]  
                )
                tool_messages.append(tool_msg)
                
        else:
            answer_type = "llm"

        # Return both AIMessage and ToolMessage (if any)
        return [response] + tool_messages, answer_type

    def generate(self):
        recent_tool_messages = []
        for message in reversed(self.state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        docs_content = "\n\n".join(doc.content for doc in tool_messages)

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
