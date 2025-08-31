def print_conversation(messages):
        for msg in messages:
            msg_type = getattr(msg, "type", None)
            if msg_type == "human":
                print("[HumanMessage]", msg.content)
            elif msg_type == "ai":
                print("[AIMessage]", msg.content)
            elif msg_type == "tool":
                print("[ToolMessage]", msg.content)
                