import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# ── Load .env (GROQ_API_KEY) ─────────────────────────────
load_dotenv()

# ── 1. State (same as your notebook) ─────────────────────
class state(TypedDict):
    messages: Annotated[list, add_messages]

# ── 2. LLM (same as your notebook) ───────────────────────
llm = init_chat_model(
    "groq:llama-3.1-8b-instant",
    temperature=0
)

# ── 3. Chatbot Node (same as your notebook) ───────────────
def chatbot(current_state: state):
    return {"messages": llm.invoke(current_state["messages"])}

# ── 4. Build Graph (same as your notebook) ────────────────
graph_builder = StateGraph(state)
graph_builder.add_node("llmchatbot", chatbot)
graph_builder.add_edge(START, "llmchatbot")
graph_builder.add_edge("llmchatbot", END)
graph = graph_builder.compile()

# ── 5. Streamlit Chat UI ──────────────────────────────────
st.set_page_config(page_title="Agentic ChatBot", page_icon="🤖")
st.title("🤖 Agentic ChatBot")
st.caption("Powered by LangGraph + Groq + LLaMA 3.1")

# Keep chat history across messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show all previous messages on screen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box at the bottom
if user_input := st.chat_input("Type your message here..."):

    # 1. Show user message on screen
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Save user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # 3. Build full conversation for LangGraph
    langgraph_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    # 4. Call your LangGraph graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = graph.invoke({"messages": langgraph_messages})
            response = result["messages"][-1].content
            st.markdown(response)

    # 5. Save assistant reply to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })