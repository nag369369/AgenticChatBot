import streamlit as st
import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

# ── Verify Keys ───────────────────────────────────────────
groq_key = os.getenv("GROQ_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
print(f"GROQ: {'✅ Found' if groq_key else '❌ MISSING'}")
print(f"TAVILY: {'✅ Found' if tavily_key else '❌ MISSING'}")

# ── Tool ──────────────────────────────────────────────────
tool = TavilySearchResults(max_results=2)
tools = [tool]

# ── State ─────────────────────────────────────────────────
class state(TypedDict):
    messages: Annotated[list, add_messages]

# ── LLM ───────────────────────────────────────────────────
llm = init_chat_model(
    "groq:llama-3.3-70b-versatile",  # stronger model
    temperature=0
)
llm_with_tools = llm.bind_tools(tools)

# ── Chatbot Node ──────────────────────────────────────────
def chatbot(current_state: state):
    response = llm_with_tools.invoke(current_state["messages"])
    print(f"\n--- DEBUG ---")
    print(f"Tool calls made: {response.tool_calls}")  # check terminal
    print(f"Response: {response.content[:100]}")
    return {"messages": [response]}

# ── Graph ─────────────────────────────────────────────────
graph_builder = StateGraph(state)
graph_builder.add_node("llmchatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_edge(START, "llmchatbot")
graph_builder.add_conditional_edges("llmchatbot", tools_condition)
graph_builder.add_edge("tools", "llmchatbot")
graph = graph_builder.compile()

# ── UI ────────────────────────────────────────────────────
st.set_page_config(page_title="Agentic ChatBot", page_icon="🤖")
st.title("🤖 Agentic ChatBot")
st.caption("Powered by LangGraph + Groq + Tavily Search")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role": "user", "content": user_input
    })

    langgraph_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        with st.spinner("Searching & thinking..."):
            result = graph.invoke({"messages": langgraph_messages})
            response = result["messages"][-1].content
            st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant", "content": response
    })