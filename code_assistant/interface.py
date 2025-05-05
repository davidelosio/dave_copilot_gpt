import streamlit as st
from code_assistant.codebase_embeddings_index import codebaseIndex
from code_assistant.conversational_engine import gptClient


def run_interface():
    cb_index = codebaseIndex()
    gpt_client = gptClient()

    # Set the page title and icon similar to ChatGPT
    st.set_page_config(page_title="Codebase Assistant", page_icon="🤖")

    st.title("Codebase Assistant")

    # Initialize session state for message history
    if 'message_history' not in st.session_state:
        st.session_state.message_history = []

    # Input area for user queries
    st.markdown("<style>.textarea {font-size: 18px;}</style>", unsafe_allow_html=True)
    query = st.text_area("Ask a question about the codebase:", height=100, key="input_area")

    if query:
        results = cb_index.search_codebase(query)
        top_snippets = [open(filepath).read() for filepath, _ in results]
        answer = gpt_client.ask_gpt4(query, top_snippets)

        # Update message history
        st.session_state.message_history.append({"role": "user", "content": query})
        st.session_state.message_history.append({"role": "assistant", "content": answer})

        # Keep only the last 100 messages
        if len(st.session_state.message_history) > 100:
            st.session_state.message_history = st.session_state.message_history[-100:]

    # Display message history
    with st.container():
        for message in reversed(st.session_state.message_history):
            if message["role"] == "user":
                st.markdown(
                    f"<div style='background-color: #343a40; color: white; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>User:</strong> {message['content']}</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div style='background-color: #444654; color: white; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Assistant:</strong> {message['content']}</div>",
                    unsafe_allow_html=True)


if __name__ == "__main__":
    run_interface()