import streamlit as st
import argparse
from code_assistant.codebase_embeddings_index import codebaseIndex
from code_assistant.conversational_engine import gptClient

# Set page config as the first Streamlit command
st.set_page_config(page_title="Codebase Assistant", page_icon="🤖")

# Cache the codebase index to avoid re-indexing on every run
@st.cache_resource
def get_codebase_index(directory):
    return codebaseIndex(directory=directory)

# Main function to run the interface
def run_interface(directory):
    # Initialize the codebase index and GPT client
    cb_index = get_codebase_index(directory)
    gpt_client = gptClient()

    st.title("Codebase Assistant")

    # Initialize message history in session state
    if 'message_history' not in st.session_state:
        st.session_state.message_history = []

    # Button to clear chat history
    if st.button("Clear History"):
        st.session_state.message_history = []
        st.rerun()  # Force a re-run to refresh the UI

    # Display the full message history in chronological order
    with st.container():
        for message in st.session_state.message_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "snippets" in message:
                    with st.expander("View Relevant Snippets"):
                        for filepath, chunk in message["snippets"]:
                            st.code(chunk, language="python")

    # Chat input for user queries, placed at the bottom
    query = st.chat_input("Ask a question about the codebase:")
    if query and query.strip():
        # Append user query to history
        st.session_state.message_history.append({"role": "user", "content": query})
        # Process the assistant's response
        with st.spinner("Processing..."):
            results = cb_index.search_codebase(query)
            top_snippets = [f"File: {filepath}\n{chunk}" for filepath, chunk, _ in results]
            answer = gpt_client.ask_gpt4(query, top_snippets)
        # Append assistant's response to history
        st.session_state.message_history.append({
            "role": "assistant",
            "content": answer,
            "snippets": [(filepath, chunk) for filepath, chunk, _ in results]
        })
        # Re-run the app to show the updated history
        st.rerun()

# Entry point with argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Codebase Assistant")
    parser.add_argument("--directory", default=".", help="Directory to index")
    args = parser.parse_args()
    run_interface(args.directory)