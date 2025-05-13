import streamlit as st
import argparse
from code_assistant.codebase_embeddings_index import codebaseIndex
from code_assistant.conversational_engine import gptClient

# Set page config as the first Streamlit command to define the app's title and icon
st.set_page_config(page_title="Codebase Assistant", page_icon="🤖")

@st.cache_resource
def get_codebase_index(directory):
    """
    Create and cache a codebase index for the specified directory.

    Args:
        directory (str): The directory containing the codebase to index.

    Returns:
        codebaseIndex: An instance of the codebaseIndex class with the indexed codebase.
    """
    return codebaseIndex(directory=directory)

def run_interface(directory):
    """
    Run the Streamlit interface for the Codebase Assistant.

    Initializes the codebase index and GPT client, displays the chat interface,
    and handles user queries and responses.

    Args:
        directory (str): The directory containing the codebase to index.
    """
    # Initialize the codebase index and GPT client
    cb_index = get_codebase_index(directory)
    gpt_client = gptClient()

    # Display the app title
    st.title("Codebase Assistant")

    # Initialize message history in session state to persist chat messages
    if 'message_history' not in st.session_state:
        st.session_state.message_history = []

    # Button to clear chat history and refresh the UI
    if st.button("Clear History"):
        st.session_state.message_history = []
        st.rerun()  # Force a re-run to refresh the UI

    # Display the full message history in chronological order (oldest to newest)
    with st.container():
        for message in st.session_state.message_history:
            with st.chat_message(message["role"]):
                # Render the message content
                st.markdown(message["content"])
                # Display code snippets for assistant messages, if available
                if message["role"] == "assistant" and "snippets" in message:
                    with st.expander("View Relevant Snippets"):
                        for filepath, chunk in message["snippets"]:
                            st.code(chunk, language="python")

    # Chat input for user queries, placed at the bottom of the page
    query = st.chat_input("Ask a question about the codebase:")
    if query and query.strip():
        # Append user query to history
        st.session_state.message_history.append({"role": "user", "content": query})
        # Process the assistant's response
        with st.spinner("Processing..."):
            # Search the codebase for relevant snippets
            results = cb_index.search_codebase(query)
            top_snippets = [f"File: {filepath}\n{chunk}" for filepath, chunk, _ in results]
            # Get the assistant's response using GPT-4
            answer = gpt_client.ask_gpt4(query, top_snippets)
        # Append assistant's response to history
        st.session_state.message_history.append({
            "role": "assistant",
            "content": answer,
            "snippets": [(filepath, chunk) for filepath, chunk, _ in results]
        })
        # Re-run the app to show the updated history
        st.rerun()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Codebase Assistant")
    parser.add_argument("--directory", default=".", help="Directory to index")
    args = parser.parse_args()
    run_interface(args.directory)