# Codebase Assistant
The Codebase Assistant is a GPT-based Retrieval-Augmented Generation (RAG) system designed to act as a coding copilot. It indexes your codebase, understands its contents, and assists you by answering questions or offering coding suggestions.
## Features

* Codebase Indexing: Indexes your codebase using embeddings and FAISS for efficient similarity search.
* GPT-4 Integration: Leverages GPT-4 to process queries and generate helpful responses.
* User-Friendly Interface: Provides a Streamlit-based interface for easy interaction.
* Extensible: Supports multiple file extensions and allows adding more as needed.

## Installation
To get started, follow these steps:
1. Clone the Repository
2. Navigate to the Project Directory:
   `cd codebase-assistant`
3. Install Dependencies, you need poetry:
   `poetry install`

4. Set Up OpenAI API Key:
   Create a .env file in the project directory.
   Add your OpenAI API key:
   `OPENAI_API_KEY=your-api-key`

## Usage

Run the Application:

`python Main.py --directory /path/to/your/codebase`

Replace /path/to/your/codebase with the path to the directory containing your codebase.

## Supported File Extensions

The assistant supports the following file extensions by default:
.py .java .scala .yml .properties

## Project Structure
Here’s an overview of the main files and their purposes:

* codebase_embedding_index.py: Manages indexing the codebase using embeddings and FAISS.
* conversational_engine.py: Handles interaction with GPT-4 to generate responses.
* interface.py: Provides the Streamlit-based user interface.
* Main.py: Launches the Streamlit app.


Thank you for using the Codebase Assistant!

