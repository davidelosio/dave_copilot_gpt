import openai
import faiss
import numpy as np
import tiktoken
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class codebaseIndex():

    basic_extensions = [".py", ".java", ".scala", ".yml", ".properties"]

    def __init__(self,
                 directory=".",
                 added_extension=None,
                 embedding_model="text-embedding-ada-002"):

        if not added_extension:
            added_extension = []

        code_data = self.load_codebase(directory, self.basic_extensions + added_extension)

        self.embeddings = []
        self.filepaths = []

        for filepath, code in code_data:
            chunks = self.chunk_text(code, max_tokens=8000)
            for chunk in chunks:
                embedding = self.get_embedding(chunk, embedding_model)
                self.embeddings.append(embedding)
                self.filepaths.append(filepath)

        dimension = len(self.embeddings[0]) if len(self.embeddings) > 0 else 0
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.embeddings))

    @staticmethod
    def load_codebase(directory, extensions=None):
        """Recursively load code files from a directory."""
        code_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if extensions and not file.endswith(tuple(extensions)):
                    continue
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()
                    code_files.append((filepath, code))
        return code_files

    @staticmethod
    def chunk_text(text, max_tokens=8000, model="text-embedding-ada-002"):
        """Split text into chunks that fit within the token limit."""
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunks.append(encoding.decode(chunk_tokens))
        return chunks

    @staticmethod
    def get_embedding(text, model="text-embedding-ada-002"):
        """Generate embeddings for a given text using OpenAI API."""
        response = openai.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding

    def search_codebase(self, query, top_k=5):
        """Retrieve the most relevant code snippets for a query."""
        query_embedding = self.get_embedding(query)
        query_embedding_array = np.array([query_embedding])
        distances, indices = self.index.search(query_embedding_array, top_k)
        results = [(self.filepaths[i], distances[0][j]) for j, i in enumerate(indices[0])]

        return results

