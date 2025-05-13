import openai
import faiss
import numpy as np
import tiktoken
import os
import json
import ast


class codebaseIndex:
    basic_extensions = [".py", ".java", ".scala", ".yml", ".properties"]
    def __init__(self,
                 directory=".",
                 added_extension=None,
                 embedding_model="text-embedding-ada-002"
                 ):
        if not added_extension:
            added_extension = []
        # Store directory and extensions as instance variables
        self.directory = directory
        self.extensions = self.basic_extensions + added_extension
        self.embedding_model = embedding_model
        self.index_filepath = "codebase_index.faiss"
        self.metadata_filepath = "codebase_metadata.json"

        # Load existing index or rebuild if changes detected
        if not self.load_index():
            code_data = self.load_codebase(directory, self.extensions)
            # Store modification times of files
            self.file_mtimes = {filepath: os.path.getmtime(filepath) for filepath, _ in code_data}
            self.embeddings = []
            self.filepaths = []
            self.chunks = []
            for filepath, code in code_data:
                chunks = self.chunk_text(code, filepath)
                for chunk in chunks:
                    embedding = self.get_embedding(chunk)
                    self.embeddings.append(embedding)
                    self.filepaths.append(filepath)
                    self.chunks.append(chunk)
            dimension = len(self.embeddings[0]) if self.embeddings else 0
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(self.embeddings))
            self.save_index()

    @staticmethod
    def load_codebase(directory, extensions=None, exclude_dirs=None):
        if exclude_dirs is None:
            exclude_dirs = ['.venv', '.git', '__pycache__']
        code_files = []
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if extensions and not file.endswith(tuple(extensions)):
                    continue
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        code = f.read()
                    code_files.append((filepath, code))
                except (IOError, UnicodeDecodeError) as e:
                    print(f"Warning: Could not read {filepath}: {e}")
        return code_files

    @staticmethod
    def chunk_text(text, filepath, max_tokens=8000, model="text-embedding-ada-002"):
        encoding = tiktoken.encoding_for_model(model)
        if filepath.endswith('.py'):
            try:
                tree = ast.parse(text)
                chunks = []
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        node_text = ast.get_source_segment(text, node)
                        tokens = encoding.encode(node_text)
                        if len(tokens) <= max_tokens:
                            chunks.append(node_text)
                        else:
                            chunks.extend(
                                [encoding.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)])
                module_text = '\n'.join(line for line in text.splitlines() if
                                        not line.strip().startswith('def') and not line.strip().startswith('class'))
                tokens = encoding.encode(module_text)
                chunks.extend([encoding.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)])
                return chunks
            except SyntaxError:
                pass
        tokens = encoding.encode(text)
        return [encoding.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]

    @staticmethod
    def get_embedding(text, model="text-embedding-ada-002"):
        response = openai.embeddings.create(model=model, input=text)
        return response.data[0].embedding

    def search_codebase(self, query, top_k=5):
        query_embedding = self.get_embedding(query)
        query_embedding_array = np.array([query_embedding])
        distances, indices = self.index.search(query_embedding_array, top_k)
        results = [(self.filepaths[i], self.chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]
        return results

    def save_index(self):
        faiss.write_index(self.index, self.index_filepath)
        metadata = {
            "directory": self.directory,
            "filepaths": self.filepaths,
            "chunks": self.chunks,
            "file_mtimes": self.file_mtimes
        }
        with open(self.metadata_filepath, "w") as f:
            json.dump(metadata, f)

    def get_current_files_mtimes(self):
        """Get the current modification times of files in the directory."""
        current_file_mtimes = {}
        for root, dirs, files in os.walk(self.directory):
            dirs[:] = [d for d in dirs if d not in ['.venv', '.git', '__pycache__']]
            for file in files:
                if self.extensions and not file.endswith(tuple(self.extensions)):
                    continue
                filepath = os.path.join(root, file)
                if os.path.isfile(filepath):
                    current_file_mtimes[filepath] = os.path.getmtime(filepath)
        return current_file_mtimes

    def load_index(self):
        """Load the index if it exists and the codebase hasn’t changed."""
        if os.path.exists(self.index_filepath) and os.path.exists(self.metadata_filepath):
            with open(self.metadata_filepath, "r") as f:
                metadata = json.load(f)
            # Check if the directory matches
            if metadata.get("directory") != self.directory:
                return False
            stored_file_mtimes = metadata.get("file_mtimes", {})
            current_file_mtimes = self.get_current_files_mtimes()
            # Check if the set of files matches (handles additions/deletions)
            if set(current_file_mtimes.keys()) != set(stored_file_mtimes.keys()):
                return False
            # Check if modification times match (handles changes)
            for fp in current_file_mtimes:
                if current_file_mtimes[fp] != stored_file_mtimes[fp]:
                    return False
            # If everything matches, load the index
            self.index = faiss.read_index(self.index_filepath)
            self.filepaths = metadata["filepaths"]
            self.chunks = metadata["chunks"]
            self.file_mtimes = stored_file_mtimes
            return True
        return False