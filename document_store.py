import os
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import uuid


class DocumentStore:
    """Class for processing and storing document content using FAISS and TF-IDF."""

    def __init__(self, persist_directory: str = "./vector_store"):
        """Initialize the document store.

        Args:
            persist_directory: Directory to persist the vector store
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize the text splitter for more granular chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )

        # TF-IDF vectorizer for embeddings
        self.vectorizer = TfidfVectorizer()

        # Initialize or load the vector store
        self.index_path = os.path.join(persist_directory, "faiss_index.bin")
        self.metadata_path = os.path.join(persist_directory, "metadata.pkl")
        self.vectors_path = os.path.join(persist_directory, "vectors.pkl")

        if (os.path.exists(self.index_path) and
                os.path.exists(self.metadata_path) and
                os.path.exists(self.vectors_path)):
            self._load_vector_store()
        else:
            self._create_vector_store()

        # Keep a record of document IDs for reference
        self.document_ids = {}

    def _create_vector_store(self):
        """Create a new vector store."""
        # Initialize empty data structures
        self.all_texts = []
        self.all_metadatas = []
        self.document_ids = {}

        # Initialize FAISS index (will be created when first document is added)
        self.index = None

    def _load_vector_store(self):
        """Load an existing vector store."""
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)

        # Load metadata and texts
        with open(self.metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.all_metadatas = data['metadatas']
            self.document_ids = data['document_ids']

        with open(self.vectors_path, 'rb') as f:
            self.all_texts = pickle.load(f)

    def _save_vector_store(self):
        """Save the vector store to disk."""
        if self.index is not None:
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)

            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump({
                    'metadatas': self.all_metadatas,
                    'document_ids': self.document_ids
                }, f)

            # Save texts
            with open(self.vectors_path, 'wb') as f:
                pickle.dump(self.all_texts, f)

    def add_document(self, document_name: str, text_chunks: List[str]) -> str:
        """Process and add a document to the store.

        Args:
            document_name: Name of the document
            text_chunks: List of text chunks from the document

        Returns:
            Document ID
        """
        # Generate a unique ID for the document
        document_id = str(uuid.uuid4())

        # Further split the chunks for better retrieval
        all_splits = []
        metadatas = []

        for i, chunk in enumerate(text_chunks):
            splits = self.text_splitter.split_text(chunk)

            for split in splits:
                all_splits.append(split)
                metadatas.append({
                    "document_id": document_id,
                    "document_name": document_name,
                    "chunk_id": i,
                    "source": f"{document_name}:chunk_{i}"
                })

        # Add new texts to our list
        self.all_texts.extend(all_splits)
        self.all_metadatas.extend(metadatas)

        # Fit TF-IDF on all texts and transform to get vectors
        vectors = self.vectorizer.fit_transform(self.all_texts).toarray().astype(np.float32)

        # Create or update FAISS index
        dimension = vectors.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)

        # Reset and add all vectors (simple approach - for production you'd want to add incrementally)
        self.index.reset()
        self.index.add(vectors)

        # Save the document ID
        self.document_ids[document_id] = document_name

        # Persist the vector store
        self._save_vector_store()

        return document_id

    def search_documents(self, query: str, top_k: int = 4) -> List[Dict]:
        """Search for relevant document chunks.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of relevant document chunks with metadata
        """
        if not self.all_texts or self.index is None:
            return []

        # Convert query to vector using the same vectorizer
        query_vector = self.vectorizer.transform([query]).toarray().astype(np.float32)

        # Search the index
        distances, indices = self.index.search(query_vector, min(top_k, len(self.all_texts)))

        # Gather results
        documents = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.all_texts):  # Skip invalid indices
                continue

            documents.append({
                "content": self.all_texts[idx],
                "metadata": self.all_metadatas[idx],
                "score": float(distances[0][i])
            })

        return documents

    def get_all_documents(self) -> List[str]:
        """Get list of all document names in the store."""
        return list(set(self.document_ids.values()))

    def clear_all_documents(self) -> None:
        """Remove all documents from the store."""
        self._create_vector_store()

        # Remove saved files
        for file_path in [self.index_path, self.metadata_path, self.vectors_path]:
            if os.path.exists(file_path):
                os.remove(file_path)

        self.document_ids = {}