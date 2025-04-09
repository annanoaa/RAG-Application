from typing import Dict, List, Tuple, Optional
from document_store import DocumentStore


class RAGEngine:
    """Implementation of Retrieval-Augmented Generation for document QA."""

    def __init__(self, document_store: DocumentStore):
        """Initialize the RAG engine.

        Args:
            document_store: Document store for retrieving relevant content
        """
        self.document_store = document_store

    def answer_question(self, question: str) -> Tuple[str, List[Dict]]:
        """Answer a question based on the documents in the store.

        Args:
            question: User's question

        Returns:
            Tuple of (answer, list of sources used)
        """
        # Retrieve relevant document chunks
        relevant_chunks = self.document_store.search_documents(question)

        if not relevant_chunks:
            return "I don't have enough information to answer that question based on the documents provided.", []

        # Generate answer based on retrieved content
        answer = self._generate_answer(question, relevant_chunks)

        # Format sources
        sources = self._format_sources(relevant_chunks)

        return answer, sources

    def _generate_answer(self, question: str, relevant_chunks: List[Dict]) -> str:
        """Generate an answer based on the question and relevant chunks.

        This is a simplified generation method. In a real-world scenario,
        you might want to use a large language model like GPT-3.5 or GPT-4.

        Args:
            question: User's question
            relevant_chunks: List of relevant document chunks

        Returns:
            Generated answer
        """
        # Extract content from relevant chunks
        context = "\n".join([chunk["content"] for chunk in relevant_chunks])

        # Check if we have relevant information
        if not context.strip():
            return "I don't have enough information to answer that question based on the documents provided."

        # Simple answer generation - combine relevant chunks
        # This is where you would normally use an LLM
        answer = f"Based on the documents, I found the following information:\n\n{context}"

        return answer

    def _format_sources(self, relevant_chunks: List[Dict]) -> List[Dict]:
        """Format sources for citation.

        Args:
            relevant_chunks: List of relevant document chunks

        Returns:
            List of formatted sources
        """
        sources = []

        for chunk in relevant_chunks:
            sources.append({
                "document": chunk["metadata"]["document_name"],
                "content": chunk["content"][:100] + "..." if len(chunk["content"]) > 100 else chunk["content"],
                "source": chunk["metadata"]["source"]
            })

        return sources