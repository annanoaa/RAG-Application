import os
import PyPDF2
from typing import List, Dict, Tuple


class DocumentLoader:
    """Class for loading text and PDF documents."""

    def __init__(self):
        self.supported_extensions = ['.txt', '.pdf']

    def is_supported_file(self, file_path: str) -> bool:
        """Check if file type is supported."""
        _, ext = os.path.splitext(file_path)
        return ext.lower() in self.supported_extensions

    def load_document(self, file_path: str) -> Tuple[str, List[str]]:
        """Load document content from file path.

        Args:
            file_path: Path to the document file

        Returns:
            Tuple of (document name, list of text chunks)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.is_supported_file(file_path):
            supported = ', '.join(self.supported_extensions)
            raise ValueError(f"Unsupported file type. Supported types: {supported}")

        document_name = os.path.basename(file_path)
        _, ext = os.path.splitext(file_path)

        if ext.lower() == '.txt':
            return document_name, self._load_text_file(file_path)
        elif ext.lower() == '.pdf':
            return document_name, self._load_pdf_file(file_path)

    def _load_text_file(self, file_path: str) -> List[str]:
        """Load content from a text file.

        Returns:
            List of text chunks from the document
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Split content into chunks of approximately 1000 characters
        # This is a simple chunking strategy and can be improved
        chunk_size = 1000
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)

        return chunks

    def _load_pdf_file(self, file_path: str) -> List[str]:
        """Load content from a PDF file.

        Returns:
            List of text chunks from the document
        """
        chunks = []

        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()

                if text.strip():  # Only add non-empty pages
                    chunks.append(f"[Page {page_num + 1}] {text}")

        return chunks