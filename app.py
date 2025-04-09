import os
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
from typing import List, Dict, Tuple, Optional

from document_loader import DocumentLoader
from document_store import DocumentStore
from rag_engine import RAGEngine


class DocumentRAGApp:
    """Main application with GUI for Document RAG."""

    def __init__(self, root):
        """Initialize the application.

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Document RAG Application")
        self.root.geometry("900x700")

        # Initialize components
        self.document_loader = DocumentLoader()
        self.document_store = DocumentStore()
        self.rag_engine = RAGEngine(self.document_store)

        # Set up the GUI
        self._setup_gui()

        # Load document list
        self._refresh_document_list()

    def _setup_gui(self):
        """Set up the GUI components."""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create left panel for document management
        left_panel = ttk.LabelFrame(main_frame, text="Document Management", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Document list
        ttk.Label(left_panel, text="Loaded Documents:").pack(anchor=tk.W)
        self.document_listbox = tk.Listbox(left_panel, height=10)
        self.document_listbox.pack(fill=tk.BOTH, expand=True, pady=5)

        # Buttons for document management
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Load Document", command=self._load_document).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Clear All", command=self._clear_all_documents).pack(side=tk.LEFT, padx=2)

        # Create right panel for QA
        right_panel = ttk.LabelFrame(main_frame, text="Question Answering", padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Question input
        ttk.Label(right_panel, text="Ask a question:").pack(anchor=tk.W)
        self.question_entry = ttk.Entry(right_panel, width=50)
        self.question_entry.pack(fill=tk.X, pady=5)
        self.question_entry.bind("<Return>", lambda e: self._answer_question())

        ttk.Button(right_panel, text="Ask", command=self._answer_question).pack(anchor=tk.W, pady=5)

        # Answer display
        ttk.Label(right_panel, text="Answer:").pack(anchor=tk.W)
        self.answer_text = scrolledtext.ScrolledText(right_panel, wrap=tk.WORD, height=15)
        self.answer_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Sources display
        ttk.Label(right_panel, text="Sources:").pack(anchor=tk.W)
        self.sources_text = scrolledtext.ScrolledText(right_panel, wrap=tk.WORD, height=10)
        self.sources_text.pack(fill=tk.BOTH, expand=True, pady=5)

    def _load_document(self):
        """Load a document from file."""
        file_path = filedialog.askopenfilename(
            title="Select Document",
            filetypes=[
                ("Supported Files", "*.txt *.pdf"),
                ("Text Files", "*.txt"),
                ("PDF Files", "*.pdf"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            document_name, text_chunks = self.document_loader.load_document(file_path)
            document_id = self.document_store.add_document(document_name, text_chunks)

            messagebox.showinfo("Success", f"Document '{document_name}' loaded successfully.")
            self._refresh_document_list()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load document: {str(e)}")

    def _refresh_document_list(self):
        """Refresh the document list display."""
        self.document_listbox.delete(0, tk.END)

        documents = self.document_store.get_all_documents()
        for doc in documents:
            self.document_listbox.insert(tk.END, doc)

    def _clear_all_documents(self):
        """Clear all documents from the store."""
        if messagebox.askyesno("Confirm", "Are you sure you want to remove all documents?"):
            self.document_store.clear_all_documents()
            self._refresh_document_list()
            messagebox.showinfo("Success", "All documents have been removed.")

    def _answer_question(self):
        """Answer the user's question."""
        question = self.question_entry.get().strip()

        if not question:
            messagebox.showwarning("Warning", "Please enter a question.")
            return

        # Clear previous answers
        self.answer_text.delete(1.0, tk.END)
        self.sources_text.delete(1.0, tk.END)

        try:
            # Get answer from RAG engine
            answer, sources = self.rag_engine.answer_question(question)

            # Display answer
            self.answer_text.insert(tk.END, answer)

            # Display sources
            if sources:
                sources_text = "The answer was derived from the following sources:\n\n"
                for i, source in enumerate(sources, 1):
                    sources_text += f"{i}. {source['document']} - {source['source']}\n"
                    sources_text += f"   Excerpt: {source['content']}\n\n"

                self.sources_text.insert(tk.END, sources_text)
            else:
                self.sources_text.insert(tk.END, "No specific sources were used to generate this answer.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


def main():
    """Run the application."""
    root = tk.Tk()
    app = DocumentRAGApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()