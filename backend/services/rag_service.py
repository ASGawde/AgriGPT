import json
import os
from typing import List, Dict
import unicodedata
import re

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/subsidies.json")
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "../data/faiss_index")



def _clean_query(text: str) -> str:
    """
    Clean query for FAISS safety WITHOUT removing Indian languages.
    - Normalize Unicode (Tamil/Hindi safe)
    - Remove control characters
    - Collapse whitespace
    """
    if not isinstance(text, str):
        return ""

    # Unicode normalize (Tamil/Hindi safe)
    text = unicodedata.normalize("NFKC", text)

    # Remove invisible control characters (keeps Indian alphabets)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()

class RAG:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAG, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        """Initialize embeddings + FAISS index."""
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None

        # ----------------------
        # 1. Try loading FAISS
        # ----------------------
        if os.path.exists(VECTOR_DB_PATH):
            try:
                print("Loading existing FAISS index...")
                self.vector_store = FAISS.load_local(
                    VECTOR_DB_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("FAISS index loaded.")
                return

            except Exception as e:
                print(f"[RAG] Failed to load FAISS index: {e}")
                print("[RAG] Deleting corrupted index and rebuilding...")

                # Delete corrupted index
                try:
                    for file in os.listdir(VECTOR_DB_PATH):
                        os.remove(os.path.join(VECTOR_DB_PATH, file))
                    os.rmdir(VECTOR_DB_PATH)
                except:
                    pass

        # ----------------------
        # 2. Rebuild fresh index
        # ----------------------
        if not os.path.exists(DATA_PATH):
            print(f"[RAG] subsidies.json not found at: {DATA_PATH}")
            return

        with open(DATA_PATH, "r") as f:
            raw_data = json.load(f)

        documents = []
        for item in raw_data:

            # Safe extraction of keys
            scheme_name = item.get("scheme_name", "Unknown Scheme")
            eligibility = item.get("eligibility", "Not Provided")
            benefits = item.get("benefits", "Not Provided")
            notes = item.get("notes", "")

            content = (
                f"Scheme: {scheme_name}\n"
                f"Eligibility: {eligibility}\n"
                f"Benefits: {benefits}\n"
                f"Notes: {notes}\n"
            )

            documents.append(
                Document(page_content=content, metadata=item)
            )

        print("[RAG] Building FAISS index...")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.vector_store.save_local(VECTOR_DB_PATH)
        print("[RAG] FAISS index built and saved successfully.")

    # ------------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: int = 2) -> List[Dict]:
        """
        Retrieve relevant subsidies.
        Safe against FAISS errors or missing vectors.
        """
        if not query or not query.strip():
            return []

        query_clean = _clean_query(query)

        if not self.vector_store:
            print("[RAG] Vector store not loaded.")
            return []

        try:
            docs = self.vector_store.similarity_search(query_clean, k=k)
            return [d.metadata for d in docs]

        except Exception as e:
            print(f"[RAG] Retrieval error: {e}")
            return []


# Singleton access
rag_service = RAG()
