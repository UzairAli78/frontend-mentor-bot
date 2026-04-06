"""
RAG Pipeline Implementation for Front-End Mentor Bot
Handles document processing, embedding, vector storage, and retrieval
"""

import os
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Vector Database
import chromadb
from chromadb.config import Settings

# Embeddings
from sentence_transformers import SentenceTransformer

# Text Processing
import nltk
from nltk.tokenize import word_tokenize


class DocumentProcessor:
    """Handles document chunking and processing"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize document processor.

        Args:
            chunk_size: Number of tokens per chunk
            overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Download NLTK data if needed
        # Both LookupError (not found) and OSError (found but corrupt) are caught
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except (LookupError, OSError):
            nltk.download('punkt_tab', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt')
        except (LookupError, OSError):
            nltk.download('punkt', quiet=True)

        # Final safety check: verify word_tokenize actually works
        try:
            word_tokenize("test")
        except Exception:
            # Force re-download both packages if tokenizer is still broken
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        # Tokenize into words
        words = word_tokenize(text)
        chunks = []

        # Create chunks with overlap
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def load_document(self, filepath: str) -> str:
        """
        Load document from file.

        Args:
            filepath: Path to document file

        Returns:
            Document content as string
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()


class QueryRouter:
    """Routes queries to appropriate document categories"""

    # Keyword mappings for each category
    CATEGORY_KEYWORDS = {
        'tech': [
            'code', 'syntax', 'html', 'css', 'javascript', 'react', 'bootstrap',
            'tailwind', 'deploy', 'github', 'git', 'pages', 'how do i', 'how to',
            'example', 'function', 'component', 'element', 'property', 'method',
            'api', 'flexbox', 'grid', 'responsive', 'npm', 'node'
        ],
        'interview': [
            'interview', 'question', 'answer', 'ask me', 'quiz', 'test',
            'practice', 'prepare', 'evaluate', 'feedback', 'compare my answer',
            'what is', 'explain', 'difference between'
        ],
        'cv': [
            'cv', 'resume', 'portfolio', 'project description', 'how to describe',
            'apply', 'application', 'internship', 'bullet point', 'experience',
            'education', 'format', 'layout', 'contact', 'github profile',
            'linkedin', 'skills section'
        ]
    }

    @staticmethod
    def route_query(query: str) -> Dict:
        """
        Determine which category a query belongs to.

        Args:
            query: User query string

        Returns:
            Dictionary with category, confidence, and multi-category flag
        """
        query_lower = query.lower()
        scores = {}

        # Calculate scores for each category
        for category, keywords in QueryRouter.CATEGORY_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            scores[category] = score

        # Determine primary category
        max_category = max(scores, key=scores.get)
        max_score = scores[max_category]

        # Default to 'tech' when no keywords match at all
        if max_score == 0:
            max_category = 'tech'

        # Check if multiple categories have high scores
        high_score_categories = [cat for cat, score in scores.items() if score > 0]
        needs_multiple = len(high_score_categories) > 1

        return {
            'category': max_category,
            'confidence': max_score,
            'needs_multiple_categories': needs_multiple,
            'all_scores': scores
        }

    @staticmethod
    def is_interview_evaluation(query: str, conversation_history: List[str]) -> bool:
        """
        Determine if query is requesting interview answer evaluation.

        Args:
            query: Current query
            conversation_history: List of previous messages

        Returns:
            True if this is an evaluation request
        """
        query_lower = query.lower()

        # Check for explicit evaluation keywords
        eval_keywords = ['my answer', 'evaluate', 'how did i do', 'was that correct']
        if any(keyword in query_lower for keyword in eval_keywords):
            return True

        # Check if previous message was an interview question
        if len(conversation_history) >= 2:
            prev_message = conversation_history[-2].lower()
            if 'question:' in prev_message or 'what is' in prev_message:
                return True

        return False


class RAGPipeline:
    """Main RAG pipeline for document retrieval and generation"""

    def __init__(
        self,
        corpus_dir: str,
        embedding_model: str = 'all-MiniLM-L6-v2',
        persist_dir: str = './vector_db',
        chunk_size: int = 500,
        overlap: int = 50
    ):
        """
        Initialize RAG pipeline.

        Args:
            corpus_dir: Directory containing corpus files
            embedding_model: Name of sentence transformer model
            persist_dir: Directory to persist vector database
            chunk_size: Chunk size for document processing
            overlap: Overlap size for chunks
        """
        self.corpus_dir = corpus_dir
        self.persist_dir = persist_dir

        # Initialize components
        self.doc_processor = DocumentProcessor(chunk_size, overlap)
        self.router = QueryRouter()

        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        # Initialize ChromaDB
        print(f"Initializing vector database in: {persist_dir}")
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="frontend_mentor_docs",
            metadata={"description": "Front-end mentor knowledge base"}
        )

        print("RAG Pipeline initialized successfully!")

    def load_corpus(self, force_reload: bool = False):
        """
        Load corpus files into vector database.

        Args:
            force_reload: If True, delete existing data and reload
        """
        # Check if already loaded
        if self.collection.count() > 0 and not force_reload:
            print(f"Vector database already loaded with {self.collection.count()} chunks")
            return

        # Clear existing data if force reload
        if force_reload and self.collection.count() > 0:
            print("Clearing existing vector database...")
            self.client.delete_collection("frontend_mentor_docs")
            self.collection = self.client.create_collection(
                name="frontend_mentor_docs",
                metadata={"description": "Front-end mentor knowledge base"}
            )

        print("Loading corpus files...")

        # Document metadata mapping
        doc_metadata = {
            'tech_docs.txt': {'source': 'tech_docs', 'category': 'tech'},
            'interview_bank.txt': {'source': 'interview_bank', 'category': 'interview'},
            'cv_guidelines.txt': {'source': 'cv_guidelines', 'category': 'cv'}
        }

        all_chunks = []
        all_metadatas = []
        all_ids = []

        for filename, metadata in doc_metadata.items():
            filepath = os.path.join(self.corpus_dir, filename)

            if not os.path.exists(filepath):
                print(f"Warning: {filepath} not found, skipping...")
                continue

            print(f"Processing {filename}...")

            # Load and chunk document
            content = self.doc_processor.load_document(filepath)
            chunks = self.doc_processor.chunk_text(content)

            print(f"  Created {len(chunks)} chunks from {filename}")

            # Prepare chunks with metadata
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })
                all_ids.append(f"{metadata['source']}_chunk_{i}")

        if not all_chunks:
            print("ERROR: No chunks were loaded. Check that corpus files exist.")
            return

        # Generate embeddings
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.embedder.encode(all_chunks, show_progress_bar=True)

        # Add to vector database in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch_end = min(i + batch_size, len(all_chunks))
            self.collection.add(
                documents=all_chunks[i:batch_end],
                embeddings=embeddings[i:batch_end].tolist(),
                metadatas=all_metadatas[i:batch_end],
                ids=all_ids[i:batch_end]
            )

        print(f"✓ Corpus loaded successfully! Total chunks: {self.collection.count()}")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents for query.

        Args:
            query: User query
            top_k: Number of results to return
            category_filter: Optional category to filter by

        Returns:
            List of retrieved documents with metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()

        # Build metadata filter
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}

        # Clamp top_k to available document count
        available = self.collection.count()
        top_k = min(top_k, available) if available > 0 else 1

        # Query vector database
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )

        # Format results
        retrieved_docs = []
        for i in range(len(results['ids'][0])):
            retrieved_docs.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })

        return retrieved_docs

    def process_query(
        self,
        query: str,
        conversation_history: Optional[List[str]] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Process query through routing and retrieval.

        Args:
            query: User query
            conversation_history: Optional conversation history

        Returns:
            Tuple of (retrieved documents, routing info)
        """
        if conversation_history is None:
            conversation_history = []

        # Route query
        routing_info = self.router.route_query(query)

        # Determine if this is interview evaluation
        is_evaluation = self.router.is_interview_evaluation(query, conversation_history)
        routing_info['is_evaluation'] = is_evaluation

        # Retrieve documents
        if routing_info['needs_multiple_categories']:
            # Search across all categories
            retrieved_docs = self.retrieve(query, top_k=5)
        else:
            # Search specific category
            retrieved_docs = self.retrieve(
                query,
                top_k=5,
                category_filter=routing_info['category']
            )

        return retrieved_docs, routing_info


def test_rag_pipeline():
    """Test function for RAG pipeline"""
    print("\n" + "=" * 60)
    print("TESTING RAG PIPELINE")
    print("=" * 60 + "\n")

    # Resolve paths: this script lives in app/, corpus/ and vector_db/ are in project root
    script_dir = os.path.dirname(os.path.abspath(__file__))   # .../app
    project_root = os.path.dirname(script_dir)                 # project root
    corpus_dir = os.path.join(project_root, 'corpus')
    persist_dir = os.path.join(project_root, 'vector_db')

    # Initialize pipeline
    pipeline = RAGPipeline(
        corpus_dir=corpus_dir,
        persist_dir=persist_dir
    )

    # Load corpus
    pipeline.load_corpus()

    # Test queries
    test_queries = [
        "How do I deploy to GitHub Pages?",
        "How should I describe my projects on my CV?",
        "Can you ask me an interview question about event listeners?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)

        retrieved_docs, routing_info = pipeline.process_query(query)

        print(f"Category: {routing_info['category']}")
        print(f"Confidence: {routing_info['confidence']}")
        print(f"Multi-category: {routing_info['needs_multiple_categories']}")
        print(f"\nRetrieved {len(retrieved_docs)} documents:")

        for i, doc in enumerate(retrieved_docs[:2], 1):
            print(f"\n  {i}. Source: {doc['metadata']['source']}")
            print(f"     Preview: {doc['content'][:100]}...")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    test_rag_pipeline()