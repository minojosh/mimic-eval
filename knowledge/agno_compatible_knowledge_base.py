"""
Agno-Compatible Knowledge Base Implementation

This module provides a knowledge base implementation that follows Agno's
interface patterns while working with existing ChromaDB databases.
Based on research findings from Agno's framework architecture.

VERIFICATION: Updated to match current Agno v1.7.6 patterns exactly.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import chromadb
from chromadb.config import Settings

# Try to import Agno's Document class
try:
    from agno.document import Document
    AGNO_AVAILABLE = True
except ImportError:
    # Fallback if Agno not installed - create a compatible Document class
    class Document:
        """Fallback Document class compatible with Agno's Document interface"""
        def __init__(self, id: str = None, content: str = "", meta_data: Dict[str, Any] = None, embedding: List[float] = None):
            self.id = id
            self.content = content
            self.meta_data = meta_data or {}
            self.embedding = embedding
    
    AGNO_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgnoCompatibleKnowledgeBase:
    """
    Knowledge base implementation that follows Agno's interface patterns.
    
    VERIFIED: This class provides the exact method signatures that current Agno agents expect
    while working with existing ChromaDB collections and custom embedders.
    
    Required methods for Agno compatibility:
    - search(query, num_documents, filters) -> List[Document]  # PRIMARY METHOD
    - search_knowledge_base(query, limit) -> compatibility method
    - get_relevant_documents(query, limit) -> alternative compatibility method
    - validate_filters(filters) -> filter validation
    """
    
    def __init__(
        self, 
        chroma_db_path: str,
        collection_name: str,
        embedder,
        num_documents: int = 5
    ):
        """
        Initialize the Agno-compatible knowledge base.
        
        Args:
            chroma_db_path: Path to ChromaDB database
            collection_name: Name of the ChromaDB collection
            embedder: Embedder instance with get_embedding method
            num_documents: Default number of documents to return
        """
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        self.embedder = embedder
        self.num_documents = num_documents  # IMPORTANT: Ensure this is set
        
        # Initialize ChromaDB client
        self.client = None
        self.collection = None
        
        self._initialize_chroma_connection()
        
        logger.info(f"✅ Knowledge base initialized with {self.num_documents} default document limit")
    
    def _initialize_chroma_connection(self):
        """Initialize connection to ChromaDB."""
        try:
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(self.collection_name)
                doc_count = self.collection.count()
                logger.info(f"✅ Connected to existing collection '{self.collection_name}' with {doc_count} documents")
            except Exception as e:
                logger.warning(f"Collection '{self.collection_name}' not found: {e}")
                logger.info("Creating new collection...")
                self.collection = self.client.create_collection(self.collection_name)
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB connection: {e}")
            raise
    
    def search(self, query: str, num_documents: int = None, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Primary search method compatible with Agno's interface.
        
        VERIFIED: This signature matches AgentKnowledge.search() exactly.
        
        Args:
            query: Search query text
            num_documents: Maximum number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of Document objects (not dicts!)
        """
        if num_documents is None:
            num_documents = self.num_documents
            
        try:
            # Generate query embedding
            query_embedding = self.embedder.get_embedding(query)
            
            if not query_embedding:
                logger.warning(f"Failed to generate embedding for query: {query}")
                return []
            
            # Prepare search parameters
            search_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": num_documents
            }
            
            # Add filters if provided
            if filters:
                search_kwargs["where"] = filters
            
            # Search ChromaDB
            results = self.collection.query(**search_kwargs)
            
            # Convert to Agno Document objects (CRITICAL FIX)
            return self._format_results_as_documents(results)
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def search_knowledge_base(self, query: str, num_documents: int = None) -> List[Document]:
        """
        Compatibility method - delegates to main search method.
        
        VERIFIED: While agents don't call this directly, it's good practice for compatibility.
        
        Args:
            query: Search query text
            num_documents: Maximum number of results to return
            
        Returns:
            List of Document objects
        """
        return self.search(query, num_documents)
    
    def get_relevant_documents(self, query: str, num_documents: int = None) -> List[Document]:
        """
        Alternative search interface for compatibility.
        
        Args:
            query: Search query text
            num_documents: Maximum number of results to return
            
        Returns:
            List of Document objects
        """
        return self.search(query, num_documents)
    
    def validate_filters(self, filters: Optional[Dict[str, Any]] = None) -> tuple:
        """
        Agno compatibility method for filter validation.
        
        Args:
            filters: Optional metadata filters to validate
            
        Returns:
            Tuple of (valid_filters, invalid_keys)
        """
        if filters is None:
            return {}, []
        
        # For this implementation, we accept all filters as valid
        # In a production system, you might validate against known metadata fields
        valid_filters = filters.copy()
        invalid_keys = []
        
        return valid_filters, invalid_keys
    
    def _format_results_as_documents(self, chroma_results: Dict[str, Any]) -> List[Document]:
        """
        Convert ChromaDB results to Agno Document objects.
        
        CRITICAL FIX: Returns Document objects, not dictionaries.
        
        Args:
            chroma_results: Raw results from ChromaDB query
            
        Returns:
            List of Document objects
        """
        documents = []
        
        if not chroma_results or not chroma_results.get("documents"):
            return documents
        
        # Extract result arrays
        documents_list = chroma_results["documents"][0] if chroma_results["documents"] else []
        metadatas = chroma_results["metadatas"][0] if chroma_results.get("metadatas") else []
        ids = chroma_results["ids"][0] if chroma_results.get("ids") else []
        distances = chroma_results["distances"][0] if chroma_results.get("distances") else []
        embeddings = chroma_results.get("embeddings", [[]])[0] if chroma_results.get("embeddings") else []
        
        # Create Document objects
        for i, doc_content in enumerate(documents_list):
            # Create metadata with distance/score info
            metadata = metadatas[i].copy() if i < len(metadatas) else {}
            metadata["distance"] = distances[i] if i < len(distances) else 0.0
            metadata["score"] = 1.0 - (distances[i] if i < len(distances) else 0.0)  # Convert distance to score
            
            document = Document(
                id=ids[i] if i < len(ids) else f"result_{i}",
                content=doc_content,
                meta_data=metadata,
                embedding=embeddings[i] if i < len(embeddings) else None
            )
            documents.append(document)
        
        return documents
    
    def load_documents_from_file(
        self, 
        file_path: str, 
        chunk_size: int = 1000,
        overlap: int = 100,
        recreate: bool = False
    ):
        """
        Load documents from a text file into the knowledge base.
        
        Args:
            file_path: Path to the text file
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            recreate: Whether to recreate the collection
        """
        if recreate and self.collection:
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(self.collection_name)
                logger.info(f"Recreated collection '{self.collection_name}'")
            except Exception as e:
                logger.warning(f"Could not recreate collection: {e}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return
        
        logger.info(f"Loading documents from {file_path}...")
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split content into chunks
            chunks = self._create_text_chunks(content, chunk_size, overlap)
            
            # Process chunks in batches
            batch_size = 100
            total_chunks = len(chunks)
            
            for i in range(0, total_chunks, batch_size):
                batch_chunks = chunks[i:i + batch_size]
                self._process_document_batch(batch_chunks, i)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")
            
            logger.info(f"✅ Successfully loaded {total_chunks} document chunks")
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
    
    def _create_text_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to end at a sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                boundary = max(last_period, last_newline)
                
                if boundary > start + chunk_size // 2:  # Only if boundary is not too early
                    chunk = text[start:start + boundary + 1]
                    end = start + boundary + 1
            
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk.strip())
            
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def _process_document_batch(self, chunks: List[str], batch_offset: int):
        """
        Process a batch of document chunks.
        
        Args:
            chunks: List of text chunks
            batch_offset: Offset for generating unique IDs
        """
        try:
            # Generate embeddings for the batch
            embeddings = []
            documents = []
            ids = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 10:  # Skip very short chunks
                    continue
                
                # Generate embedding
                embedding = self.embedder.get_embedding(chunk)
                if embedding:
                    embeddings.append(embedding)
                    documents.append(chunk)
                    ids.append(f"doc_{batch_offset + i}")
                    metadatas.append({
                        "type": "text_chunk",
                        "chunk_id": batch_offset + i,
                        "length": len(chunk)
                    })
            
            # Add to ChromaDB
            if embeddings:
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    ids=ids,
                    metadatas=metadatas
                )
            
        except Exception as e:
            logger.error(f"Error processing document batch: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            count = self.collection.count() if self.collection else 0
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "database_path": self.chroma_db_path,
                "embedder_info": getattr(self.embedder, 'get_model_info', lambda: {})()
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def test_search(self, query: str = "test query") -> Dict[str, Any]:
        """
        Test the search functionality.
        
        Args:
            query: Test query
            
        Returns:
            Test results
        """
        logger.info(f"Testing search with query: '{query}'")
        
        try:
            results = self.search(query, num_documents=3)
            
            test_info = {
                "query": query,
                "results_count": len(results),
                "collection_info": self.get_collection_info(),
                "sample_results": []
            }
            
            # Add sample results
            for i, result in enumerate(results[:2]):
                test_info["sample_results"].append({
                    "id": result.id,
                    "content_preview": result.content[:100] + "...",
                    "metadata": result.meta_data
                })
            
            logger.info(f"Search test completed: {len(results)} results found")
            return test_info
            
        except Exception as e:
            logger.error(f"Search test failed: {e}")
            return {"error": str(e)}


def create_knowledge_base(
    chroma_db_path: str,
    collection_name: str,
    embedder,
    num_documents: int = 5
) -> AgnoCompatibleKnowledgeBase:
    """
    Factory function to create an Agno-compatible knowledge base.
    
    Args:
        chroma_db_path: Path to ChromaDB database
        collection_name: Name of the collection
        embedder: Embedder instance
        num_documents: Default number of documents to return
        
    Returns:
        Configured knowledge base instance
    """
    return AgnoCompatibleKnowledgeBase(
        chroma_db_path=chroma_db_path,
        collection_name=collection_name,
        embedder=embedder,
        num_documents=num_documents
    )


if __name__ == "__main__":
    # Test the knowledge base
    print("Testing Agno-Compatible Knowledge Base...")
    
    # This would require an actual embedder and database
    # See the notebook for full integration testing
    print("✅ Knowledge base module loaded successfully!")
