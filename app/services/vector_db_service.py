"""
Vector database service for storing and retrieving embeddings
"""

import logging
import os
import json
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorDBService:
    """Service for managing vector database operations"""
    
    def __init__(self):
        self.client = None
        self.db_path = settings.VECTOR_DB_PATH
        self.collection_name = "biomed_papers"
        
    async def initialize(self):
        """Initialize the vector database"""
        try:
            logger.info(f"Initializing vector database at {self.db_path}")
            
            # Ensure directory exists
            os.makedirs(self.db_path, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=settings.SENTENCE_TRANSFORMER_MODEL
                    )
                )
                logger.info("Loaded existing collection")
            except Exception:
                # Create new collection
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=settings.SENTENCE_TRANSFORMER_MODEL
                    ),
                    metadata={"description": "Biomedical literature embeddings"}
                )
                logger.info("Created new collection")
            
            logger.info("Vector database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            raise
    
    async def store_embeddings(self, paper_id: str, content_type: str, 
                              embeddings_data: Dict[str, Any], model_name: str):
        """Store embeddings for a paper"""
        try:
            logger.info(f"Storing embeddings for paper {paper_id}, content type: {content_type}")
            
            if content_type == "chunk":
                # Handle chunked embeddings
                for i, chunk_data in enumerate(embeddings_data):
                    doc_id = f"{paper_id}_{content_type}_{i}"
                    text = chunk_data['text']
                    embedding = chunk_data['embedding']
                    
                    # Add to collection
                    self.collection.add(
                        documents=[text],
                        embeddings=[embedding],
                        metadatas=[{
                            "paper_id": paper_id,
                            "content_type": content_type,
                            "chunk_index": i,
                            "model_name": model_name
                        }],
                        ids=[doc_id]
                    )
            else:
                # Handle single embedding
                text = embeddings_data['text']
                embedding = embeddings_data['embedding']
                doc_id = f"{paper_id}_{content_type}"
                
                # Add to collection
                self.collection.add(
                    documents=[text],
                    embeddings=[embedding],
                    metadatas=[{
                        "paper_id": paper_id,
                        "content_type": content_type,
                        "model_name": model_name
                    }],
                    ids=[doc_id]
                )
            
            logger.info(f"Successfully stored embeddings for paper {paper_id}")
            
        except Exception as e:
            logger.error(f"Failed to store embeddings for paper {paper_id}: {e}")
            raise
    
    async def search_similar_papers(self, query_embedding: List[float], 
                                   content_type: str, top_k: int = 10, 
                                   threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar papers using embeddings"""
        try:
            logger.info(f"Searching for similar papers with content type: {content_type}")
            
            # Convert query embedding to numpy array
            query_emb = np.array(query_embedding).reshape(1, -1)
            
            # Search in vector database
            results = self.collection.query(
                query_embeddings=query_emb.tolist(),
                n_results=top_k,
                where={"content_type": content_type}
            )
            
            # Format results
            search_results = []
            for i, doc_id in enumerate(results['ids'][0]):
                # Get metadata
                metadata = results['metadatas'][0][i]
                document = results['documents'][0][i]
                distance = results['distances'][0][i]
                
                # Convert distance to similarity score (Chroma uses distance, not similarity)
                similarity_score = 1 - distance
                
                # Filter by threshold
                if similarity_score >= threshold:
                    search_results.append({
                        'paper_id': metadata['paper_id'],
                        'content': document,
                        'similarity_score': similarity_score,
                        'metadata': metadata
                    })
            
            logger.info(f"Found {len(search_results)} similar papers above threshold")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching similar papers: {e}")
            raise
    
    async def get_paper_embeddings(self, paper_id: str) -> List[Dict[str, Any]]:
        """Get all embeddings for a specific paper"""
        try:
            # Query for all documents with the given paper_id
            results = self.collection.get(
                where={"paper_id": paper_id}
            )
            
            embeddings = []
            for i, doc_id in enumerate(results['ids']):
                embeddings.append({
                    'id': doc_id,
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i],
                    'embedding': results['embeddings'][i] if 'embeddings' in results else None
                })
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings for paper {paper_id}: {e}")
            raise
    
    async def delete_paper_embeddings(self, paper_id: str) -> bool:
        """Delete all embeddings for a specific paper"""
        try:
            # Get all embeddings for the paper
            embeddings = await self.get_paper_embeddings(paper_id)
            
            if not embeddings:
                logger.info(f"No embeddings found for paper {paper_id}")
                return True
            
            # Get all document IDs
            doc_ids = [emb['id'] for emb in embeddings]
            
            # Delete from collection
            self.collection.delete(
                ids=doc_ids
            )
            
            logger.info(f"Deleted {len(doc_ids)} embeddings for paper {paper_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embeddings for paper {paper_id}: {e}")
            raise
    
    async def update_paper_embeddings(self, paper_id: str, content_type: str,
                                     embeddings_data: Dict[str, Any], model_name: str):
        """Update embeddings for a paper (delete old, add new)"""
        try:
            # Delete existing embeddings
            await self.delete_paper_embeddings(paper_id)
            
            # Store new embeddings
            await self.store_embeddings(paper_id, content_type, embeddings_data, model_name)
            
            logger.info(f"Successfully updated embeddings for paper {paper_id}")
            
        except Exception as e:
            logger.error(f"Failed to update embeddings for paper {paper_id}: {e}")
            raise
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database collection"""
        try:
            # Get collection count
            count = self.collection.count()
            
            # Get sample documents to analyze
            sample = self.collection.peek(limit=100)
            
            # Analyze content types
            content_types = {}
            if sample['metadatas']:
                for metadata in sample['metadatas']:
                    content_type = metadata.get('content_type', 'unknown')
                    content_types[content_type] = content_types.get(content_type, 0) + 1
            
            return {
                "total_documents": count,
                "sample_size": len(sample['ids']) if sample['ids'] else 0,
                "content_types": content_types,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            raise
    
    async def search_papers_by_metadata(self, filters: Dict[str, Any], 
                                       limit: int = 100) -> List[Dict[str, Any]]:
        """Search papers using metadata filters"""
        try:
            # Convert filters to ChromaDB where clause
            where_clause = {}
            for key, value in filters.items():
                where_clause[key] = value
            
            # Query collection
            results = self.collection.get(
                where=where_clause,
                limit=limit
            )
            
            papers = []
            for i, doc_id in enumerate(results['ids']):
                papers.append({
                    'id': doc_id,
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
            
            return papers
            
        except Exception as e:
            logger.error(f"Error searching papers by metadata: {e}")
            raise
    
    async def cleanup_old_embeddings(self, days_old: int = 30) -> int:
        """Clean up embeddings older than specified days"""
        try:
            # This is a placeholder - in a real implementation you would
            # track creation timestamps and delete old embeddings
            # For now, just return 0
            logger.info(f"Cleanup requested for embeddings older than {days_old} days")
            logger.warning("Cleanup not implemented - returning 0")
            return 0
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise


# Global vector database service instance
vector_db_service = VectorDBService()
