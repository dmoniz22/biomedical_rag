"""
Embedding service for generating and managing text embeddings
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using transformer models"""
    
    def __init__(self):
        self.model = None
        self.model_name = settings.SENTENCE_TRANSFORMER_MODEL
        self.embedding_dimension = settings.EMBEDDING_DIMENSION
        
    async def initialize(self):
        """Initialize the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                self._load_model
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            # Use CPU if CUDA not available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer(self.model_name, device=device)
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self._embed_text_sync,
                text
            )
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise
    
    def _embed_text_sync(self, text: str) -> List[float]:
        """Synchronous text embedding"""
        try:
            # Encode the text
            embedding = self.model.encode([text])[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error in text embedding: {e}")
            raise
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a search query"""
        return await self.embed_text(query)
    
    async def embed_keywords(self, keywords: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple keywords"""
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self._embed_keywords_sync,
                keywords
            )
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Failed to embed keywords: {e}")
            raise
    
    def _embed_keywords_sync(self, keywords: List[str]) -> List[np.ndarray]:
        """Synchronous keyword embeddings"""
        try:
            embeddings = self.model.encode(keywords)
            return [emb for emb in embeddings]
        except Exception as e:
            logger.error(f"Error in keyword embeddings: {e}")
            raise
    
    async def generate_paper_embeddings(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate embeddings for different parts of a paper"""
        try:
            embeddings = {}
            
            # Title embedding
            if paper_data.get('title'):
                embeddings['title'] = await self.embed_text(paper_data['title'])
            
            # Abstract embedding
            if paper_data.get('abstract'):
                embeddings['abstract'] = await self.embed_text(paper_data['abstract'])
            
            # MeSH terms embedding
            if paper_data.get('mesh_terms'):
                mesh_text = ' '.join(paper_data['mesh_terms'])
                embeddings['mesh_terms'] = await self.embed_text(mesh_text)
            
            # Full text embeddings (chunked)
            if paper_data.get('full_text'):
                # Split into chunks for large documents
                full_text = paper_data['full_text']
                chunks = self._split_text_into_chunks(full_text)
                
                full_text_chunks = []
                for chunk in chunks:
                    chunk_embedding = await self.embed_text(chunk)
                    full_text_chunks.append({
                        'text': chunk,
                        'embedding': chunk_embedding
                    })
                
                embeddings['full_text_chunks'] = full_text_chunks
            
            logger.info(f"Generated embeddings for paper with {len(embeddings)} content types")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate paper embeddings: {e}")
            raise
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks for embedding"""
        try:
            chunk_size = settings.CHUNK_SIZE
            overlap = settings.CHUNK_OVERLAP
            
            words = text.split()
            chunks = []
            
            # Create overlapping chunks
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk = ' '.join(words[start:end])
                chunks.append(chunk)
                
                # Move start position with overlap
                start = end - overlap if end < len(words) else end
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting text into chunks: {e}")
            # Return original text as single chunk if splitting fails
            return [text]
    
    async def similarity_search(self, query_embedding: List[float], 
                               document_embeddings: List[List[float]], 
                               top_k: int = 10) -> List[Dict[str, Any]]:
        """Find most similar documents using cosine similarity"""
        try:
            # Convert to numpy arrays
            query_emb = np.array(query_embedding)
            doc_embs = np.array(document_embeddings)
            
            # Calculate cosine similarity
            similarities = np.dot(doc_embs, query_emb) / (
                np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(query_emb)
            )
            
            # Get top k results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append({
                    'similarity_score': float(similarities[idx]),
                    'document_index': int(idx)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise


# Global embedding service instance
embedding_service = EmbeddingService()