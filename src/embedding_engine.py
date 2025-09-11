import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import streamlit as st
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """Handles all embedding generation, caching, and similarity calculations"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding engine with specified model
        
        Args:
            model_name: Name of sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self.cache_dir = Path("embeddings_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    @st.cache_resource
    def load_model(_self, model_name: str) -> SentenceTransformer:
        """Load and cache the sentence transformer model"""
        try:
            logger.info(f"Loading model: {model_name}")
            model = SentenceTransformer(model_name)
            logger.info(f"Model loaded successfully. Embedding dimension: {model.get_sentence_embedding_dimension()}")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    def initialize_model(self) -> bool:
        """Initialize the embedding model"""
        try:
            self.model = self.load_model(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            return False
    
    def get_cache_key(self, texts: List[str], model_name: str) -> str:
        """Generate cache key for embedding storage"""
        import hashlib
        # Create hash from model name and text content
        content_hash = hashlib.md5(
            (model_name + "".join(texts)).encode('utf-8')
        ).hexdigest()
        return f"{model_name}_{content_hash}"
    
    def save_embeddings_to_cache(self, cache_key: str, embeddings: np.ndarray, 
                                 metadata: Dict) -> bool:
        """Save embeddings to cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            cache_data = {
                'embeddings': embeddings,
                'metadata': metadata,
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'created_at': datetime.now().isoformat()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Embeddings cached to {cache_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching embeddings: {str(e)}")
            return False
    
    def load_embeddings_from_cache(self, cache_key: str) -> Tuple[bool, Optional[np.ndarray], Optional[Dict]]:
        """Load embeddings from cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if not cache_file.exists():
                return False, None, None
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache data
            if (cache_data.get('model_name') != self.model_name or 
                cache_data.get('embedding_dim') != self.embedding_dim):
                logger.warning("Cache model mismatch, will regenerate embeddings")
                return False, None, None
            
            logger.info(f"Loaded embeddings from cache: {cache_file}")
            return True, cache_data['embeddings'], cache_data['metadata']
            
        except Exception as e:
            logger.error(f"Error loading cached embeddings: {str(e)}")
            return False, None, None
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32,
                                  show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts with batching"""
        if not self.model:
            if not self.initialize_model():
                raise RuntimeError("Failed to initialize embedding model")
        
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Use sentence transformer's built-in batching
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def embed_corpus(self, corpus_data: Dict, corpus_type: str = "corpus",
                     use_cache: bool = True) -> Tuple[bool, Optional[np.ndarray], Optional[Dict]]:
        """
        Generate embeddings for an entire corpus
        
        Args:
            corpus_data: Processed corpus data from DataHandler
            corpus_type: Type identifier for caching
            use_cache: Whether to use cached embeddings
            
        Returns:
            success, embeddings array, embedding metadata
        """
        try:
            # Extract all paragraphs and create metadata
            all_paragraphs = []
            paragraph_metadata = []
            
            for filename, doc_data in corpus_data['documents'].items():
                for para_idx, paragraph in enumerate(doc_data['paragraphs']):
                    all_paragraphs.append(paragraph)
                    paragraph_metadata.append({
                        'filename': filename,
                        'paragraph_index': para_idx,
                        'document_metadata': doc_data['metadata'],
                        'text_length': len(paragraph)
                    })
            
            if not all_paragraphs:
                return False, None, None
            
            # Check cache first
            cache_key = self.get_cache_key(all_paragraphs, self.model_name) + f"_{corpus_type}"
            
            if use_cache:
                cached, embeddings, metadata = self.load_embeddings_from_cache(cache_key)
                if cached:
                    return True, embeddings, metadata
            
            # Generate embeddings
            embeddings = self.generate_embeddings_batch(all_paragraphs)
            
            # Create comprehensive metadata
            embedding_metadata = {
                'paragraph_metadata': paragraph_metadata,
                'corpus_metadata': corpus_data['metadata'],
                'total_paragraphs': len(all_paragraphs),
                'embedding_dimension': self.embedding_dim,
                'model_name': self.model_name,
                'corpus_type': corpus_type,
                'generated_at': datetime.now().isoformat()
            }
            
            # Cache embeddings
            if use_cache:
                self.save_embeddings_to_cache(cache_key, embeddings, embedding_metadata)
            
            # Clean up memory
            gc.collect()
            
            return True, embeddings, embedding_metadata
            
        except Exception as e:
            logger.error(f"Error embedding corpus: {str(e)}")
            return False, None, None