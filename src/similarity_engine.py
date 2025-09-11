import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import umap
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class SimilarityEngine:
    """Handles similarity calculations and vector space analysis"""
    
    def __init__(self):
        self.pca_model = None
        self.umap_model = None
        self.nn_model = None
        
    def calculate_cosine_similarity(self, embeddings1: np.ndarray, 
                                   embeddings2: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between two sets of embeddings
        
        Args:
            embeddings1: First set of embeddings (n_samples1, embedding_dim)
            embeddings2: Second set of embeddings (n_samples2, embedding_dim)
            
        Returns:
            Similarity matrix (n_samples1, n_samples2)
        """
        try:
            similarity_matrix = cosine_similarity(embeddings1, embeddings2)
            logger.info(f"Calculated similarity matrix shape: {similarity_matrix.shape}")
            return similarity_matrix
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            raise
    
    def find_most_similar(self, query_embeddings: np.ndarray, 
                         reference_embeddings: np.ndarray,
                         k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k most similar embeddings for each query
        
        Args:
            query_embeddings: Query embeddings
            reference_embeddings: Reference embeddings to search in
            k: Number of similar items to return
            
        Returns:
            indices of most similar items, similarity scores
        """
        try:
            # Calculate similarity matrix
            similarity_matrix = self.calculate_cosine_similarity(
                query_embeddings, reference_embeddings
            )
            
            # Find top k most similar for each query
            top_k_indices = np.argpartition(similarity_matrix, -k, axis=1)[:, -k:]
            top_k_scores = np.take_along_axis(similarity_matrix, top_k_indices, axis=1)
            
            # Sort by similarity score (descending)
            sort_indices = np.argsort(-top_k_scores, axis=1)
            top_k_indices = np.take_along_axis(top_k_indices, sort_indices, axis=1)
            top_k_scores = np.take_along_axis(top_k_scores, sort_indices, axis=1)
            
            logger.info(f"Found top {k} similar items for {len(query_embeddings)} queries")
            return top_k_indices, top_k_scores
            
        except Exception as e:
            logger.error(f"Error finding most similar: {str(e)}")
            raise
    
    def setup_nearest_neighbors(self, embeddings: np.ndarray, 
                               n_neighbors: int = 10) -> bool:
        """Setup nearest neighbors model for efficient similarity search"""
        try:
            self.nn_model = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric='cosine',
                algorithm='auto'
            )
            self.nn_model.fit(embeddings)
            logger.info(f"Nearest neighbors model fitted with {len(embeddings)} samples")
            return True
        except Exception as e:
            logger.error(f"Error setting up nearest neighbors: {str(e)}")
            return False
    
    def find_nearest_neighbors(self, query_embeddings: np.ndarray,
                              k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Find nearest neighbors using fitted model"""
        if self.nn_model is None:
            raise RuntimeError("Nearest neighbors model not fitted")
        
        try:
            distances, indices = self.nn_model.kneighbors(query_embeddings, n_neighbors=k)
            # Convert cosine distance to similarity
            similarities = 1 - distances
            return indices, similarities
        except Exception as e:
            logger.error(f"Error finding nearest neighbors: {str(e)}")
            raise
    
    def fit_pca(self, embeddings: np.ndarray, n_components: int = 50) -> bool:
        """Fit PCA for dimensionality reduction"""
        try:
            self.pca_model = PCA(n_components=n_components, random_state=42)
            self.pca_model.fit(embeddings)
            
            explained_variance = np.sum(self.pca_model.explained_variance_ratio_)
            logger.info(f"PCA fitted: {n_components} components explain {explained_variance:.3f} of variance")
            return True
        except Exception as e:
            logger.error(f"Error fitting PCA: {str(e)}")
            return False
    
    def transform_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings using fitted PCA"""
        if self.pca_model is None:
            raise RuntimeError("PCA model not fitted")
        
        try:
            transformed = self.pca_model.transform(embeddings)
            logger.info(f"PCA transformed embeddings shape: {transformed.shape}")
            return transformed
        except Exception as e:
            logger.error(f"Error transforming with PCA: {str(e)}")
            raise
    
    def fit_umap(self, embeddings: np.ndarray, n_components: int = 2,
                 n_neighbors: int = 15, min_dist: float = 0.1) -> bool:
        """Fit UMAP for dimensionality reduction"""
        try:
            self.umap_model = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=42,
                metric='cosine'
            )
            self.umap_model.fit(embeddings)
            logger.info(f"UMAP fitted with {n_components} components")
            return True
        except Exception as e:
            logger.error(f"Error fitting UMAP: {str(e)}")
            return False
    
    def transform_umap(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings using fitted UMAP"""
        if self.umap_model is None:
            raise RuntimeError("UMAP model not fitted")
        
        try:
            transformed = self.umap_model.transform(embeddings)
            logger.info(f"UMAP transformed embeddings shape: {transformed.shape}")
            return transformed
        except Exception as e:
            logger.error(f"Error transforming with UMAP: {str(e)}")
            raise
    
    def get_similarity_statistics(self, similarity_matrix: np.ndarray) -> Dict:
        """Calculate statistics for similarity matrix"""
        try:
            stats = {
                'mean_similarity': float(np.mean(similarity_matrix)),
                'std_similarity': float(np.std(similarity_matrix)),
                'min_similarity': float(np.min(similarity_matrix)),
                'max_similarity': float(np.max(similarity_matrix)),
                'median_similarity': float(np.median(similarity_matrix)),
                'q25_similarity': float(np.percentile(similarity_matrix, 25)),
                'q75_similarity': float(np.percentile(similarity_matrix, 75))
            }
            logger.info("Calculated similarity statistics")
            return stats
        except Exception as e:
            logger.error(f"Error calculating similarity statistics: {str(e)}")
            return {}