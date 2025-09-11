import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class CustomVectorManager:
    """Manages custom vector creation with improved mathematical rigor"""
    
    def __init__(self, embedding_engine=None):
        self.embedding_engine = embedding_engine
        self.custom_vectors = {}
        self.max_vectors = 50
        self.validation_examples = {}  # Store validation examples
        
    def create_vector_from_terms(self, vector_name: str, positive_terms: List[str], 
                                negative_terms: List[str] = None, 
                                description: str = "",
                                method: str = "orthogonal_projection") -> Tuple[bool, str, Optional[np.ndarray]]:
        """
        Create a custom vector using improved mathematical methods
        
        Args:
            vector_name: Name for the vector
            positive_terms: Terms representing positive direction
            negative_terms: Terms representing negative direction  
            description: Vector description
            method: Method for vector composition ('orthogonal_projection', 'weighted_difference', 'pca_axis')
        """
        try:
            if not self.embedding_engine or not hasattr(self.embedding_engine, 'model') or self.embedding_engine.model is None:
                return False, "Embedding model not initialized", None
            
            if len(self.custom_vectors) >= self.max_vectors:
                return False, f"Maximum number of vectors ({self.max_vectors}) reached", None
            
            if vector_name in self.custom_vectors:
                return False, f"Vector '{vector_name}' already exists", None
            
            if not positive_terms:
                return False, "At least one positive term is required", None
            
            # Validate terms exist in model vocabulary
            valid_pos, invalid_pos = self._validate_model_vocabulary(positive_terms)
            if invalid_pos:
                logger.warning(f"Terms not in vocabulary: {invalid_pos}")
            
            if not valid_pos:
                return False, "No valid positive terms found in model vocabulary", None
            
            valid_neg = []
            if negative_terms:
                valid_neg, invalid_neg = self._validate_model_vocabulary(negative_terms)
                if invalid_neg:
                    logger.warning(f"Negative terms not in vocabulary: {invalid_neg}")
            
            # Generate embeddings
            positive_embeddings = self.embedding_engine.generate_embeddings_batch(valid_pos, show_progress=False)
            negative_embeddings = None
            if valid_neg:
                negative_embeddings = self.embedding_engine.generate_embeddings_batch(valid_neg, show_progress=False)
            
            # Create vector using specified method
            if method == "orthogonal_projection":
                custom_vector, metadata = self._create_orthogonal_vector(
                    positive_embeddings, negative_embeddings, valid_pos, valid_neg
                )
            elif method == "weighted_difference":
                custom_vector, metadata = self._create_weighted_difference_vector(
                    positive_embeddings, negative_embeddings, valid_pos, valid_neg
                )
            elif method == "pca_axis":
                custom_vector, metadata = self._create_pca_axis_vector(
                    positive_embeddings, negative_embeddings, valid_pos, valid_neg
                )
            else:
                return False, f"Unknown method: {method}", None
            
            if custom_vector is None:
                return False, "Failed to create vector", None
            
            # Validate vector quality
            quality_score, quality_report = self._assess_vector_quality(
                custom_vector, positive_embeddings, negative_embeddings
            )
            
            # Store vector with comprehensive metadata
            vector_info = {
                'name': vector_name,
                'vector': custom_vector,
                'positive_terms': valid_pos,
                'negative_terms': valid_neg,
                'description': description,
                'method': method,
                'created_at': datetime.now().isoformat(),
                'dimension': len(custom_vector),
                'norm': float(np.linalg.norm(custom_vector)),
                'quality_score': quality_score,
                'quality_report': quality_report,
                'mathematical_metadata': metadata
            }
            
            self.custom_vectors[vector_name] = vector_info
            
            success_msg = f"Vector '{vector_name}' created successfully (Quality: {quality_score:.2f}/1.0)"
            if quality_score < 0.5:
                success_msg += f"\nWarning: Low quality score. {quality_report['recommendations'][0] if quality_report.get('recommendations') else ''}"
            
            logger.info(f"Created vector '{vector_name}' using {method} method")
            return True, success_msg, custom_vector
            
        except Exception as e:
            error_msg = f"Error creating vector: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, None
    
    def _validate_model_vocabulary(self, terms: List[str]) -> Tuple[List[str], List[str]]:
        """Check if terms exist in the model's effective vocabulary"""
        valid_terms = []
        invalid_terms = []
        
        for term in terms:
            term = term.strip()
            if not term:
                continue
                
            # Basic validation
            if len(term) < 2:
                invalid_terms.append(term)
                continue
            
            try:
                # Test if the model can embed the term
                test_embedding = self.embedding_engine.generate_embeddings_batch([term], show_progress=False)
                if test_embedding is not None and len(test_embedding) > 0:
                    # Check if embedding is not just zeros/nulls
                    if np.any(test_embedding[0]):
                        valid_terms.append(term)
                    else:
                        invalid_terms.append(term)
                else:
                    invalid_terms.append(term)
            except Exception:
                invalid_terms.append(term)
        
        return valid_terms, invalid_terms
    
    def _create_orthogonal_vector(self, pos_embeddings: np.ndarray, neg_embeddings: Optional[np.ndarray],
                                 pos_terms: List[str], neg_terms: List[str]) -> Tuple[np.ndarray, Dict]:
        """Create vector using orthogonal projection method"""
        try:
            # Calculate centroids
            pos_centroid = np.mean(pos_embeddings, axis=0)
            
            if neg_embeddings is not None:
                neg_centroid = np.mean(neg_embeddings, axis=0)
                
                # Calculate the difference vector
                diff_vector = pos_centroid - neg_centroid
                
                # Project onto the space orthogonal to the average of both centroids
                combined_centroid = (pos_centroid + neg_centroid) / 2
                
                # Remove component parallel to combined centroid to reduce bias
                parallel_component = np.dot(diff_vector, combined_centroid) / np.dot(combined_centroid, combined_centroid)
                orthogonal_vector = diff_vector - parallel_component * combined_centroid
                
                # If orthogonal vector is too small, fall back to difference
                if np.linalg.norm(orthogonal_vector) < 0.1:
                    custom_vector = diff_vector
                    method_note = "fell back to difference vector"
                else:
                    custom_vector = orthogonal_vector
                    method_note = "used orthogonal projection"
            else:
                # No negative terms - use positive centroid
                custom_vector = pos_centroid
                method_note = "positive centroid only"
            
            # Normalize
            custom_vector = custom_vector / np.linalg.norm(custom_vector)
            
            metadata = {
                'method_details': f"orthogonal_projection: {method_note}",
                'pos_centroid_norm': float(np.linalg.norm(pos_centroid)),
                'neg_centroid_norm': float(np.linalg.norm(neg_embeddings.mean(axis=0))) if neg_embeddings is not None else 0.0,
                'final_norm': float(np.linalg.norm(custom_vector))
            }
            
            return custom_vector, metadata
            
        except Exception as e:
            logger.error(f"Error in orthogonal projection: {str(e)}")
            return None, {}
    
    def _create_weighted_difference_vector(self, pos_embeddings: np.ndarray, neg_embeddings: Optional[np.ndarray],
                                          pos_terms: List[str], neg_terms: List[str]) -> Tuple[np.ndarray, Dict]:
        """Create vector using weighted difference based on term importance"""
        try:
            # Calculate weights based on variance (more variable terms get higher weight)
            pos_weights = np.var(pos_embeddings, axis=1) if len(pos_embeddings) > 1 else np.ones(1)
            pos_weights = pos_weights / np.sum(pos_weights)  # Normalize
            
            # Weighted average of positive terms
            pos_centroid = np.average(pos_embeddings, axis=0, weights=pos_weights)
            
            if neg_embeddings is not None:
                neg_weights = np.var(neg_embeddings, axis=1) if len(neg_embeddings) > 1 else np.ones(1)
                neg_weights = neg_weights / np.sum(neg_weights)
                
                neg_centroid = np.average(neg_embeddings, axis=0, weights=neg_weights)
                
                # Create weighted difference
                # Weight the difference by the ratio of positive to negative terms
                pos_strength = len(pos_terms) / (len(pos_terms) + len(neg_terms))
                neg_strength = len(neg_terms) / (len(pos_terms) + len(neg_terms))
                
                custom_vector = pos_strength * pos_centroid - neg_strength * neg_centroid
            else:
                custom_vector = pos_centroid
            
            # Normalize
            custom_vector = custom_vector / np.linalg.norm(custom_vector)
            
            metadata = {
                'method_details': 'weighted_difference with variance-based weighting',
                'pos_weights': pos_weights.tolist(),
                'neg_weights': neg_weights.tolist() if neg_embeddings is not None else [],
                'pos_strength': pos_strength if neg_embeddings is not None else 1.0,
                'neg_strength': neg_strength if neg_embeddings is not None else 0.0
            }
            
            return custom_vector, metadata
            
        except Exception as e:
            logger.error(f"Error in weighted difference: {str(e)}")
            return None, {}
    
    def _create_pca_axis_vector(self, pos_embeddings: np.ndarray, neg_embeddings: Optional[np.ndarray],
                               pos_terms: List[str], neg_terms: List[str]) -> Tuple[np.ndarray, Dict]:
        """Create vector using PCA to find the primary axis of variation"""
        try:
            # Combine all embeddings
            if neg_embeddings is not None:
                all_embeddings = np.vstack([pos_embeddings, neg_embeddings])
                labels = [1] * len(pos_embeddings) + [-1] * len(neg_embeddings)
            else:
                all_embeddings = pos_embeddings
                labels = [1] * len(pos_embeddings)
            
            if len(all_embeddings) < 2:
                # Fall back to simple centroid
                custom_vector = np.mean(all_embeddings, axis=0)
                custom_vector = custom_vector / np.linalg.norm(custom_vector)
                metadata = {'method_details': 'insufficient data for PCA, used centroid'}
                return custom_vector, metadata
            
            # Center the data
            scaler = StandardScaler()
            centered_embeddings = scaler.fit_transform(all_embeddings)
            
            # Apply PCA
            pca = PCA(n_components=min(len(all_embeddings), 5))
            pca.fit(centered_embeddings)
            
            # Use the first principal component as our vector
            custom_vector = pca.components_[0]
            
            # Ensure the vector points in the direction of positive terms
            if neg_embeddings is not None:
                pos_projection = np.mean([np.dot(emb, custom_vector) for emb in pos_embeddings])
                neg_projection = np.mean([np.dot(emb, custom_vector) for emb in neg_embeddings])
                
                if pos_projection < neg_projection:
                    custom_vector = -custom_vector
            
            # Normalize
            custom_vector = custom_vector / np.linalg.norm(custom_vector)
            
            metadata = {
                'method_details': 'PCA first component',
                'explained_variance_ratio': float(pca.explained_variance_ratio_[0]),
                'total_variance_explained': float(np.sum(pca.explained_variance_ratio_)),
                'n_components_used': len(pca.components_)
            }
            
            return custom_vector, metadata
            
        except Exception as e:
            logger.error(f"Error in PCA axis creation: {str(e)}")
            return None, {}
    
    def _assess_vector_quality(self, custom_vector: np.ndarray, 
                              pos_embeddings: np.ndarray, 
                              neg_embeddings: Optional[np.ndarray]) -> Tuple[float, Dict]:
        """Assess the quality of the created vector"""
        try:
            quality_metrics = {}
            
            # 1. Separation quality: how well does the vector separate positive and negative terms?
            pos_projections = np.dot(pos_embeddings, custom_vector)
            pos_mean = np.mean(pos_projections)
            pos_std = np.std(pos_projections)
            
            if neg_embeddings is not None:
                neg_projections = np.dot(neg_embeddings, custom_vector)
                neg_mean = np.mean(neg_projections)
                neg_std = np.std(neg_projections)
                
                # Calculate separation (Cohen's d-like measure)
                pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
                separation = abs(pos_mean - neg_mean) / (pooled_std + 1e-8)
                
                quality_metrics['separation'] = float(separation)
                quality_metrics['pos_mean_projection'] = float(pos_mean)
                quality_metrics['neg_mean_projection'] = float(neg_mean)
            else:
                # For positive-only vectors, check consistency (low variance is good)
                consistency = 1 / (1 + pos_std)  # Higher is better
                quality_metrics['consistency'] = float(consistency)
                separation = consistency
            
            # 2. Vector stability: check that the vector is not too sensitive to individual terms
            if len(pos_embeddings) > 1:
                # Leave-one-out stability test
                stability_scores = []
                for i in range(len(pos_embeddings)):
                    # Create vector without term i
                    reduced_pos = np.delete(pos_embeddings, i, axis=0)
                    if len(reduced_pos) > 0:
                        reduced_centroid = np.mean(reduced_pos, axis=0)
                        if neg_embeddings is not None:
                            reduced_vector = reduced_centroid - np.mean(neg_embeddings, axis=0)
                        else:
                            reduced_vector = reduced_centroid
                        
                        reduced_vector = reduced_vector / np.linalg.norm(reduced_vector)
                        
                        # Calculate similarity to original vector
                        similarity = np.dot(custom_vector, reduced_vector)
                        stability_scores.append(similarity)
                
                stability = np.mean(stability_scores) if stability_scores else 1.0
                quality_metrics['stability'] = float(stability)
            else:
                stability = 1.0
                quality_metrics['stability'] = 1.0
            
            # 3. Overall quality score (weighted combination)
            quality_score = 0.6 * min(separation / 2.0, 1.0) + 0.4 * stability
            
            # Generate recommendations
            recommendations = []
            if separation < 1.0:
                recommendations.append("Consider using more contrasting terms to improve separation")
            if stability < 0.8:
                recommendations.append("Vector may be unstable - consider adding more terms")
            if quality_score > 0.8:
                recommendations.append("Vector shows good quality metrics")
            
            quality_report = {
                'metrics': quality_metrics,
                'overall_score': float(quality_score),
                'recommendations': recommendations
            }
            
            return quality_score, quality_report
            
        except Exception as e:
            logger.error(f"Error assessing vector quality: {str(e)}")
            return 0.0, {'error': str(e)}
    
    def add_validation_examples(self, vector_name: str, positive_examples: List[str], 
                               negative_examples: List[str] = None) -> Tuple[bool, str]:
        """Add validation examples for a vector to test its effectiveness"""
        try:
            if vector_name not in self.custom_vectors:
                return False, f"Vector '{vector_name}' does not exist"
            
            # Test the vector against the examples
            vector = self.custom_vectors[vector_name]['vector']
            
            if positive_examples:
                pos_embeddings = self.embedding_engine.generate_embeddings_batch(positive_examples, show_progress=False)
                pos_projections = np.dot(pos_embeddings, vector)
                pos_score = np.mean(pos_projections)
            else:
                pos_score = 0.0
            
            if negative_examples:
                neg_embeddings = self.embedding_engine.generate_embeddings_batch(negative_examples, show_progress=False)
                neg_projections = np.dot(neg_embeddings, vector)
                neg_score = np.mean(neg_projections)
            else:
                neg_score = 0.0
            
            validation_result = {
                'positive_examples': positive_examples,
                'negative_examples': negative_examples or [],
                'positive_score': float(pos_score),
                'negative_score': float(neg_score),
                'separation_score': float(pos_score - neg_score),
                'validated_at': datetime.now().isoformat()
            }
            
            self.validation_examples[vector_name] = validation_result
            
            # Update vector info with validation
            self.custom_vectors[vector_name]['validation'] = validation_result
            
            separation = pos_score - neg_score
            if separation > 0.5:
                status = "Vector performs well on validation examples"
            elif separation > 0.0:
                status = "Vector shows some separation on validation examples"
            else:
                status = "Vector may not be working as expected - consider revising terms"
            
            return True, f"{status} (separation: {separation:.3f})"
            
        except Exception as e:
            error_msg = f"Error validating vector: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def validate_terms(self, terms: List[str]) -> Tuple[List[str], List[str]]:
        """Validate terms and return valid/invalid lists"""
        valid_terms = []
        invalid_terms = []
        
        for term in terms:
            term = term.strip()
            if term:
                # Basic validation - you can enhance this
                if len(term) >= 2 and term.replace(' ', '').replace('-', '').isalpha():
                    valid_terms.append(term)
                else:
                    invalid_terms.append(term)
        
        return valid_terms, invalid_terms
    
    def update_vector(self, vector_name: str, positive_terms: List[str], 
                     negative_terms: List[str] = None, 
                     description: str = "") -> Tuple[bool, str]:
        """Update an existing vector"""
        try:
            if vector_name not in self.custom_vectors:
                return False, f"Vector '{vector_name}' does not exist"
            
            # Create new vector with updated terms
            success, message, new_vector = self.create_vector_from_terms(
                f"{vector_name}_temp", positive_terms, negative_terms, description
            )
            
            if success:
                # Update the existing vector
                self.custom_vectors[vector_name].update({
                    'vector': new_vector,
                    'positive_terms': positive_terms,
                    'negative_terms': negative_terms or [],
                    'description': description,
                    'updated_at': datetime.now().isoformat(),
                    'norm': float(np.linalg.norm(new_vector))
                })
                
                # Remove temporary vector
                del self.custom_vectors[f"{vector_name}_temp"]
                
                return True, f"Vector '{vector_name}' updated successfully"
            else:
                return False, message
                
        except Exception as e:
            error_msg = f"Error updating vector: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def delete_vector(self, vector_name: str) -> Tuple[bool, str]:
        """Delete a custom vector"""
        try:
            if vector_name not in self.custom_vectors:
                return False, f"Vector '{vector_name}' does not exist"
            
            del self.custom_vectors[vector_name]
            logger.info(f"Deleted custom vector '{vector_name}'")
            return True, f"Vector '{vector_name}' deleted successfully"
            
        except Exception as e:
            error_msg = f"Error deleting vector: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def get_vector(self, vector_name: str) -> Optional[np.ndarray]:
        """Get a custom vector by name"""
        if vector_name in self.custom_vectors:
            return self.custom_vectors[vector_name]['vector']
        return None
    
    def get_vector_info(self, vector_name: str) -> Optional[Dict]:
        """Get full information about a vector"""
        return self.custom_vectors.get(vector_name)
    
    def list_vectors(self) -> List[Dict]:
        """List all custom vectors with basic info"""
        vector_list = []
        for name, info in self.custom_vectors.items():
            vector_list.append({
                'name': name,
                'description': info['description'],
                'positive_terms_count': len(info['positive_terms']),
                'negative_terms_count': len(info['negative_terms']),
                'created_at': info['created_at'],
                'dimension': info['dimension']
            })
        return vector_list
    
    def calculate_vector_similarity(self, vector1_name: str, vector2_name: str) -> Optional[float]:
        """Calculate similarity between two custom vectors"""
        try:
            vector1 = self.get_vector(vector1_name)
            vector2 = self.get_vector(vector2_name)
            
            if vector1 is None or vector2 is None:
                return None
            
            # Calculate cosine similarity
            similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating vector similarity: {str(e)}")
            return None
    
    def project_embeddings_onto_vector(self, embeddings: np.ndarray, vector_name: str) -> Optional[np.ndarray]:
        """Project embeddings onto a custom vector"""
        try:
            custom_vector = self.get_vector(vector_name)
            if custom_vector is None:
                return None
            
            # Calculate dot product (projection)
            projections = np.dot(embeddings, custom_vector)
            return projections
            
        except Exception as e:
            logger.error(f"Error projecting embeddings: {str(e)}")
            return None
    
    def get_vector_statistics(self, vector_name: str, embeddings: np.ndarray) -> Optional[Dict]:
        """Get statistics about how a vector relates to a set of embeddings"""
        try:
            projections = self.project_embeddings_onto_vector(embeddings, vector_name)
            if projections is None:
                return None
            
            stats = {
                'mean_projection': float(np.mean(projections)),
                'std_projection': float(np.std(projections)),
                'min_projection': float(np.min(projections)),
                'max_projection': float(np.max(projections)),
                'median_projection': float(np.median(projections)),
                'q25_projection': float(np.percentile(projections, 25)),
                'q75_projection': float(np.percentile(projections, 75)),
                'vector_name': vector_name,
                'total_documents': len(projections)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating vector statistics: {str(e)}")
            return None
    
    def export_vectors(self) -> Dict:
        """Export all custom vectors for saving"""
        return {
            'vectors': self.custom_vectors,
            'export_date': datetime.now().isoformat(),
            'version': '1.0'
        }
    
    def import_vectors(self, vector_data: Dict) -> Tuple[bool, str]:
        """Import custom vectors from saved data"""
        try:
            if 'vectors' not in vector_data:
                return False, "Invalid vector data format"
            
            imported_count = 0
            for name, info in vector_data['vectors'].items():
                if name not in self.custom_vectors and len(self.custom_vectors) < self.max_vectors:
                    # Convert vector back to numpy array if it was stored as list
                    if isinstance(info['vector'], list):
                        info['vector'] = np.array(info['vector'])
                    
                    self.custom_vectors[name] = info
                    imported_count += 1
            
            return True, f"Imported {imported_count} vectors successfully"
            
        except Exception as e:
            error_msg = f"Error importing vectors: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def get_vector_creation_methods(self) -> List[Dict]:
        """Get available vector creation methods with descriptions"""
        return [
        {
            'name': 'orthogonal_projection',
            'display_name': 'Orthogonal Projection',
            'description': 'Creates vector orthogonal to bias, reduces overfitting to specific terms',
            'best_for': 'Balanced conceptual dimensions',
            'complexity': 'Medium'
        },
        {
            'name': 'weighted_difference',
            'display_name': 'Weighted Difference',
            'description': 'Uses term importance weighting, emphasizes more variable terms',
            'best_for': 'When some terms are more important than others',
            'complexity': 'High'
        },
        {
            'name': 'pca_axis',
            'display_name': 'PCA First Component',
            'description': 'Finds main axis of variation between positive and negative terms',
            'best_for': 'Clear conceptual opposites with good separation',
            'complexity': 'High'
        }
    ]