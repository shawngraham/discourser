import streamlit as st
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .embedding_engine import EmbeddingEngine
from .similarity_engine import SimilarityEngine
from .vector_analysis_engine import VectorAnalysisEngine
from .custom_vector_manager import CustomVectorManager
from .vector_projection_engine import VectorProjectionEngine
import logging

logger = logging.getLogger(__name__)

class AnalysisCoordinator:
    """Coordinates embedding generation and similarity analysis with combined corpus support"""
    
    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
        self.similarity_engine = SimilarityEngine()
        self.vector_analysis_engine = VectorAnalysisEngine()
        self.custom_vector_manager = CustomVectorManager()
        self.vector_projection_engine = VectorProjectionEngine()
        
        self.core_embeddings = None
        self.target_embeddings = None
        self.combined_embeddings = None  # New: combined corpus embeddings
        self.core_metadata = None
        self.target_metadata = None
        self.combined_metadata = None  # New: combined corpus metadata
        self.suggested_terms = None
        self.combined_suggested_terms = None  # New: terms from combined corpus
        
    def initialize_analysis(self, model_name: str = "all-MiniLM-L6-v2") -> bool:
        """Initialize the analysis engines"""
        try:
            self.embedding_engine.model_name = model_name
            success = self.embedding_engine.initialize_model()
            if success:
                # Initialize custom vector manager with embedding engine
                self.custom_vector_manager.embedding_engine = self.embedding_engine
                logger.info("Analysis coordinator initialized successfully")
            return success
        except Exception as e:
            logger.error(f"Error initializing analysis: {str(e)}")
            return False
    
    def process_core_corpus(self, core_corpus: Dict, 
                           progress_callback=None) -> Tuple[bool, str]:
        """Process core corpus and generate embeddings"""
        try:
            if progress_callback:
                progress_callback("Generating core corpus embeddings...")
            
            success, embeddings, metadata = self.embedding_engine.embed_corpus(
                core_corpus, corpus_type="core"
            )
            
            if success:
                self.core_embeddings = embeddings
                self.core_metadata = metadata
                
                # Setup nearest neighbors for efficient search
                self.similarity_engine.setup_nearest_neighbors(embeddings)
                
                if progress_callback:
                    progress_callback("Analyzing corpus for term suggestions...")
                
                # Extract suggested terms for vector creation
                self.suggested_terms = self.vector_analysis_engine.extract_corpus_terms(core_corpus)
                
                if progress_callback:
                    progress_callback("Core corpus processing complete!")
                
                return True, f"Successfully processed {len(embeddings)} core paragraphs"
            else:
                return False, "Failed to generate core corpus embeddings"
                
        except Exception as e:
            error_msg = f"Error processing core corpus: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def process_target_corpus(self, target_corpus: Dict,
                             progress_callback=None) -> Tuple[bool, str]:
        """Process target corpus and generate embeddings"""
        try:
            if self.core_embeddings is None:
                return False, "Core corpus must be processed first"
            
            if progress_callback:
                progress_callback("Generating target corpus embeddings...")
            
            success, embeddings, metadata = self.embedding_engine.embed_corpus(
                target_corpus, corpus_type="target"
            )
            
            if success:
                self.target_embeddings = embeddings
                self.target_metadata = metadata
                
                if progress_callback:
                    progress_callback("Target corpus processing complete!")
                
                return True, f"Successfully processed {len(embeddings)} target paragraphs"
            else:
                return False, "Failed to generate target corpus embeddings"
                
        except Exception as e:
            error_msg = f"Error processing target corpus: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def create_combined_corpus_embeddings(self, progress_callback=None) -> Tuple[bool, str]:
        """Create combined corpus embeddings and metadata for vector analysis"""
        try:
            if self.core_embeddings is None or self.target_embeddings is None:
                return False, "Both core and target corpora must be processed first"
            
            if progress_callback:
                progress_callback("Creating combined corpus embeddings...")
            
            # Combine embeddings
            self.combined_embeddings = np.vstack([self.core_embeddings, self.target_embeddings])
            
            # Combine metadata with corpus labels
            combined_paragraph_metadata = []
            
            # Add core metadata with corpus labels
            for meta in self.core_metadata['paragraph_metadata']:
                combined_meta = meta.copy()
                combined_meta['corpus_type'] = 'core'
                combined_meta['global_index'] = len(combined_paragraph_metadata)
                combined_paragraph_metadata.append(combined_meta)
            
            # Add target metadata with corpus labels
            for meta in self.target_metadata['paragraph_metadata']:
                combined_meta = meta.copy()
                combined_meta['corpus_type'] = 'target'
                combined_meta['global_index'] = len(combined_paragraph_metadata)
                combined_paragraph_metadata.append(combined_meta)
            
            # Create combined metadata structure
            self.combined_metadata = {
                'paragraph_metadata': combined_paragraph_metadata,
                'corpus_metadata': {
                    'core': self.core_metadata['corpus_metadata'],
                    'target': self.target_metadata['corpus_metadata']
                },
                'total_paragraphs': len(combined_paragraph_metadata),
                'core_paragraphs': len(self.core_metadata['paragraph_metadata']),
                'target_paragraphs': len(self.target_metadata['paragraph_metadata']),
                'embedding_dimension': self.embedding_engine.embedding_dim,  
                'model_name': self.embedding_engine.model_name,
                'corpus_type': 'combined',
                'generated_at': self.core_metadata.get('generated_at')
            }
            
            if progress_callback:
                progress_callback("Analyzing combined corpus for term suggestions...")
            
            # Extract terms from combined corpus data
            combined_corpus_data = self._create_combined_corpus_data()
            self.combined_suggested_terms = self.vector_analysis_engine.extract_corpus_terms(combined_corpus_data)
            
            if progress_callback:
                progress_callback("Combined corpus processing complete!")
            
            return True, f"Successfully created combined corpus with {len(self.combined_embeddings)} paragraphs"
            
        except Exception as e:
            error_msg = f"Error creating combined corpus: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _create_combined_corpus_data(self) -> Dict:
        """Create combined corpus data structure for term extraction"""
        try:
            combined_documents = {}
            
            # Add core corpus documents with prefix
            if 'core_corpus' in st.session_state:
                for filename, doc_data in st.session_state.core_corpus['documents'].items():
                    prefixed_name = f"core_{filename}"
                    combined_doc = doc_data.copy()
                    combined_doc['metadata']['corpus_type'] = 'core'
                    combined_documents[prefixed_name] = combined_doc
            
            # Add target corpus documents with prefix
            if 'target_corpus' in st.session_state:
                for filename, doc_data in st.session_state.target_corpus['documents'].items():
                    prefixed_name = f"target_{filename}"
                    combined_doc = doc_data.copy()
                    combined_doc['metadata']['corpus_type'] = 'target'
                    combined_documents[prefixed_name] = combined_doc
            
            # Calculate total paragraphs
            total_paragraphs = sum(doc['paragraph_count'] for doc in combined_documents.values())
            
            combined_corpus_data = {
                'documents': combined_documents,
                'metadata': {
                    'core_docs': len([d for d in combined_documents.values() if d['metadata']['corpus_type'] == 'core']),
                    'target_docs': len([d for d in combined_documents.values() if d['metadata']['corpus_type'] == 'target']),
                    'total_docs': len(combined_documents)
                },
                'total_paragraphs': total_paragraphs
            }
            
            return combined_corpus_data
            
        except Exception as e:
            logger.error(f"Error creating combined corpus data: {str(e)}")
            return {'documents': {}, 'metadata': {}, 'total_paragraphs': 0}
    
    def get_suggested_terms(self, corpus_type: str = "core", top_n: int = 20) -> List[Dict]:
        """Get suggested terms for vector creation from specified corpus"""
        if corpus_type == "core" and self.suggested_terms:
            return self.vector_analysis_engine.get_suggested_vector_endpoints(self.suggested_terms, top_n)
        elif corpus_type == "target" and hasattr(self, 'target_suggested_terms') and self.target_suggested_terms:
            return self.vector_analysis_engine.get_suggested_vector_endpoints(self.target_suggested_terms, top_n)
        elif corpus_type == "combined" and self.combined_suggested_terms:
            return self.vector_analysis_engine.get_suggested_vector_endpoints(self.combined_suggested_terms, top_n)
        return []
    
    def create_custom_vector_from_corpus(self, vector_name: str, positive_terms: List[str], 
                                       negative_terms: List[str] = None, 
                                       description: str = "", 
                                       method: str = "orthogonal_projection",
                                       corpus_type: str = "core") -> Tuple[bool, str]:
        """Create a custom vector from terms, using specified corpus for context"""
        
        # Temporarily set the embedding engine to use appropriate corpus context
        original_suggested_terms = self.custom_vector_manager.embedding_engine
        
        if corpus_type == "combined" and self.combined_embeddings is not None:
            # Use combined corpus context for better term representation
            pass  # The embedding engine will use the global model context
        
        return self.custom_vector_manager.create_vector_from_terms(
            vector_name, positive_terms, negative_terms, description, method
        )
    
    def analyze_corpus_overlap_on_vector(self, vector_name: str) -> Tuple[bool, Optional[Dict], str]:
        """Analyze how core and target corpora overlap along a custom vector"""
        try:
            if self.combined_embeddings is None:
                return False, None, "Combined corpus embeddings not available. Create combined corpus first."
            
            custom_vector = self.custom_vector_manager.get_vector(vector_name)
            if custom_vector is None:
                return False, None, f"Vector '{vector_name}' not found"
            
            # Project all documents onto the vector
            combined_projections = self.vector_projection_engine.project_documents_onto_vector(
                self.combined_embeddings, custom_vector, self.combined_metadata['paragraph_metadata']
            )
            
            if not combined_projections:
                return False, None, "Failed to calculate projections"
            
            # Separate projections by corpus type
            core_projections = []
            target_projections = []
            
            for proj in combined_projections['projections']:
                if proj['document_metadata']['corpus_type'] == 'core':
                    core_projections.append(proj)
                else:
                    target_projections.append(proj)
            
            # Calculate overlap statistics
            core_scores = [p['projection_score'] for p in core_projections]
            target_scores = [p['projection_score'] for p in target_projections]
            
            overlap_analysis = {
                'vector_name': vector_name,
                'vector_info': self.custom_vector_manager.get_vector_info(vector_name),
                'core_projections': core_projections,
                'target_projections': target_projections,
                'core_statistics': {
                    'mean': float(np.mean(core_scores)),
                    'std': float(np.std(core_scores)),
                    'min': float(np.min(core_scores)),
                    'max': float(np.max(core_scores)),
                    'count': len(core_scores)
                },
                'target_statistics': {
                    'mean': float(np.mean(target_scores)),
                    'std': float(np.std(target_scores)),
                    'min': float(np.min(target_scores)),
                    'max': float(np.max(target_scores)),
                    'count': len(target_scores)
                },
                'overlap_metrics': self._calculate_overlap_metrics(core_scores, target_scores),
                'distribution_comparison': self._analyze_distribution_differences(core_scores, target_scores)
            }
            
            return True, overlap_analysis, "Overlap analysis complete"
            
        except Exception as e:
            error_msg = f"Error analyzing corpus overlap: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def _calculate_overlap_metrics(self, core_scores: List[float], target_scores: List[float]) -> Dict:
        """Calculate various overlap metrics between two distributions"""
        try:
            core_array = np.array(core_scores)
            target_array = np.array(target_scores)
            
            # Calculate overlapping range
            core_range = (np.min(core_array), np.max(core_array))
            target_range = (np.min(target_array), np.max(target_array))
            
            overlap_min = max(core_range[0], target_range[0])
            overlap_max = min(core_range[1], target_range[1])
            overlap_range = max(0, overlap_max - overlap_min)
            
            total_range = max(core_range[1], target_range[1]) - min(core_range[0], target_range[0])
            range_overlap_ratio = overlap_range / total_range if total_range > 0 else 0
            
            # Calculate distribution overlap using histogram method
            all_scores = np.concatenate([core_array, target_array])
            bins = np.linspace(np.min(all_scores), np.max(all_scores), 50)
            
            core_hist, _ = np.histogram(core_array, bins=bins, density=True)
            target_hist, _ = np.histogram(target_array, bins=bins, density=True)
            
            # Bhattacharyya coefficient (histogram overlap)
            bhattacharyya = np.sum(np.sqrt(core_hist * target_hist)) * (bins[1] - bins[0])
            
            # Wasserstein distance (earth mover's distance)
            try:
                from scipy.stats import wasserstein_distance
                wasserstein_dist = wasserstein_distance(core_array, target_array)
            except ImportError:
                wasserstein_dist = abs(np.mean(core_array) - np.mean(target_array))
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(core_array) - 1) * np.var(core_array) + 
                                (len(target_array) - 1) * np.var(target_array)) / 
                               (len(core_array) + len(target_array) - 2))
            cohens_d = (np.mean(core_array) - np.mean(target_array)) / pooled_std if pooled_std > 0 else 0
            
            return {
                'range_overlap_ratio': float(range_overlap_ratio),
                'bhattacharyya_coefficient': float(bhattacharyya),
                'wasserstein_distance': float(wasserstein_dist),
                'cohens_d': float(cohens_d),
                'mean_difference': float(np.mean(core_array) - np.mean(target_array)),
                'overlap_interpretation': self._interpret_overlap_metrics(range_overlap_ratio, bhattacharyya, abs(cohens_d))
            }
            
        except Exception as e:
            logger.error(f"Error calculating overlap metrics: {str(e)}")
            return {'error': str(e)}
    
    def _interpret_overlap_metrics(self, range_overlap: float, bhattacharyya: float, abs_cohens_d: float) -> str:
        """Provide human-readable interpretation of overlap metrics"""
        if range_overlap > 0.8 and bhattacharyya > 0.8 and abs_cohens_d < 0.2:
            return "Very high overlap - corpora are very similar on this dimension"
        elif range_overlap > 0.6 and bhattacharyya > 0.6 and abs_cohens_d < 0.5:
            return "High overlap - corpora show substantial similarity on this dimension"
        elif range_overlap > 0.4 and bhattacharyya > 0.4 and abs_cohens_d < 0.8:
            return "Moderate overlap - corpora have some similarity but clear differences"
        elif range_overlap > 0.2 and bhattacharyya > 0.2 and abs_cohens_d < 1.2:
            return "Low overlap - corpora show distinct patterns on this dimension"
        else:
            return "Very low overlap - corpora are quite different on this dimension"
    
    def _analyze_distribution_differences(self, core_scores: List[float], target_scores: List[float]) -> Dict:
        """Analyze distributional differences between core and target corpora"""
        try:
            from scipy import stats
            
            core_array = np.array(core_scores)
            target_array = np.array(target_scores)
            
            # Statistical tests
            try:
                # Mann-Whitney U test (non-parametric)
                u_statistic, u_p_value = stats.mannwhitneyu(core_array, target_array, alternative='two-sided')
                
                # Kolmogorov-Smirnov test (distribution shape)
                ks_statistic, ks_p_value = stats.ks_2samp(core_array, target_array)
                
                statistical_tests = {
                    'mann_whitney_u': {'statistic': float(u_statistic), 'p_value': float(u_p_value)},
                    'kolmogorov_smirnov': {'statistic': float(ks_statistic), 'p_value': float(ks_p_value)}
                }
            except:
                statistical_tests = {'error': 'Statistical tests not available'}
            
            # Descriptive statistics comparison
            percentiles = [10, 25, 50, 75, 90]
            core_percentiles = {f'p{p}': float(np.percentile(core_array, p)) for p in percentiles}
            target_percentiles = {f'p{p}': float(np.percentile(target_array, p)) for p in percentiles}
            
            return {
                'statistical_tests': statistical_tests,
                'core_percentiles': core_percentiles,
                'target_percentiles': target_percentiles,
                'skewness': {
                    'core': float(stats.skew(core_array)),
                    'target': float(stats.skew(target_array))
                },
                'kurtosis': {
                    'core': float(stats.kurtosis(core_array)),
                    'target': float(stats.kurtosis(target_array))
                }
            }
            
        except ImportError:
            return {'error': 'scipy not available for statistical analysis'}
        except Exception as e:
            logger.error(f"Error in distribution analysis: {str(e)}")
            return {'error': str(e)}
    
    def create_2d_corpus_comparison_space(self, vector1_name: str, vector2_name: str) -> Tuple[bool, Optional[Dict], str]:
        """Create a 2D space showing how both corpora are distributed"""
        try:
            if self.combined_embeddings is None:
                return False, None, "Combined corpus embeddings not available"
            
            vector1 = self.custom_vector_manager.get_vector(vector1_name)
            vector2 = self.custom_vector_manager.get_vector(vector2_name)
            
            if vector1 is None or vector2 is None:
                return False, None, "One or both vectors not found"
            
            # Create 2D space with combined embeddings
            space_results = self.vector_projection_engine.create_2d_vector_space(
                self.combined_embeddings, vector1, vector2, self.combined_metadata['paragraph_metadata']
            )
            
            # Separate by corpus type for comparison
            core_coordinates = []
            target_coordinates = []
            
            for coord in space_results['coordinates']:
                if coord['document_metadata']['corpus_type'] == 'core':
                    core_coordinates.append(coord)
                else:
                    target_coordinates.append(coord)
            
            # Add corpus comparison information
            space_results.update({
                'vector1_info': self.custom_vector_manager.get_vector_info(vector1_name),
                'vector2_info': self.custom_vector_manager.get_vector_info(vector2_name),
                'core_coordinates': core_coordinates,
                'target_coordinates': target_coordinates,
                'corpus_comparison': {
                    'core_count': len(core_coordinates),
                    'target_count': len(target_coordinates),
                    'core_centroid': {
                        'x': float(np.mean([c['x'] for c in core_coordinates])),
                        'y': float(np.mean([c['y'] for c in core_coordinates]))
                    },
                    'target_centroid': {
                        'x': float(np.mean([c['x'] for c in target_coordinates])),
                        'y': float(np.mean([c['y'] for c in target_coordinates]))
                    }
                }
            })
            
            return True, space_results, "2D corpus comparison space created successfully"
            
        except Exception as e:
            error_msg = f"Error creating 2D corpus comparison: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg

    def get_analysis_summary(self) -> Dict:
        """Get summary of current analysis state"""
        summary = {
            'model_name': self.embedding_engine.model_name,
            'embedding_dimension': self.embedding_engine.embedding_dim,
            'core_corpus_processed': self.core_embeddings is not None,
            'target_corpus_processed': self.target_embeddings is not None,
            'combined_corpus_processed': self.combined_embeddings is not None,
            'suggested_terms_available': self.suggested_terms is not None,
            'combined_suggested_terms_available': self.combined_suggested_terms is not None,
            'custom_vectors_count': len(self.custom_vector_manager.custom_vectors)
        }
        
        if self.core_embeddings is not None:
            summary.update({
                'core_paragraphs': len(self.core_embeddings),
                'core_documents': len(set(meta['filename'] for meta in self.core_metadata['paragraph_metadata']))
            })
        
        if self.target_embeddings is not None:
            summary.update({
                'target_paragraphs': len(self.target_embeddings),
                'target_documents': len(set(meta['filename'] for meta in self.target_metadata['paragraph_metadata']))
            })
        
        if self.combined_embeddings is not None:
            summary.update({
                'combined_paragraphs': len(self.combined_embeddings),
                'combined_documents': self.combined_metadata['core_paragraphs'] + self.combined_metadata['target_paragraphs']
            })
        
        return summary

    # Add these methods to your existing AnalysisCoordinator class:

    def analyze_document_projections(self, vector_name: str, 
                                   use_target: bool = True) -> Tuple[bool, Optional[Dict], str]:
        """Analyze how documents project onto a custom vector"""
        try:
            custom_vector = self.custom_vector_manager.get_vector(vector_name)
            if custom_vector is None:
                return False, None, f"Vector '{vector_name}' not found"
            
            if use_target:
                if self.target_embeddings is None:
                    return False, None, "Target corpus not processed"
                embeddings = self.target_embeddings
                metadata = self.target_metadata['paragraph_metadata']
            else:
                if self.core_embeddings is None:
                    return False, None, "Core corpus not processed"
                embeddings = self.core_embeddings
                metadata = self.core_metadata['paragraph_metadata']
            
            # Project documents onto vector
            projection_results = self.vector_projection_engine.project_documents_onto_vector(
                embeddings, custom_vector, metadata
            )
            
            # Analyze vector performance
            performance_analysis = self.vector_projection_engine.analyze_vector_performance(projection_results)
            
            # Find extreme documents
            extreme_docs = self.vector_projection_engine.find_extreme_documents(projection_results)
            
            combined_results = {
                'projection_results': projection_results,
                'performance_analysis': performance_analysis,
                'extreme_documents': extreme_docs,
                'vector_info': self.custom_vector_manager.get_vector_info(vector_name)
            }
            
            return True, combined_results, "Projection analysis complete"
            
        except Exception as e:
            error_msg = f"Error analyzing projections: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg

    def analyze_corpus_overlap_on_vector(self, vector_name: str) -> Tuple[bool, Optional[Dict], str]:
        """Analyze how core and target corpora overlap along a custom vector"""
        try:
            if self.combined_embeddings is None:
                return False, None, "Combined corpus embeddings not available. Create combined corpus first."
            
            custom_vector = self.custom_vector_manager.get_vector(vector_name)
            if custom_vector is None:
                return False, None, f"Vector '{vector_name}' not found"
            
            # Project all documents onto the vector
            combined_projections = self.vector_projection_engine.project_documents_onto_vector(
                self.combined_embeddings, custom_vector, self.combined_metadata['paragraph_metadata']
            )
            
            if not combined_projections:
                return False, None, "Failed to calculate projections"
            
            # Separate projections by corpus type
            core_projections = []
            target_projections = []
            
            for proj in combined_projections['projections']:
                if proj['document_metadata']['corpus_type'] == 'core':
                    core_projections.append(proj)
                else:
                    target_projections.append(proj)
            
            # Calculate overlap statistics
            core_scores = [p['projection_score'] for p in core_projections]
            target_scores = [p['projection_score'] for p in target_projections]
            
            overlap_analysis = {
                'vector_name': vector_name,
                'vector_info': self.custom_vector_manager.get_vector_info(vector_name),
                'core_projections': core_projections,
                'target_projections': target_projections,
                'core_statistics': {
                    'mean': float(np.mean(core_scores)),
                    'std': float(np.std(core_scores)),
                    'min': float(np.min(core_scores)),
                    'max': float(np.max(core_scores)),
                    'count': len(core_scores)
                },
                'target_statistics': {
                    'mean': float(np.mean(target_scores)),
                    'std': float(np.std(target_scores)),
                    'min': float(np.min(target_scores)),
                    'max': float(np.max(target_scores)),
                    'count': len(target_scores)
                },
                'overlap_metrics': self._calculate_overlap_metrics(core_scores, target_scores),
                'distribution_comparison': self._analyze_distribution_differences(core_scores, target_scores)
            }
            
            return True, overlap_analysis, "Overlap analysis complete"
            
        except Exception as e:
            error_msg = f"Error analyzing corpus overlap: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg

    def _calculate_overlap_metrics(self, core_scores: List[float], target_scores: List[float]) -> Dict:
        """Calculate various overlap metrics between two distributions"""
        try:
            core_array = np.array(core_scores)
            target_array = np.array(target_scores)
            
            # Calculate overlapping range
            core_range = (np.min(core_array), np.max(core_array))
            target_range = (np.min(target_array), np.max(target_array))
            
            overlap_min = max(core_range[0], target_range[0])
            overlap_max = min(core_range[1], target_range[1])
            overlap_range = max(0, overlap_max - overlap_min)
            
            total_range = max(core_range[1], target_range[1]) - min(core_range[0], target_range[0])
            range_overlap_ratio = overlap_range / total_range if total_range > 0 else 0
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(core_array) - 1) * np.var(core_array) + 
                                (len(target_array) - 1) * np.var(target_array)) / 
                               (len(core_array) + len(target_array) - 2))
            cohens_d = (np.mean(core_array) - np.mean(target_array)) / pooled_std if pooled_std > 0 else 0
            
            return {
                'range_overlap_ratio': float(range_overlap_ratio),
                'cohens_d': float(cohens_d),
                'mean_difference': float(np.mean(core_array) - np.mean(target_array)),
                'overlap_interpretation': self._interpret_overlap_metrics(range_overlap_ratio, abs(cohens_d))
            }
            
        except Exception as e:
            logger.error(f"Error calculating overlap metrics: {str(e)}")
            return {'error': str(e)}

    def _interpret_overlap_metrics(self, range_overlap: float, abs_cohens_d: float) -> str:
        """Provide human-readable interpretation of overlap metrics"""
        if range_overlap > 0.8 and abs_cohens_d < 0.2:
            return "Very high overlap - corpora are very similar on this dimension"
        elif range_overlap > 0.6 and abs_cohens_d < 0.5:
            return "High overlap - corpora show substantial similarity on this dimension"
        elif range_overlap > 0.4 and abs_cohens_d < 0.8:
            return "Moderate overlap - corpora have some similarity but clear differences"
        elif range_overlap > 0.2 and abs_cohens_d < 1.2:
            return "Low overlap - corpora show distinct patterns on this dimension"
        else:
            return "Very low overlap - corpora are quite different on this dimension"

    def _analyze_distribution_differences(self, core_scores: List[float], target_scores: List[float]) -> Dict:
        """Analyze distributional differences between core and target corpora"""
        try:
            core_array = np.array(core_scores)
            target_array = np.array(target_scores)
            
            # Basic statistical comparison
            percentiles = [10, 25, 50, 75, 90]
            core_percentiles = {f'p{p}': float(np.percentile(core_array, p)) for p in percentiles}
            target_percentiles = {f'p{p}': float(np.percentile(target_array, p)) for p in percentiles}
            
            # Try to import scipy for statistical tests
            try:
                from scipy import stats
                
                # Mann-Whitney U test (non-parametric)
                u_statistic, u_p_value = stats.mannwhitneyu(core_array, target_array, alternative='two-sided')
                
                # Kolmogorov-Smirnov test (distribution shape)
                ks_statistic, ks_p_value = stats.ks_2samp(core_array, target_array)
                
                statistical_tests = {
                    'mann_whitney_u': {'statistic': float(u_statistic), 'p_value': float(u_p_value)},
                    'kolmogorov_smirnov': {'statistic': float(ks_statistic), 'p_value': float(ks_p_value)}
                }
            except ImportError:
                statistical_tests = {'error': 'scipy not available for statistical tests'}
            except Exception:
                statistical_tests = {'error': 'statistical tests failed'}
            
            return {
                'statistical_tests': statistical_tests,
                'core_percentiles': core_percentiles,
                'target_percentiles': target_percentiles,
                'skewness': {
                    'core': float(np.mean((core_array - np.mean(core_array))**3) / (np.std(core_array)**3)) if len(core_array) > 2 else 0.0,
                    'target': float(np.mean((target_array - np.mean(target_array))**3) / (np.std(target_array)**3)) if len(target_array) > 2 else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in distribution analysis: {str(e)}")
            return {'error': str(e)}

    def create_2d_vector_space(self, vector1_name: str, vector2_name: str,
                              use_target: bool = True, use_combined: bool = False) -> Tuple[bool, Optional[Dict], str]:
        """Create a 2D space defined by two vectors with combined corpus support"""
        try:
            vector1 = self.custom_vector_manager.get_vector(vector1_name)
            vector2 = self.custom_vector_manager.get_vector(vector2_name)
            
            if vector1 is None or vector2 is None:
                return False, None, "One or both vectors not found"
            
            if use_combined:
                # Use combined corpus
                if self.combined_embeddings is None:
                    return False, None, "Combined corpus not available. Create combined corpus first."
                embeddings = self.combined_embeddings
                metadata = self.combined_metadata['paragraph_metadata']
            elif use_target:
                if self.target_embeddings is None:
                    return False, None, "Target corpus not processed"
                embeddings = self.target_embeddings
                metadata = self.target_metadata['paragraph_metadata']
            else:
                if self.core_embeddings is None:
                    return False, None, "Core corpus not processed"
                embeddings = self.core_embeddings
                metadata = self.core_metadata['paragraph_metadata']
            
            space_results = self.vector_projection_engine.create_2d_vector_space(
                embeddings, vector1, vector2, metadata
            )
            
            space_results.update({
                'vector1_info': self.custom_vector_manager.get_vector_info(vector1_name),
                'vector2_info': self.custom_vector_manager.get_vector_info(vector2_name)
            })
            
            # Add combined corpus analysis if using combined embeddings
            if use_combined:
                space_results.update(self._add_combined_corpus_analysis_2d(space_results))
            
            corpus_type = "combined" if use_combined else ("target" if use_target else "core")
            return True, space_results, f"2D vector space created successfully using {corpus_type} corpus"
            
        except Exception as e:
            error_msg = f"Error creating 2D vector space: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg

    def create_3d_vector_space(self, vector1_name: str, vector2_name: str, vector3_name: str,
                              use_target: bool = True, use_combined: bool = False) -> Tuple[bool, Optional[Dict], str]:
        """Create a 3D space defined by three vectors with combined corpus support"""
        try:
            vector1 = self.custom_vector_manager.get_vector(vector1_name)
            vector2 = self.custom_vector_manager.get_vector(vector2_name)
            vector3 = self.custom_vector_manager.get_vector(vector3_name)
            
            if vector1 is None or vector2 is None or vector3 is None:
                return False, None, "One or more vectors not found"
            
            if use_combined:
                # Use combined corpus
                if self.combined_embeddings is None:
                    return False, None, "Combined corpus not available. Create combined corpus first."
                embeddings = self.combined_embeddings
                metadata = self.combined_metadata['paragraph_metadata']
            elif use_target:
                if self.target_embeddings is None:
                    return False, None, "Target corpus not processed"
                embeddings = self.target_embeddings
                metadata = self.target_metadata['paragraph_metadata']
            else:
                if self.core_embeddings is None:
                    return False, None, "Core corpus not processed"
                embeddings = self.core_embeddings
                metadata = self.core_metadata['paragraph_metadata']
            
            space_results = self.vector_projection_engine.create_3d_vector_space(
                embeddings, vector1, vector2, vector3, metadata
            )
            
            space_results.update({
                'vector1_info': self.custom_vector_manager.get_vector_info(vector1_name),
                'vector2_info': self.custom_vector_manager.get_vector_info(vector2_name),
                'vector3_info': self.custom_vector_manager.get_vector_info(vector3_name)
            })
            
            # Add combined corpus analysis if using combined embeddings
            if use_combined:
                space_results.update(self._add_combined_corpus_analysis_3d(space_results))
            
            corpus_type = "combined" if use_combined else ("target" if use_target else "core")
            return True, space_results, f"3D vector space created successfully using {corpus_type} corpus"
            
        except Exception as e:
            error_msg = f"Error creating 3D vector space: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg

    def _add_combined_corpus_analysis_2d(self, space_results: Dict) -> Dict:
        """Add combined corpus analysis to 2D space results"""
        try:
            coordinates = space_results['coordinates']
            
            # Separate coordinates by corpus type
            core_coordinates = []
            target_coordinates = []
            
            for coord in coordinates:
                if coord['document_metadata']['corpus_type'] == 'core':
                    core_coordinates.append(coord)
                else:
                    target_coordinates.append(coord)
            
            # Calculate centroids
            if core_coordinates and target_coordinates:
                core_centroid = {
                    'x': float(np.mean([c['x'] for c in core_coordinates])),
                    'y': float(np.mean([c['y'] for c in core_coordinates]))
                }
                target_centroid = {
                    'x': float(np.mean([c['x'] for c in target_coordinates])),
                    'y': float(np.mean([c['y'] for c in target_coordinates]))
                }
                
                # Calculate distance between centroids
                centroid_distance = np.sqrt(
                    (core_centroid['x'] - target_centroid['x'])**2 +
                    (core_centroid['y'] - target_centroid['y'])**2
                )
                
                # Calculate spread for each corpus
                core_spread = np.sqrt(
                    np.var([c['x'] for c in core_coordinates]) +
                    np.var([c['y'] for c in core_coordinates])
                )
                target_spread = np.sqrt(
                    np.var([c['x'] for c in target_coordinates]) +
                    np.var([c['y'] for c in target_coordinates])
                )
                
                return {
                    'corpus_comparison': {
                        'core_count': len(core_coordinates),
                        'target_count': len(target_coordinates),
                        'core_centroid': core_centroid,
                        'target_centroid': target_centroid,
                        'centroid_distance': float(centroid_distance),
                        'core_spread': float(core_spread),
                        'target_spread': float(target_spread),
                        'relative_spread_ratio': float(min(core_spread, target_spread) / max(core_spread, target_spread)) if max(core_spread, target_spread) > 0 else 0
                    },
                    'core_coordinates': core_coordinates,
                    'target_coordinates': target_coordinates
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error in combined corpus 2D analysis: {str(e)}")
            return {}

    def _add_combined_corpus_analysis_3d(self, space_results: Dict) -> Dict:
        """Add combined corpus analysis to 3D space results"""
        try:
            coordinates = space_results['coordinates']
            
            # Separate coordinates by corpus type
            core_coordinates = []
            target_coordinates = []
            
            for coord in coordinates:
                if coord['document_metadata']['corpus_type'] == 'core':
                    core_coordinates.append(coord)
                else:
                    target_coordinates.append(coord)
            
            # Calculate centroids
            if core_coordinates and target_coordinates:
                core_centroid = {
                    'x': float(np.mean([c['x'] for c in core_coordinates])),
                    'y': float(np.mean([c['y'] for c in core_coordinates])),
                    'z': float(np.mean([c['z'] for c in core_coordinates]))
                }
                target_centroid = {
                    'x': float(np.mean([c['x'] for c in target_coordinates])),
                    'y': float(np.mean([c['y'] for c in target_coordinates])),
                    'z': float(np.mean([c['z'] for c in target_coordinates]))
                }
                
                # Calculate distance between centroids
                centroid_distance = np.sqrt(
                    (core_centroid['x'] - target_centroid['x'])**2 +
                    (core_centroid['y'] - target_centroid['y'])**2 +
                    (core_centroid['z'] - target_centroid['z'])**2
                )
                
                # Calculate spread for each corpus
                core_spread = np.sqrt(
                    np.var([c['x'] for c in core_coordinates]) +
                    np.var([c['y'] for c in core_coordinates]) +
                    np.var([c['z'] for c in core_coordinates])
                )
                target_spread = np.sqrt(
                    np.var([c['x'] for c in target_coordinates]) +
                    np.var([c['y'] for c in target_coordinates]) +
                    np.var([c['z'] for c in target_coordinates])
                )
                
                return {
                    'corpus_comparison': {
                        'core_count': len(core_coordinates),
                        'target_count': len(target_coordinates),
                        'core_centroid': core_centroid,
                        'target_centroid': target_centroid,
                        'centroid_distance': float(centroid_distance),
                        'core_spread': float(core_spread),
                        'target_spread': float(target_spread),
                        'relative_spread_ratio': float(min(core_spread, target_spread) / max(core_spread, target_spread)) if max(core_spread, target_spread) > 0 else 0
                    },
                    'core_coordinates': core_coordinates,
                    'target_coordinates': target_coordinates
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error in combined corpus 3D analysis: {str(e)}")
            return {}

    
    def create_2d_combined_corpus_space(self, vector1_name: str, vector2_name: str) -> Tuple[bool, Optional[Dict], str]:
        """Convenience method to create 2D space with combined corpus"""
        return self.create_2d_vector_space(vector1_name, vector2_name, use_target=False, use_combined=True)

    def create_3d_combined_corpus_space(self, vector1_name: str, vector2_name: str, vector3_name: str) -> Tuple[bool, Optional[Dict], str]:
        """Convenience method to create 3D space with combined corpus"""
        return self.create_3d_vector_space(vector1_name, vector2_name, vector3_name, use_target=False, use_combined=True)
     