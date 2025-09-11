import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class VectorProjectionEngine:
    """Handles document positioning and projection calculations along custom vectors"""
    
    def __init__(self):
        self.projection_cache = {}
        
    def project_documents_onto_vector(self, embeddings: np.ndarray, custom_vector: np.ndarray, 
                                     metadata: List[Dict]) -> Dict:
        """Project document embeddings onto a custom vector"""
        try:
            # Normalize the custom vector
            normalized_vector = custom_vector / np.linalg.norm(custom_vector)
            
            # Calculate projections (dot product)
            projections = np.dot(embeddings, normalized_vector)
            
            # Create results with metadata
            results = []
            for i, projection in enumerate(projections):
                if i < len(metadata):
                    result = {
                        'projection_score': float(projection),
                        'document_metadata': metadata[i],
                        'paragraph_index': i
                    }
                    results.append(result)
            
            # Sort by projection score
            results.sort(key=lambda x: x['projection_score'], reverse=True)
            
            # Calculate statistics
            stats = {
                'mean_projection': float(np.mean(projections)),
                'std_projection': float(np.std(projections)),
                'min_projection': float(np.min(projections)),
                'max_projection': float(np.max(projections)),
                'median_projection': float(np.median(projections)),
                'q25_projection': float(np.percentile(projections, 25)),
                'q75_projection': float(np.percentile(projections, 75)),
                'total_documents': len(projections)
            }
            
            return {
                'projections': results,
                'statistics': stats,
                'raw_projections': projections
            }
            
        except Exception as e:
            logger.error(f"Error projecting documents onto vector: {str(e)}")
            return {}
    
    def create_2d_vector_space(self, embeddings: np.ndarray, vector1: np.ndarray, 
                              vector2: np.ndarray, metadata: List[Dict]) -> Dict:
        """Create a 2D space defined by two custom vectors"""
        try:
            # Normalize vectors
            v1_norm = vector1 / np.linalg.norm(vector1)
            v2_norm = vector2 / np.linalg.norm(vector2)
            
            # Check if vectors are orthogonal (optional warning)
            dot_product = np.dot(v1_norm, v2_norm)
            orthogonality = abs(dot_product)
            
            # Project onto both vectors
            projections_v1 = np.dot(embeddings, v1_norm)
            projections_v2 = np.dot(embeddings, v2_norm)
            
            # Create 2D coordinates
            coordinates = []
            for i, (x, y) in enumerate(zip(projections_v1, projections_v2)):
                if i < len(metadata):
                    coord = {
                        'x': float(x),
                        'y': float(y),
                        'document_metadata': metadata[i],
                        'paragraph_index': i,
                        'quadrant': self._get_quadrant(x, y)
                    }
                    coordinates.append(coord)
            
            # Calculate quadrant statistics
            quadrant_stats = self._calculate_quadrant_stats(coordinates)
            
            return {
                'coordinates': coordinates,
                'vector1_projections': projections_v1,
                'vector2_projections': projections_v2,
                'orthogonality_score': float(1 - orthogonality),  # 1 = perfectly orthogonal
                'quadrant_statistics': quadrant_stats,
                'space_bounds': {
                    'x_min': float(np.min(projections_v1)),
                    'x_max': float(np.max(projections_v1)),
                    'y_min': float(np.min(projections_v2)),
                    'y_max': float(np.max(projections_v2))
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating 2D vector space: {str(e)}")
            return {}
    
    def create_3d_vector_space(self, embeddings: np.ndarray, vector1: np.ndarray, 
                              vector2: np.ndarray, vector3: np.ndarray, 
                              metadata: List[Dict]) -> Dict:
        """Create a 3D space defined by three custom vectors"""
        try:
            # Normalize vectors
            v1_norm = vector1 / np.linalg.norm(vector1)
            v2_norm = vector2 / np.linalg.norm(vector2)
            v3_norm = vector3 / np.linalg.norm(vector3)
            
            # Project onto all three vectors
            projections_v1 = np.dot(embeddings, v1_norm)
            projections_v2 = np.dot(embeddings, v2_norm)
            projections_v3 = np.dot(embeddings, v3_norm)
            
            # Create 3D coordinates
            coordinates = []
            for i, (x, y, z) in enumerate(zip(projections_v1, projections_v2, projections_v3)):
                if i < len(metadata):
                    coord = {
                        'x': float(x),
                        'y': float(y),
                        'z': float(z),
                        'document_metadata': metadata[i],
                        'paragraph_index': i,
                        'octant': self._get_octant(x, y, z)
                    }
                    coordinates.append(coord)
            
            # Calculate octant statistics
            octant_stats = self._calculate_octant_stats(coordinates)
            
            return {
                'coordinates': coordinates,
                'vector1_projections': projections_v1,
                'vector2_projections': projections_v2,
                'vector3_projections': projections_v3,
                'octant_statistics': octant_stats,
                'space_bounds': {
                    'x_min': float(np.min(projections_v1)),
                    'x_max': float(np.max(projections_v1)),
                    'y_min': float(np.min(projections_v2)),
                    'y_max': float(np.max(projections_v2)),
                    'z_min': float(np.min(projections_v3)),
                    'z_max': float(np.max(projections_v3))
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating 3D vector space: {str(e)}")
            return {}
    
    def _get_quadrant(self, x: float, y: float) -> str:
        """Determine which quadrant a 2D point belongs to"""
        if x >= 0 and y >= 0:
            return "Q1 (+,+)"
        elif x < 0 and y >= 0:
            return "Q2 (-,+)"
        elif x < 0 and y < 0:
            return "Q3 (-,-)"
        else:
            return "Q4 (+,-)"
    
    def _get_octant(self, x: float, y: float, z: float) -> str:
        """Determine which octant a 3D point belongs to"""
        signs = ('+' if x >= 0 else '-', '+' if y >= 0 else '-', '+' if z >= 0 else '-')
        return f"O({signs[0]},{signs[1]},{signs[2]})"
    
    def _calculate_quadrant_stats(self, coordinates: List[Dict]) -> Dict:
        """Calculate statistics for each quadrant"""
        quadrants = {}
        total = len(coordinates)
        
        for coord in coordinates:
            quad = coord['quadrant']
            if quad not in quadrants:
                quadrants[quad] = {'count': 0, 'percentage': 0, 'documents': []}
            quadrants[quad]['count'] += 1
            quadrants[quad]['documents'].append(coord)
        
        # Calculate percentages
        for quad in quadrants:
            quadrants[quad]['percentage'] = (quadrants[quad]['count'] / total) * 100
        
        return quadrants
    
    def _calculate_octant_stats(self, coordinates: List[Dict]) -> Dict:
        """Calculate statistics for each octant"""
        octants = {}
        total = len(coordinates)
        
        for coord in coordinates:
            oct = coord['octant']
            if oct not in octants:
                octants[oct] = {'count': 0, 'percentage': 0, 'documents': []}
            octants[oct]['count'] += 1
            octants[oct]['documents'].append(coord)
        
        # Calculate percentages
        for oct in octants:
            octants[oct]['percentage'] = (octants[oct]['count'] / total) * 100
        
        return octants
    
    def calculate_relative_positioning(self, embeddings: np.ndarray, metadata: List[Dict],
                                     core_embeddings: np.ndarray, core_metadata: List[Dict],
                                     custom_vector: np.ndarray) -> Dict:
        """Calculate document positions relative to core corpus along a vector"""
        try:
            # Project both target and core embeddings onto the vector
            target_projections = self.project_documents_onto_vector(embeddings, custom_vector, metadata)
            core_projections = self.project_documents_onto_vector(core_embeddings, custom_vector, core_metadata)
            
            if not target_projections or not core_projections:
                return {}
            
            # Get projection arrays
            target_scores = target_projections['raw_projections']
            core_scores = core_projections['raw_projections']
            
            # Calculate percentiles relative to core corpus
            relative_positions = []
            for i, score in enumerate(target_scores):
                percentile = (np.sum(core_scores <= score) / len(core_scores)) * 100
                
                if i < len(metadata):
                    relative_positions.append({
                        'projection_score': float(score),
                        'core_percentile': float(percentile),
                        'document_metadata': metadata[i],
                        'paragraph_index': i,
                        'position_category': self._categorize_position(percentile)
                    })
            
            # Sort by core percentile
            relative_positions.sort(key=lambda x: x['core_percentile'], reverse=True)
            
            return {
                'relative_positions': relative_positions,
                'target_stats': target_projections['statistics'],
                'core_stats': core_projections['statistics'],
                'comparison_stats': {
                    'target_mean_vs_core': float(np.mean(target_scores) - np.mean(core_scores)),
                    'target_std_vs_core': float(np.std(target_scores) / np.std(core_scores)),
                    'overlap_coefficient': self._calculate_overlap(target_scores, core_scores)
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating relative positioning: {str(e)}")
            return {}
    
    def _categorize_position(self, percentile: float) -> str:
        """Categorize position based on percentile"""
        if percentile >= 90:
            return "Far Above Core"
        elif percentile >= 75:
            return "Above Core"
        elif percentile >= 25:
            return "Within Core Range"
        elif percentile >= 10:
            return "Below Core"
        else:
            return "Far Below Core"
    
    def _calculate_overlap(self, array1: np.ndarray, array2: np.ndarray) -> float:
        """Calculate overlap coefficient between two distributions"""
        try:
            # Simple overlap calculation using histogram overlap
            min_val = min(np.min(array1), np.min(array2))
            max_val = max(np.max(array1), np.max(array2))
            
            # Create histograms
            bins = np.linspace(min_val, max_val, 50)
            hist1, _ = np.histogram(array1, bins=bins, density=True)
            hist2, _ = np.histogram(array2, bins=bins, density=True)
            
            # Calculate overlap
            overlap = np.sum(np.minimum(hist1, hist2)) / np.sum(np.maximum(hist1, hist2))
            return float(overlap)
            
        except Exception:
            return 0.0
    
    def find_extreme_documents(self, projection_results: Dict, n_extreme: int = 5) -> Dict:
        """Find documents at the extremes of vector projections"""
        try:
            if 'projections' not in projection_results:
                return {}
            
            projections = projection_results['projections']
            
            # Sort by projection score
            sorted_projections = sorted(projections, key=lambda x: x['projection_score'])
            
            return {
                'most_positive': sorted_projections[-n_extreme:],
                'most_negative': sorted_projections[:n_extreme],
                'most_neutral': self._find_neutral_documents(projections, n_extreme)
            }
            
        except Exception as e:
            logger.error(f"Error finding extreme documents: {str(e)}")
            return {}
    
    def _find_neutral_documents(self, projections: List[Dict], n_neutral: int) -> List[Dict]:
        """Find documents closest to zero projection (most neutral)"""
        try:
            # Sort by absolute value of projection score
            sorted_by_abs = sorted(projections, key=lambda x: abs(x['projection_score']))
            return sorted_by_abs[:n_neutral]
        except Exception:
            return []
    
    def analyze_vector_performance(self, projection_results: Dict) -> Dict:
        """Analyze how well a vector discriminates between documents"""
        try:
            if 'raw_projections' not in projection_results:
                return {}
            
            projections = projection_results['raw_projections']
            
            # Calculate discrimination metrics
            discrimination_stats = {
                'range': float(np.max(projections) - np.min(projections)),
                'variance': float(np.var(projections)),
                'coefficient_of_variation': float(np.std(projections) / abs(np.mean(projections))) if np.mean(projections) != 0 else float('inf'),
                'quartile_range': float(np.percentile(projections, 75) - np.percentile(projections, 25)),
                'normality_test': self._test_normality(projections)
            }
            
            # Categorize performance
            performance_category = self._categorize_vector_performance(discrimination_stats)
            
            return {
                'discrimination_stats': discrimination_stats,
                'performance_category': performance_category,
                'recommendations': self._get_vector_recommendations(discrimination_stats)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing vector performance: {str(e)}")
            return {}
    
    def _test_normality(self, data: np.ndarray) -> Dict:
        """Simple normality test using skewness and kurtosis"""
        try:
            from scipy import stats
            skewness = float(stats.skew(data))
            kurtosis = float(stats.kurtosis(data))
            
            return {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'is_normal_ish': abs(skewness) < 1 and abs(kurtosis) < 3
            }
        except ImportError:
            # Fallback if scipy not available
            return {
                'skewness': 0.0,
                'kurtosis': 0.0,
                'is_normal_ish': True
            }
    
    def _categorize_vector_performance(self, stats: Dict) -> str:
        """Categorize vector performance based on discrimination stats"""
        cv = stats['coefficient_of_variation']
        variance = stats['variance']
        
        if cv > 1.0 and variance > 0.1:
            return "Excellent Discrimination"
        elif cv > 0.5 and variance > 0.05:
            return "Good Discrimination"
        elif cv > 0.3 and variance > 0.02:
            return "Moderate Discrimination"
        else:
            return "Poor Discrimination"
    
    def _get_vector_recommendations(self, stats: Dict) -> List[str]:
        """Get recommendations for improving vector performance"""
        recommendations = []
        
        if stats['coefficient_of_variation'] < 0.3:
            recommendations.append("Consider using more contrasting terms to increase discrimination")
        
        if stats['range'] < 0.5:
            recommendations.append("Vector may be too subtle - try stronger positive/negative terms")
        
        if not stats['normality_test']['is_normal_ish']:
            recommendations.append("Distribution is skewed - consider if this reflects real patterns")
        
        if stats['variance'] < 0.02:
            recommendations.append("Low variance suggests documents are very similar on this dimension")
        
        if not recommendations:
            recommendations.append("Vector shows good discrimination across documents")
        
        return recommendations