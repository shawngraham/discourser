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
    """Coordinates embedding generation and similarity analysis"""
    
    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
        self.similarity_engine = SimilarityEngine()
        self.vector_analysis_engine = VectorAnalysisEngine()
        self.custom_vector_manager = CustomVectorManager()
        self.vector_projection_engine = VectorProjectionEngine()
        
        self.core_embeddings = None
        self.target_embeddings = None
        self.core_metadata = None
        self.target_metadata = None
        self.suggested_terms = None
        
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
    
    def get_suggested_terms(self, top_n: int = 20) -> List[Dict]:
        """Get suggested terms for vector creation"""
        if self.suggested_terms:
            return self.vector_analysis_engine.get_suggested_vector_endpoints(self.suggested_terms, top_n)
        return []
    
    def create_custom_vector(self, vector_name: str, positive_terms: List[str], 
                            negative_terms: List[str] = None, 
                            description: str = "", method: str = "orthogonal_projection") -> Tuple[bool, str]:
        """Create a custom vector from terms"""
        return self.custom_vector_manager.create_vector_from_terms(
            vector_name, positive_terms, negative_terms, description, method
        )
    
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
    
    def create_2d_vector_space(self, vector1_name: str, vector2_name: str,
                              use_target: bool = True) -> Tuple[bool, Optional[Dict], str]:
        """Create a 2D space defined by two vectors"""
        try:
            vector1 = self.custom_vector_manager.get_vector(vector1_name)
            vector2 = self.custom_vector_manager.get_vector(vector2_name)
            
            if vector1 is None or vector2 is None:
                return False, None, "One or both vectors not found"
            
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
            
            space_results = self.vector_projection_engine.create_2d_vector_space(
                embeddings, vector1, vector2, metadata
            )
            
            space_results.update({
                'vector1_info': self.custom_vector_manager.get_vector_info(vector1_name),
                'vector2_info': self.custom_vector_manager.get_vector_info(vector2_name)
            })
            
            return True, space_results, "2D vector space created successfully"
            
        except Exception as e:
            error_msg = f"Error creating 2D vector space: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def create_3d_vector_space(self, vector1_name: str, vector2_name: str, vector3_name: str,
                              use_target: bool = True) -> Tuple[bool, Optional[Dict], str]:
        """Create a 3D space defined by three vectors"""
        try:
            vector1 = self.custom_vector_manager.get_vector(vector1_name)
            vector2 = self.custom_vector_manager.get_vector(vector2_name)
            vector3 = self.custom_vector_manager.get_vector(vector3_name)
            
            if vector1 is None or vector2 is None or vector3 is None:
                return False, None, "One or more vectors not found"
            
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
            
            space_results = self.vector_projection_engine.create_3d_vector_space(
                embeddings, vector1, vector2, vector3, metadata
            )
            
            space_results.update({
                'vector1_info': self.custom_vector_manager.get_vector_info(vector1_name),
                'vector2_info': self.custom_vector_manager.get_vector_info(vector2_name),
                'vector3_info': self.custom_vector_manager.get_vector_info(vector3_name)
            })
            
            return True, space_results, "3D vector space created successfully"
            
        except Exception as e:
            error_msg = f"Error creating 3D vector space: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def calculate_core_target_similarity(self) -> Tuple[bool, Optional[np.ndarray], str]:
        """Calculate similarity between target and core corpus"""
        try:
            if self.core_embeddings is None or self.target_embeddings is None:
                return False, None, "Both corpora must be processed first"
            
            similarity_matrix = self.similarity_engine.calculate_cosine_similarity(
                self.target_embeddings, self.core_embeddings
            )
            
            return True, similarity_matrix, "Similarity calculation successful"
            
        except Exception as e:
            error_msg = f"Error calculating similarity: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def find_most_influential_core_texts(self, k: int = 10) -> Tuple[bool, Optional[Dict], str]:
        """Find most influential core texts based on average similarity"""
        try:
            success, similarity_matrix, msg = self.calculate_core_target_similarity()
            if not success:
                return False, None, msg
            
            # Calculate average similarity for each core paragraph
            avg_similarities = np.mean(similarity_matrix, axis=0)
            
            # Get top k most influential
            top_k_indices = np.argsort(avg_similarities)[-k:][::-1]
            top_k_scores = avg_similarities[top_k_indices]
            
            # Get metadata for top paragraphs
            influential_texts = []
            for i, (idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
                para_meta = self.core_metadata['paragraph_metadata'][idx]
                influential_texts.append({
                    'rank': i + 1,
                    'paragraph_index': idx,
                    'average_similarity': float(score),
                    'filename': para_meta['filename'],
                    'paragraph_in_doc': para_meta['paragraph_index'],
                    'document_title': para_meta['document_metadata']['title'],
                    'document_author': para_meta['document_metadata']['author']
                })
            
            result = {
                'influential_texts': influential_texts,
                'similarity_stats': self.similarity_engine.get_similarity_statistics(similarity_matrix)
            }
            
            return True, result, f"Found {k} most influential core texts"
            
        except Exception as e:
            error_msg = f"Error finding influential texts: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def prepare_dimensionality_reduction(self, method: str = "umap", 
                                       n_components: int = 2) -> Tuple[bool, str]:
        """Prepare dimensionality reduction for visualization"""
        try:
            if self.core_embeddings is None:
                return False, "Core corpus must be processed first"
            
            if method.lower() == "pca":
                success = self.similarity_engine.fit_pca(
                    self.core_embeddings, n_components=min(50, len(self.core_embeddings))
                )
                method_name = "PCA"
            elif method.lower() == "umap":
                success = self.similarity_engine.fit_umap(
                    self.core_embeddings, n_components=n_components
                )
                method_name = "UMAP"
            else:
                return False, f"Unknown dimensionality reduction method: {method}"
            
            if success:
                return True, f"{method_name} fitted successfully"
            else:
                return False, f"Failed to fit {method_name}"
                
        except Exception as e:
            error_msg = f"Error preparing dimensionality reduction: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def run_enhanced_topic_modeling(self, corpus_data: Dict, config: Dict) -> Tuple[bool, Dict, str]:
        return self.vector_analysis_engine.run_enhanced_topic_modeling(corpus_data, config)

    # Updated method for AnalysisCoordinator with paragraph numbers
    def calculate_topic_target_similarities(self, topic_results: Dict) -> Optional[Dict]:
        """Calculate how target corpus documents relate to core corpus topics"""
        try:
            if self.target_embeddings is None or self.core_embeddings is None:
                return None
            
            topic_model = topic_results['topic_model']
            
            # Get target corpus texts for topic assignment
            target_texts = []
            target_metadata = []
            
            for filename, doc_data in st.session_state.target_corpus['documents'].items():
                for para_idx, paragraph in enumerate(doc_data['paragraphs']):
                    target_texts.append(paragraph)
                    target_metadata.append({
                        'filename': filename,
                        'paragraph_index': para_idx,
                        'paragraph_number': para_idx + 1,  # 1-indexed for display
                        'title': doc_data['metadata']['title'],
                        'author': doc_data['metadata']['author']
                    })
            
            if not target_texts:
                return None
            
            # Transform target documents to get topic assignments and probabilities
            target_topics, target_probabilities = topic_model.transform(target_texts)
            
            # Calculate topic composition for each target document
            target_analysis = []
            
            for i, (text, topic_id, prob_dist) in enumerate(zip(target_texts, target_topics, target_probabilities)):
                if i >= len(target_metadata):
                    continue
                    
                doc_meta = target_metadata[i]
                
                # Get topic probabilities (if available)
                topic_composition = {}
                if prob_dist is not None and len(prob_dist) > 0:
                    # Get top topics for this document
                    topic_probs = [(topic_idx, prob) for topic_idx, prob in enumerate(prob_dist) if prob > 0.05]
                    topic_probs.sort(key=lambda x: x[1], reverse=True)
                    
                    for topic_idx, prob in topic_probs[:5]:
                        if topic_idx in topic_results['topic_labels']:
                            topic_composition[topic_idx] = {
                                'probability': float(prob),
                                'label': topic_results['topic_labels'][topic_idx],
                                'top_words': [word for word, _ in topic_results['topic_words'].get(topic_idx, [])[:5]]
                            }
                
                # Find most similar core document using embeddings
                if i < len(self.target_embeddings):
                    target_embedding = self.target_embeddings[i]
                    similarities = np.dot(self.core_embeddings, target_embedding)
                    most_similar_core_idx = np.argmax(similarities)
                    max_similarity = similarities[most_similar_core_idx]
                    
                    # Get core document metadata with paragraph number
                    if most_similar_core_idx < len(self.core_metadata['paragraph_metadata']):
                        core_meta = self.core_metadata['paragraph_metadata'][most_similar_core_idx]
                        most_similar_core = {
                            'title': core_meta['document_metadata']['title'],
                            'author': core_meta['document_metadata']['author'],
                            'filename': core_meta['filename'],
                            'paragraph_number': core_meta['paragraph_index'] + 1,  # 1-indexed
                            'similarity': float(max_similarity)
                        }
                    else:
                        most_similar_core = None
                else:
                    most_similar_core = None
                
                target_analysis.append({
                    'target_document': {
                        'title': doc_meta['title'],
                        'author': doc_meta['author'],
                        'filename': doc_meta['filename'],
                        'paragraph_number': doc_meta['paragraph_number']
                    },
                    'primary_topic': {
                        'id': int(topic_id) if topic_id != -1 else -1,
                        'label': topic_results['topic_labels'].get(topic_id, 'Outlier') if topic_id != -1 else 'Outlier'
                    },
                    'topic_composition': topic_composition,
                    'most_similar_core': most_similar_core,
                    'text_preview': text[:200] + "..." if len(text) > 200 else text
                })
            
            # Calculate overall statistics and prepare network data
            topic_distribution = {}
            network_edges = []
            
            for analysis in target_analysis:
                topic_id = analysis['primary_topic']['id']
                if topic_id not in topic_distribution:
                    topic_distribution[topic_id] = 0
                topic_distribution[topic_id] += 1
                
                # Create network edges for visualization
                if analysis['most_similar_core']:
                    network_edges.append({
                        'source': f"Target: {analysis['target_document']['title']} (P{analysis['target_document']['paragraph_number']})",
                        'target': f"Core: {analysis['most_similar_core']['title']} (P{analysis['most_similar_core']['paragraph_number']})",
                        'weight': analysis['most_similar_core']['similarity'],
                        'topic': analysis['primary_topic']['label']
                    })
            
            return {
                'target_analysis': target_analysis,
                'topic_distribution': topic_distribution,
                'network_edges': network_edges,
                'total_target_documents': len(target_analysis),
                'topics_found_in_target': len(set(analysis['primary_topic']['id'] for analysis in target_analysis))
            }
            
        except Exception as e:
            logger.error(f"Error calculating topic-target similarities: {str(e)}")
            return None

    def create_topic_network_visualization(self, target_similarities: Dict) -> Optional[Dict]:
        """Create network visualization of topic-document relationships"""
        try:
            import networkx as nx
            import plotly.graph_objects as go
            import plotly.express as px
            import numpy as np
            
            # Create network graph
            G = nx.Graph()
            
            # Add nodes and edges from similarity data
            for edge in target_similarities['network_edges']:
                if edge['weight'] > 0.3:  # Only show strong connections
                    G.add_edge(
                        edge['source'], 
                        edge['target'], 
                        weight=edge['weight'],
                        topic=edge['topic']
                    )
            
            if len(G.nodes()) == 0:
                return None
            
            # Calculate layout
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Prepare node data
            node_trace = []
            edge_trace = []
            
            # Create edges
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                edge_trace.append(
                    go.Scatter(
                        x=[x0, x1, None], 
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(width=edge[2]['weight']*3, color='lightgray'),
                        hoverinfo='none',
                        showlegend=False
                    )
                )
            
            # Create nodes
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                
                # Color by type (target vs core)
                if node.startswith('Target:'):
                    node_color.append('lightblue')
                else:
                    node_color.append('lightcoral')
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(size=10, color=node_color, line=dict(width=1)),
                text=[t.split(':')[1][:30] + "..." for t in node_text],  # Truncate labels
                textposition="middle center",
                hovertext=node_text,
                hoverinfo="text"
            )
            
            # Create figure
            fig = go.Figure(data=[node_trace] + edge_trace,
                           layout=go.Layout(
                               title='Topic-Document Network<br>Blue=Target, Red=Core, Line thickness=Similarity',
                               titlefont_size=16,
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=20,l=5,r=5,t=40),
                               annotations=[ dict(
                                   text="Hover over nodes for full document names",
                                   showarrow=False,
                                   xref="paper", yref="paper",
                                   x=0.005, y=-0.002,
                                   xanchor='left', yanchor='bottom',
                                   font=dict(color="gray", size=12)
                               )],
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                           ))
            
            return {
                'figure': fig,
                'network_stats': {
                    'nodes': len(G.nodes()),
                    'edges': len(G.edges()),
                    'density': nx.density(G),
                    'components': nx.number_connected_components(G)
                }
            }
            
        except ImportError:
            return {'error': 'NetworkX and Plotly required for network visualization'}
        except Exception as e:
            logger.error(f"Error creating network visualization: {str(e)}")
            return {'error': str(e)}

 
    def get_analysis_summary(self) -> Dict:
        """Get summary of current analysis state"""
        summary = {
            'model_name': self.embedding_engine.model_name,
            'embedding_dimension': self.embedding_engine.embedding_dim,
            'core_corpus_processed': self.core_embeddings is not None,
            'target_corpus_processed': self.target_embeddings is not None,
            'suggested_terms_available': self.suggested_terms is not None,
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
        
        return summary