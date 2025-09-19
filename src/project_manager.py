import json
import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ProjectManager:
    """Manages project saving, loading, and persistence with complete state preservation"""
    
    def __init__(self):
        self.projects_dir = Path("projects")
        self.projects_dir.mkdir(exist_ok=True)
    
    def create_project(self, project_name: str) -> tuple[bool, str]:
        """Create a new project directory"""
        try:
            project_path = self.projects_dir / project_name
            project_path.mkdir(exist_ok=True)
            
            # Create project metadata
            metadata = {
                "project_name": project_name,
                "created_date": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(project_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            return True, "Project created successfully"
            
        except Exception as e:
            return False, f"Error creating project: {str(e)}"
    
    def save_project(self, project_name: str) -> tuple[bool, str]:
        """Save complete project state including all analysis results"""
        try:
            project_path = self.projects_dir / project_name
            project_path.mkdir(exist_ok=True)
            
            save_data = {}
            saved_components = []
            
            # Save corpus data
            if 'core_corpus' in st.session_state:
                save_data['core_corpus'] = st.session_state.core_corpus
                saved_components.append("Core corpus")
            
            if 'target_corpus' in st.session_state:
                save_data['target_corpus'] = st.session_state.target_corpus
                saved_components.append("Target corpus")
            
            # Save UI state
            ui_state_keys = [
                'core_files_uploaded', 'target_files_uploaded', 'current_step',
                'core_embeddings_ready', 'target_embeddings_ready'
            ]
            for key in ui_state_keys:
                if key in st.session_state:
                    save_data[key] = st.session_state[key]
            
            # Save complete analysis coordinator state
            if 'analysis_coordinator' in st.session_state:
                coordinator = st.session_state.analysis_coordinator
                
                # Save embedding engine state
                embedding_state = {
                    'model_name': coordinator.embedding_engine.model_name,
                    'embedding_dim': coordinator.embedding_engine.embedding_dim,
                    'core_embeddings': coordinator.core_embeddings,
                    'target_embeddings': coordinator.target_embeddings,
                    'core_metadata': coordinator.core_metadata,
                    'target_metadata': coordinator.target_metadata,
                    'suggested_terms': coordinator.suggested_terms
                }
                save_data['embedding_state'] = embedding_state
                saved_components.append("Embeddings")
                
                # Save custom vectors
                if hasattr(coordinator, 'custom_vector_manager') and coordinator.custom_vector_manager:
                    custom_vectors_data = self._serialize_custom_vectors(coordinator.custom_vector_manager)
                    save_data['custom_vectors'] = custom_vectors_data
                    if custom_vectors_data.get('vectors'):
                        saved_components.append(f"Custom vectors ({len(custom_vectors_data['vectors'])})")
                
                # Save similarity engine state
                if hasattr(coordinator, 'similarity_engine') and coordinator.similarity_engine:
                    similarity_state = self._serialize_similarity_engine(coordinator.similarity_engine)
                    save_data['similarity_state'] = similarity_state
                    if any(similarity_state.values()):
                        saved_components.append("Similarity models")
            
            # Save topic modeling results
            topic_keys = [
                'topic_results', 'topic_target_similarities', 'topic_config',
                'enhanced_topic_results'
            ]
            for key in topic_keys:
                if key in st.session_state:
                    if key == 'topic_results' and st.session_state[key]:
                        # Special handling for topic results with BERTopic model
                        topic_data = self._serialize_topic_results(st.session_state[key])
                        save_data[key] = topic_data
                        saved_components.append("Topic modeling")
                    else:
                        save_data[key] = st.session_state[key]
            
            # Save analysis results
            analysis_keys = [
                'similarity_matrix', 'influential_texts', 'projection_results',
                'vector_analysis_results', '2d_space_results', '3d_space_results',
                'dimensionality_reduction_results', 'custom_vector_statistics'
            ]
            for key in analysis_keys:
                if key in st.session_state and st.session_state[key] is not None:
                    save_data[key] = st.session_state[key]
                    saved_components.append(key.replace('_', ' ').title())
            
            # Save visualization states
            viz_keys = [
                'current_visualization_mode', 'selected_vectors_for_viz',
                'plot_configurations', 'export_settings'
            ]
            for key in viz_keys:
                if key in st.session_state:
                    save_data[key] = st.session_state[key]
            
            # Use pickle for complex data structures
            with open(project_path / "project_data.pkl", "wb") as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save a human-readable summary
            self._save_project_summary(project_path, save_data, saved_components)
            
            # Update metadata
            metadata_path = project_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                metadata["last_modified"] = datetime.now().isoformat()
                metadata["saved_components"] = saved_components
                metadata["data_size_mb"] = round(
                    (project_path / "project_data.pkl").stat().st_size / (1024 * 1024), 2
                )
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
            
            components_str = ", ".join(saved_components) if saved_components else "No analysis data"
            return True, f"Project saved successfully. Components: {components_str}"
            
        except Exception as e:
            logger.error(f"Error saving project: {str(e)}")
            return False, f"Error saving project: {str(e)}"
    
    def _serialize_custom_vectors(self, vector_manager) -> Dict:
        """Serialize custom vector manager state"""
        try:
            vectors_data = {}
            for name, vector_info in vector_manager.custom_vectors.items():
                # Convert numpy arrays to lists for JSON serialization
                serialized_info = vector_info.copy()
                if 'vector' in serialized_info:
                    serialized_info['vector'] = vector_info['vector'].tolist()
                vectors_data[name] = serialized_info
            
            return {
                'vectors': vectors_data,
                'validation_examples': vector_manager.validation_examples,
                'max_vectors': vector_manager.max_vectors
            }
        except Exception as e:
            logger.error(f"Error serializing custom vectors: {str(e)}")
            return {}
    
    def _serialize_similarity_engine(self, similarity_engine) -> Dict:
        """Serialize similarity engine state"""
        try:
            state = {
                'has_pca_model': similarity_engine.pca_model is not None,
                'has_umap_model': similarity_engine.umap_model is not None,
                'has_nn_model': similarity_engine.nn_model is not None
            }
            
            # Save PCA state if it exists
            if similarity_engine.pca_model is not None:
                state['pca_components'] = similarity_engine.pca_model.components_.tolist()
                state['pca_explained_variance_ratio'] = similarity_engine.pca_model.explained_variance_ratio_.tolist()
                state['pca_n_components'] = similarity_engine.pca_model.n_components_
            
            return state
        except Exception as e:
            logger.error(f"Error serializing similarity engine: {str(e)}")
            return {}
    
    def _serialize_topic_results(self, topic_results: Dict) -> Dict:
        """Serialize topic modeling results, handling BERTopic model specially"""
        try:
            serialized = topic_results.copy()
            
            # Remove the actual BERTopic model object (too complex to serialize)
            if 'topic_model' in serialized:
                del serialized['topic_model']
                
            # Convert numpy arrays to lists
            if 'topics' in serialized:
                serialized['topics'] = serialized['topics'].tolist() if hasattr(serialized['topics'], 'tolist') else serialized['topics']
            
            if 'probabilities' in serialized:
                probs = serialized['probabilities']
                if probs is not None and hasattr(probs, 'tolist'):
                    serialized['probabilities'] = probs.tolist()
            
            # Keep the topics DataFrame as dict
            if 'topics_dataframe' in serialized:
                df = serialized['topics_dataframe']
                if hasattr(df, 'to_dict'):
                    serialized['topics_dataframe'] = df.to_dict()
            
            return serialized
        except Exception as e:
            logger.error(f"Error serializing topic results: {str(e)}")
            return {}
    
    def _save_project_summary(self, project_path: Path, save_data: Dict, saved_components: List[str]):
        """Save a human-readable project summary"""
        try:
            summary = {
                "saved_at": datetime.now().isoformat(),
                "components_saved": saved_components,
                "data_summary": {}
            }
            
            # Add data summaries
            if 'core_corpus' in save_data:
                core = save_data['core_corpus']
                summary["data_summary"]["core_corpus"] = {
                    "documents": len(core.get('documents', {})),
                    "total_paragraphs": core.get('total_paragraphs', 0)
                }
            
            if 'target_corpus' in save_data:
                target = save_data['target_corpus']
                summary["data_summary"]["target_corpus"] = {
                    "documents": len(target.get('documents', {})),
                    "total_paragraphs": target.get('total_paragraphs', 0)
                }
            
            if 'embedding_state' in save_data:
                embed_state = save_data['embedding_state']
                summary["data_summary"]["embeddings"] = {
                    "model_name": embed_state.get('model_name'),
                    "embedding_dimension": embed_state.get('embedding_dim'),
                    "core_embeddings_shape": list(embed_state['core_embeddings'].shape) if embed_state.get('core_embeddings') is not None else None,
                    "target_embeddings_shape": list(embed_state['target_embeddings'].shape) if embed_state.get('target_embeddings') is not None else None
                }
            
            if 'custom_vectors' in save_data:
                vectors = save_data['custom_vectors'].get('vectors', {})
                summary["data_summary"]["custom_vectors"] = {
                    "count": len(vectors),
                    "vector_names": list(vectors.keys())
                }
            
            with open(project_path / "project_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving project summary: {str(e)}")
    
    def load_project(self, project_name: str) -> tuple[bool, str]:
        """Load complete project state"""
        try:
            project_path = self.projects_dir / project_name
            
            if not project_path.exists():
                return False, "Project not found"
            
            # Load project data
            data_file = project_path / "project_data.pkl"
            if not data_file.exists():
                return False, "No project data found"
            
            with open(data_file, "rb") as f:
                save_data = pickle.load(f)
            
            # Clear existing session state except for managers
            keys_to_preserve = ['project_manager', 'data_handler']
            for key in list(st.session_state.keys()):
                if key not in keys_to_preserve:
                    del st.session_state[key]
            
            # Restore basic session state
            loaded_components = []
            
            # Restore corpora
            for corpus_key in ['core_corpus', 'target_corpus']:
                if corpus_key in save_data:
                    st.session_state[corpus_key] = save_data[corpus_key]
                    loaded_components.append(corpus_key.replace('_', ' ').title())
            
            # Restore UI state
            ui_keys = [
                'core_files_uploaded', 'target_files_uploaded', 'current_step',
                'core_embeddings_ready', 'target_embeddings_ready'
            ]
            for key in ui_keys:
                if key in save_data:
                    st.session_state[key] = save_data[key]
            
            # Restore analysis coordinator
            if 'embedding_state' in save_data:
                self._restore_analysis_coordinator(save_data)
                loaded_components.append("Analysis coordinator")
            
            # Restore custom vectors
            if 'custom_vectors' in save_data:
                self._restore_custom_vectors(save_data['custom_vectors'])
                loaded_components.append(f"Custom vectors")
            
            # Restore all analysis results
            analysis_keys = [
                'topic_results', 'topic_target_similarities', 'topic_config',
                'similarity_matrix', 'influential_texts', 'projection_results',
                'vector_analysis_results', '2d_space_results', '3d_space_results'
            ]
            for key in analysis_keys:
                if key in save_data:
                    st.session_state[key] = save_data[key]
                    if save_data[key] is not None:
                        loaded_components.append(key.replace('_', ' ').title())
            
            # Restore visualization state
            viz_keys = [
                'current_visualization_mode', 'selected_vectors_for_viz',
                'plot_configurations', 'export_settings'
            ]
            for key in viz_keys:
                if key in save_data:
                    st.session_state[key] = save_data[key]
            
            components_str = ", ".join(loaded_components) if loaded_components else "Basic project data"
            return True, f"Project loaded successfully. Components: {components_str}"
            
        except Exception as e:
            logger.error(f"Error loading project: {str(e)}")
            return False, f"Error loading project: {str(e)}"
    
    def _restore_analysis_coordinator(self, save_data: Dict):
        """Restore analysis coordinator with all its state"""
        try:
            from src.analysis_coordinator import AnalysisCoordinator
            
            coordinator = AnalysisCoordinator()
            embed_state = save_data['embedding_state']
            
            # Restore embedding engine
            coordinator.embedding_engine.model_name = embed_state.get('model_name', 'all-MiniLM-L6-v2')
            coordinator.embedding_engine.embedding_dim = embed_state.get('embedding_dim')
            
            # Restore embeddings and metadata
            coordinator.core_embeddings = embed_state.get('core_embeddings')
            coordinator.target_embeddings = embed_state.get('target_embeddings')
            coordinator.core_metadata = embed_state.get('core_metadata')
            coordinator.target_metadata = embed_state.get('target_metadata')
            coordinator.suggested_terms = embed_state.get('suggested_terms')
            
            # Setup nearest neighbors if core embeddings exist
            if coordinator.core_embeddings is not None:
                coordinator.similarity_engine.setup_nearest_neighbors(coordinator.core_embeddings)
            
            # Restore similarity engine state
            if 'similarity_state' in save_data:
                self._restore_similarity_engine(coordinator.similarity_engine, save_data['similarity_state'])
            
            st.session_state.analysis_coordinator = coordinator
            
        except Exception as e:
            logger.error(f"Error restoring analysis coordinator: {str(e)}")
    
    def _restore_custom_vectors(self, vectors_data: Dict):
        """Restore custom vectors to the analysis coordinator"""
        try:
            if 'analysis_coordinator' not in st.session_state:
                return
            
            coordinator = st.session_state.analysis_coordinator
            vector_manager = coordinator.custom_vector_manager
            
            # Clear existing vectors
            vector_manager.custom_vectors = {}
            
            # Restore vectors
            for name, vector_info in vectors_data.get('vectors', {}).items():
                restored_info = vector_info.copy()
                # Convert list back to numpy array
                if 'vector' in restored_info:
                    restored_info['vector'] = np.array(vector_info['vector'])
                vector_manager.custom_vectors[name] = restored_info
            
            # Restore validation examples
            if 'validation_examples' in vectors_data:
                vector_manager.validation_examples = vectors_data['validation_examples']
            
        except Exception as e:
            logger.error(f"Error restoring custom vectors: {str(e)}")
    
    def _restore_similarity_engine(self, similarity_engine, similarity_state: Dict):
        """Restore similarity engine state"""
        try:
            # Note: We don't restore the actual fitted models as they depend on the current embeddings
            # Instead, we just note what was previously fitted
            pass
        except Exception as e:
            logger.error(f"Error restoring similarity engine: {str(e)}")
    
    def list_projects(self) -> List[str]:
        """List all available projects"""
        try:
            projects = []
            for item in self.projects_dir.iterdir():
                if item.is_dir() and (item / "metadata.json").exists():
                    projects.append(item.name)
            return sorted(projects)
            
        except Exception:
            return []
    
    def get_project_details(self, project_name: str) -> Dict:
        """Get detailed information about a specific project"""
        try:
            project_path = self.projects_dir / project_name
            
            if not project_path.exists():
                return {"error": "Project not found"}
            
            details = {"project_name": project_name}
            
            # Load metadata
            metadata_path = project_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                details.update(metadata)
            
            # Load summary if it exists
            summary_path = project_path / "project_summary.json"
            if summary_path.exists():
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                details.update(summary)
            
            return details
            
        except Exception as e:
            return {"error": str(e)}
    
    def delete_project(self, project_name: str) -> tuple[bool, str]:
        """Delete a project"""
        try:
            project_path = self.projects_dir / project_name
            if project_path.exists():
                import shutil
                shutil.rmtree(project_path)
                return True, "Project deleted successfully"
            else:
                return False, "Project not found"
            
        except Exception as e:
            return False, f"Error deleting project: {str(e)}"