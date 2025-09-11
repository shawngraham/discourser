import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st
from datetime import datetime

class ProjectManager:
    """Manages project saving, loading, and persistence"""
    
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
        """Save current project state"""
        try:
            project_path = self.projects_dir / project_name
            project_path.mkdir(exist_ok=True)
            
            # Save session state data
            save_data = {}
            
            # Save corpus data if it exists
            if 'core_corpus' in st.session_state:
                save_data['core_corpus'] = st.session_state.core_corpus
            
            if 'target_corpus' in st.session_state:
                save_data['target_corpus'] = st.session_state.target_corpus
            
            # Save file upload status for UI display
            if 'core_files_uploaded' in st.session_state:
                save_data['core_files_uploaded'] = st.session_state.core_files_uploaded
            
            if 'target_files_uploaded' in st.session_state:
                save_data['target_files_uploaded'] = st.session_state.target_files_uploaded
            
            # Save other project data
            save_data['current_step'] = st.session_state.get('current_step', 'project_setup')
            save_data['core_embeddings_ready'] = st.session_state.get('core_embeddings_ready', False)
            save_data['target_embeddings_ready'] = st.session_state.get('target_embeddings_ready', False)
            
            # Save analysis coordinator state if it exists
            if 'analysis_coordinator' in st.session_state:
                coordinator = st.session_state.analysis_coordinator
                analysis_state = {
                    'model_name': coordinator.embedding_engine.model_name,
                    'core_embeddings': coordinator.core_embeddings,
                    'target_embeddings': coordinator.target_embeddings,
                    'core_metadata': coordinator.core_metadata,
                    'target_metadata': coordinator.target_metadata
                }
                save_data['analysis_state'] = analysis_state
            
            # Use pickle for complex data structures
            with open(project_path / "project_data.pkl", "wb") as f:
                pickle.dump(save_data, f)
            
            # Update metadata
            metadata_path = project_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                metadata["last_modified"] = datetime.now().isoformat()
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
            
            return True, "Project saved successfully"
            
        except Exception as e:
            return False, f"Error saving project: {str(e)}"
    
    def load_project(self, project_name: str) -> tuple[bool, str]:
        """Load existing project"""
        try:
            project_path = self.projects_dir / project_name
            
            if not project_path.exists():
                return False, "Project not found"
            
            # Load project data
            data_file = project_path / "project_data.pkl"
            if data_file.exists():
                with open(data_file, "rb") as f:
                    save_data = pickle.load(f)
                
                # Clear existing session state except for managers
                keys_to_preserve = ['project_manager', 'data_handler']
                for key in list(st.session_state.keys()):
                    if key not in keys_to_preserve:
                        del st.session_state[key]
                
                # Restore session state
                has_core = False
                has_target = False
                
                for key, value in save_data.items():
                    if key == 'analysis_state':
                        # Restore analysis coordinator state
                        try:
                            from src.analysis_coordinator import AnalysisCoordinator
                            coordinator = AnalysisCoordinator()
                            coordinator.embedding_engine.model_name = value.get('model_name', 'all-MiniLM-L6-v2')
                            coordinator.core_embeddings = value.get('core_embeddings')
                            coordinator.target_embeddings = value.get('target_embeddings')
                            coordinator.core_metadata = value.get('core_metadata')
                            coordinator.target_metadata = value.get('target_metadata')
                            st.session_state.analysis_coordinator = coordinator
                        except Exception as e:
                            print(f"Warning: Could not restore analysis coordinator: {e}")
                    else:
                        st.session_state[key] = value
                        if key == 'core_corpus':
                            has_core = True
                        elif key == 'target_corpus':
                            has_target = True
                
                # Build status message
                status_parts = ["Project loaded"]
                if has_core:
                    status_parts.append(f"Core corpus: {len(st.session_state.core_corpus.get('documents', {}))} documents")
                if has_target:
                    status_parts.append(f"Target corpus: {len(st.session_state.target_corpus.get('documents', {}))} documents")
                
                if not has_core and not has_target:
                    status_parts.append("No corpus data found - you'll need to upload corpora")
                
                return True, " | ".join(status_parts)
            else:
                return True, "Project loaded (no saved data found)"
            
        except Exception as e:
            return False, f"Error loading project: {str(e)}"
    
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