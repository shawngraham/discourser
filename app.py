import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data_handler import DataHandler
from src.ui_components import UIComponents
from src.project_manager import ProjectManager

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Discourser",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'project_manager' not in st.session_state:
        st.session_state.project_manager = ProjectManager()
    
    if 'data_handler' not in st.session_state:
        st.session_state.data_handler = DataHandler()
    
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'project_setup'
    
    # Main UI
    st.title("📚 Discourser")
    st.markdown("---")
    
    # Sidebar navigation
    ui = UIComponents()
    ui.render_sidebar()
    
    # Main content area
    if st.session_state.current_step == 'project_setup':
        ui.render_project_setup()
    elif st.session_state.current_step == 'core_corpus':
        ui.render_core_corpus_upload()
    elif st.session_state.current_step == 'target_corpus':
        ui.render_target_corpus_upload()
    elif st.session_state.current_step == 'analysis':
        ui.render_analysis_page()
    elif st.session_state.current_step == 'vectors':
        ui.render_vector_analysis_page()
    elif st.session_state.current_step == 'results':
        ui.render_comprehensive_results()

if __name__ == "__main__":
    main()