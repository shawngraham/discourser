import streamlit as st
from .components.navigation.sidebar import SidebarComponent
from .components.setup.project_setup import ProjectSetupComponent
from .components.setup.corpus_upload import CorpusUploadComponent
from .components.analysis.preprocessing import PreprocessingComponent
from .components.analysis.vector_analysis import VectorAnalysisComponent
from .components.results.comprehensive_results import ComprehensiveResultsComponent

class UIComponents:
    """Main UI orchestrator that coordinates all component classes"""
    
    def __init__(self):
        # Initialize all component classes
        self.sidebar = SidebarComponent()
        self.project_setup = ProjectSetupComponent()
        self.corpus_upload = CorpusUploadComponent()
        self.preprocessing = PreprocessingComponent()
        self.vector_analysis = VectorAnalysisComponent()
        self.results = ComprehensiveResultsComponent()
    
    # Navigation
    def render_sidebar(self):
        """Render the main navigation sidebar"""
        return self.sidebar.render_sidebar()
    
    # Setup methods
    def render_project_setup(self):
        """Render project setup page"""
        return self.project_setup.render_project_setup()
    
    def render_core_corpus_upload(self):
        """Render core corpus upload page"""
        return self.corpus_upload.render_core_corpus_upload()
    
    def render_target_corpus_upload(self):
        """Render target corpus upload page"""
        return self.corpus_upload.render_target_corpus_upload()
    
    # Analysis methods
    def render_analysis_page(self):
        """Render the analysis page with embedding generation"""
        return self.preprocessing.render_analysis_page()
    
    def render_vector_analysis_page(self):
        """Render the vector analysis and custom dimensions page"""
        return self.vector_analysis.render_vector_analysis_page()
    
    # Results methods
    def render_comprehensive_results(self):
        """Render comprehensive results page with all analyses and export options"""
        return self.results.render_comprehensive_results()