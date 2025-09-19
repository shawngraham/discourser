import streamlit as st
from typing import Optional, Dict, List, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime

# Import all helper functions
from .shared.helpers import (
    generate_key_findings,
    generate_document_report,
    generate_influence_report,
    find_mixed_clusters,
    analyze_quadrant_distribution,
    generate_pca_insights,
    generate_pca_recommendations,
    generate_combined_corpus_insights,
    generate_combined_corpus_recommendations,
    render_analysis_navigation_buttons
)

class BaseComponent:
    """Base class for all UI components with shared utilities"""
    
    def __init__(self):
        self.session_state = st.session_state
    
    def _check_prerequisites(self, required_items: List[str]) -> bool:
        """Check if required session state items exist"""
        return all(hasattr(self.session_state, item) for item in required_items)
    
    def _show_navigation_warning(self, message: str, target_step: str):
        """Show warning with navigation button"""
        st.warning(message)
        if st.button(f"Go to {target_step.replace('_', ' ').title()}"):
            self.session_state.current_step = target_step
            st.rerun()
    
    def _format_data(self, data):
        """Shared formatting logic for data display"""
        # Add any common formatting utilities here
        pass
    
    # Helper method wrappers - maintain original method signatures
    def _generate_key_findings(self, coordinator):
        """Generate key findings from all analyses"""
        return generate_key_findings(coordinator)
    
    def _generate_document_report(self, coordinator):
        """Generate comprehensive document-level analysis report"""
        return generate_document_report(coordinator)
    
    def _generate_influence_report(self, coordinator):
        """Generate influence patterns analysis"""
        return generate_influence_report(coordinator)
    
    def _find_mixed_clusters(self, df_plot):
        """Find regions with both core and target documents (indicating influence zones)"""
        return find_mixed_clusters(df_plot)
    
    def _analyze_quadrant_distribution(self, df_plot):
        """Analyze distribution of core vs target documents across quadrants"""
        return analyze_quadrant_distribution(df_plot)
    
    def _generate_pca_insights(self, pca_results: Dict, corpus_choice: str) -> List[str]:
        """Generate insights based on PCA results"""
        return generate_pca_insights(pca_results, corpus_choice)
    
    def _generate_pca_recommendations(self, pca_results: Dict, coordinator) -> List[str]:
        """Generate recommendations based on PCA results"""
        return generate_pca_recommendations(pca_results, coordinator)
    
    def _generate_combined_corpus_insights(self, pca_results: Dict, coordinator) -> List[str]:
        """Generate insights specific to combined corpus analysis"""
        return generate_combined_corpus_insights(pca_results, coordinator)
    
    def _generate_combined_corpus_recommendations(self, pca_results: Dict, coordinator) -> List[str]:
        """Generate recommendations for combined corpus analysis"""
        return generate_combined_corpus_recommendations(pca_results, coordinator)
    
    def _render_analysis_navigation_buttons(self):
        """Render navigation buttons to help users get to analysis steps"""
        return render_analysis_navigation_buttons()