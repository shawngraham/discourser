import streamlit as st
from ..base import BaseComponent

class SidebarComponent(BaseComponent):

    def render_sidebar(self):
        """Render the main navigation sidebar"""
        st.sidebar.title("Navigation")
        
        # Progress indicator
        steps = [
            ("Project Setup", 'project_setup'),
            ("Core Corpus", 'core_corpus'), 
            ("Target Corpus", 'target_corpus'),
            ("Preprocessing", 'analysis'),
            ("Vector Analysis", 'vectors'),
            ("Results", 'results')
        ]
        
        current_step = st.session_state.current_step
        
        st.sidebar.markdown("### Progress")
        for i, (step_name, step_key) in enumerate(steps):
            if step_key == current_step:
                st.sidebar.markdown(f"**→ {i+1}. {step_name}**")
            else:
                if st.sidebar.button(f"{i+1}. {step_name}", key=f"nav_{step_key}"):
                    st.session_state.current_step = step_key
                    st.rerun()
        
        # Project info
        if hasattr(st.session_state, 'project_name') and st.session_state.project_name:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### Current Project")
            st.sidebar.markdown(f"**{st.session_state.project_name}**")
            
            if st.sidebar.button("Save Project"):
                success, message = st.session_state.project_manager.save_project(st.session_state.project_name)
                if success:
                    st.sidebar.success("Project saved!")
                else:
                    st.sidebar.error(f"Error: {message}")