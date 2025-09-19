import streamlit as st
from ..base import BaseComponent

class ProjectSetupComponent(BaseComponent):
    def render_project_setup(self):
        """Render project setup page"""
        st.header("Project Setup")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            Welcome to Discourser! This application helps you analyze 
            the influence of a core body of work across other corpora.
            
            **Getting Started:**
            1. Create a new project or load an existing one
            2. Upload your core corpus (the influential source documents)
            3. Upload your target corpus (documents to analyze)
            4. Define analysis vectors and run analysis
            5. Explore results and generate reports
            """)
        
        with col2:
            st.markdown("### Project Management")
            
            # New project
            project_name = st.text_input("New Project Name")
            if st.button("Create New Project"):
                if project_name:
                    success, message = st.session_state.project_manager.create_project(project_name)
                    if success:
                        st.session_state.project_name = project_name
                        st.success(f"Created project: {project_name}")
                        st.session_state.current_step = 'core_corpus'
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please enter a project name")
            
            st.markdown("---")
            
            # Load existing project
            existing_projects = st.session_state.project_manager.list_projects()
            if existing_projects:
                selected_project = st.selectbox("Load Existing Project", [""] + existing_projects)
                if st.button("Load Project"):
                    if selected_project:
                        success, message = st.session_state.project_manager.load_project(selected_project)
                        if success:
                            st.session_state.project_name = selected_project
                            
                            # Show what was loaded and don't auto-navigate
                            st.success(f"✅ {message}")
                            
                            # Show detailed status
                            if 'core_corpus' in st.session_state:
                                st.info(f"📁 Core corpus loaded: {len(st.session_state.core_corpus['documents'])} documents, {st.session_state.core_corpus['total_paragraphs']} paragraphs")
                            
                            if 'target_corpus' in st.session_state:
                                st.info(f"📁 Target corpus loaded: {len(st.session_state.target_corpus['documents'])} documents, {st.session_state.target_corpus['total_paragraphs']} paragraphs")
                            
                            if 'analysis_coordinator' in st.session_state:
                                coordinator = st.session_state.analysis_coordinator
                                if coordinator.core_embeddings is not None:
                                    st.info(f"🧠 Core embeddings ready: {coordinator.embedding_engine.model_name}")
                                if coordinator.target_embeddings is not None:
                                    st.info(f"🧠 Target embeddings ready")
                            
                            # Show navigation options
                            st.markdown("**Next Steps:**")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                if st.button("📁 View Core Corpus"):
                                    st.session_state.current_step = 'core_corpus'
                                    st.rerun()
                            
                            with col2:
                                if st.button("📁 View Target Corpus"):
                                    st.session_state.current_step = 'target_corpus'
                                    st.rerun()
                            
                            with col3:
                                if st.button("🔬 Go to Preprocessing/Analysis"):
                                    st.session_state.current_step = 'analysis'
                                    st.rerun()
                            
                            with col4:
                                if st.button("📊 View Results"):
                                    st.session_state.current_step = 'results'
                                    st.rerun()
                            
                        else:
                            st.error(message)
                    else:
                        st.error("Please select a project")
