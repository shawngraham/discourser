import streamlit as st
from typing import Optional
from datetime import datetime
import time
import re
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import json
import io

class UIComponents:
    """Handles all UI components and rendering"""
    
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
            2. Upload your core corpus (influential academic texts)
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
    
    def render_core_corpus_upload(self):
        """Render core corpus upload page"""
        st.header("Core Corpus Upload")
        
        # Show current status if corpus is loaded
        if 'core_corpus' in st.session_state:
            st.success("✅ Core corpus already loaded!")
            corpus_data = st.session_state.core_corpus
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**Current Core Corpus:**")
                st.write(f"- Documents: {len(corpus_data['documents'])}")
                st.write(f"- Total paragraphs: {corpus_data['total_paragraphs']}")
                
                # Show file info if available
                if 'core_files_uploaded' in st.session_state:
                    file_info = st.session_state.core_files_uploaded
                    st.write(f"- CSV file: {file_info['csv_name']}")
                    st.write(f"- Text files: {len(file_info['text_files'])} files")
                
                # Show sample documents
                with st.expander("Preview Documents"):
                    for filename, doc_data in list(corpus_data['documents'].items())[:3]:
                        st.markdown(f"**{filename}**")
                        st.write(f"Title: {doc_data['metadata']['title']}")
                        st.write(f"Paragraphs: {doc_data['paragraph_count']}")
                        st.write(f"First paragraph: {doc_data['paragraphs'][0][:200]}...")
                        st.markdown("---")
            
            with col2:
                if st.button("Replace Core Corpus"):
                    # Clear existing data to allow re-upload
                    if 'core_corpus' in st.session_state:
                        del st.session_state.core_corpus
                    if 'core_files_uploaded' in st.session_state:
                        del st.session_state.core_files_uploaded
                    st.rerun()
                
                if st.button("Continue to Target Corpus"):
                    st.session_state.current_step = 'target_corpus'
                    st.rerun()
            
            return
        
        st.markdown("""
        Upload your core corpus - the collection of influential academic texts that will serve 
        as the foundation for analysis.
        
        **Requirements:**
        - CSV metadata file with columns: filename, title, author, date, source
        - Text files (.txt) with filenames matching the CSV
        - Paragraphs in text files separated by double blank lines
        """)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # CSV upload
            st.subheader("1. Upload Metadata CSV")
            csv_file = st.file_uploader(
                "Choose CSV file", 
                type=['csv'],
                key="core_csv"
            )
            
            # Text files upload
            st.subheader("2. Upload Text Files")
            text_files = st.file_uploader(
                "Choose text files",
                type=['txt'],
                accept_multiple_files=True,
                key="core_texts"
            )
            
            if csv_file and text_files:
                st.subheader("3. Validation Results")
                
                # Process files
                text_files_dict = {f.name: f.read() for f in text_files}
                
                with st.spinner("Validating corpus..."):
                    success, processed_data, errors = st.session_state.data_handler.process_corpus(
                        csv_file, text_files_dict
                    )
                
                if success:
                    st.success("✅ Core corpus validation successful!")
                    
                    # Show summary
                    st.markdown("**Corpus Summary:**")
                    st.write(f"- Documents: {len(processed_data['documents'])}")
                    st.write(f"- Total paragraphs: {processed_data['total_paragraphs']}")
                    
                    # Store in session state
                    st.session_state.core_corpus = processed_data
                    
                    # Store file upload info for later display
                    st.session_state.core_files_uploaded = {
                        'csv_name': csv_file.name,
                        'text_files': [f.name for f in text_files],
                        'upload_time': datetime.now().isoformat()
                    }
                    
                    # Show sample
                    with st.expander("Preview Documents"):
                        for filename, doc_data in list(processed_data['documents'].items())[:3]:
                            st.markdown(f"**{filename}**")
                            st.write(f"Title: {doc_data['metadata']['title']}")
                            st.write(f"Paragraphs: {doc_data['paragraph_count']}")
                            st.write(f"First paragraph: {doc_data['paragraphs'][0][:200]}...")
                            st.markdown("---")
                    
                    if st.button("Continue to Target Corpus"):
                        st.session_state.current_step = 'target_corpus'
                        st.rerun()
                        
                else:
                    st.error("❌ Validation failed!")
                    for error in errors:
                        st.error(f"• {error}")
        
        with col2:
            st.markdown("### CSV Format Example")
            st.code("""filename,title,author,date,source
paper1.txt,The Origin of Species,Darwin,1859-11-24,Academic
theory2.txt,Relativity Theory,Einstein,1915-11-25,Academic
manifesto.txt,Communist Manifesto,Marx,1848-02-21,Political
""", language="csv")
            
            st.markdown("### Text File Format")
            st.code("""First paragraph of the document.
This continues the first paragraph.


Second paragraph starts here after 
double blank line.


Third paragraph and so on.
""", language="text")
    
    def render_target_corpus_upload(self):
        """Render target corpus upload page"""
        st.header("Target Corpus Upload")
        
        if 'core_corpus' not in st.session_state:
            st.warning("Please upload core corpus first.")
            if st.button("Go to Core Corpus"):
                st.session_state.current_step = 'core_corpus'
                st.rerun()
            return
        
        # Show current status if corpus is loaded
        if 'target_corpus' in st.session_state:
            st.success("✅ Target corpus already loaded!")
            corpus_data = st.session_state.target_corpus
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**Current Target Corpus:**")
                st.write(f"- Documents: {len(corpus_data['documents'])}")
                st.write(f"- Total paragraphs: {corpus_data['total_paragraphs']}")
                
                # Show file info if available
                if 'target_files_uploaded' in st.session_state:
                    file_info = st.session_state.target_files_uploaded
                    st.write(f"- CSV file: {file_info['csv_name']}")
                    st.write(f"- Text files: {len(file_info['text_files'])} files")
                
                # Show sample documents
                with st.expander("Preview Documents"):
                    for filename, doc_data in list(corpus_data['documents'].items())[:3]:
                        st.markdown(f"**{filename}**")
                        st.write(f"Title: {doc_data['metadata']['title']}")
                        st.write(f"Paragraphs: {doc_data['paragraph_count']}")
                        st.write(f"First paragraph: {doc_data['paragraphs'][0][:200]}...")
                        st.markdown("---")
            
            with col2:
                if st.button("Replace Target Corpus"):
                    # Clear existing data to allow re-upload
                    if 'target_corpus' in st.session_state:
                        del st.session_state.target_corpus
                    if 'target_files_uploaded' in st.session_state:
                        del st.session_state.target_files_uploaded
                    st.rerun()
                
                if st.button("Continue to Preprocessing/Analysis"):
                    st.session_state.current_step = 'analysis'
                    st.rerun()
            
            return
        
        st.markdown("""
        Upload your target corpus - the documents you want to analyze for influence 
        from the core corpus.
        
        **Same format as core corpus:** CSV metadata + text files with double blank line paragraph separation.
        """)
        
        # CSV upload
        st.subheader("1. Upload Metadata CSV")
        csv_file = st.file_uploader(
            "Choose CSV file", 
            type=['csv'],
            key="target_csv"
        )
        
        # Text files upload
        st.subheader("2. Upload Text Files") 
        text_files = st.file_uploader(
            "Choose text files",
            type=['txt'],
            accept_multiple_files=True,
            key="target_texts"
        )
        
        if csv_file and text_files:
            st.subheader("3. Validation Results")
            
            # Process files
            text_files_dict = {f.name: f.read() for f in text_files}
            
            with st.spinner("Validating target corpus..."):
                success, processed_data, errors = st.session_state.data_handler.process_corpus(
                    csv_file, text_files_dict
                )
            
            if success:
                st.success("✅ Target corpus validation successful!")
                
                # Show summary
                st.markdown("**Target Corpus Summary:**")
                st.write(f"- Documents: {len(processed_data['documents'])}")
                st.write(f"- Total paragraphs: {processed_data['total_paragraphs']}")
                
                # Store in session state
                st.session_state.target_corpus = processed_data
                
                # Store file upload info for later display
                st.session_state.target_files_uploaded = {
                    'csv_name': csv_file.name,
                    'text_files': [f.name for f in text_files],
                    'upload_time': datetime.now().isoformat()
                }
                
                if st.button("Continue to Preprocessing/Analysis"):
                    st.session_state.current_step = 'analysis'
                    st.rerun()
                    
            else:
                st.error("❌ Validation failed!")
                for error in errors:
                    st.error(f"• {error}")
    
    def render_analysis_page(self):
        """Render the analysis page with embedding generation"""
        st.header("Preprocessing/Analysis")
        
        # Check if corpora are loaded
        if 'core_corpus' not in st.session_state or 'target_corpus' not in st.session_state:
            st.warning("Please upload both core and target corpora first.")
            return
        
        # Initialize analysis coordinator
        if 'analysis_coordinator' not in st.session_state:
            from src.analysis_coordinator import AnalysisCoordinator
            st.session_state.analysis_coordinator = AnalysisCoordinator()
        
        coordinator = st.session_state.analysis_coordinator
        
        # Model selection
        st.subheader("1. Model Configuration")
        
        model_options = {
            "all-MiniLM-L6-v2": "Fast, good for most tasks (384 dim)",
            "all-mpnet-base-v2": "Higher quality, slower (768 dim)", 
            "all-distilroberta-v1": "Balanced speed/quality (768 dim)",
            "paraphrase-multilingual-MiniLM-L12-v2": "Multilingual support (384 dim)"
        }
        
        selected_model = st.selectbox(
            "Choose embedding model:",
            options=list(model_options.keys()),
            format_func=lambda x: f"{x} - {model_options[x]}"
        )
        
        # Initialize model
        if st.button("Initialize Model"):
            with st.spinner("Loading model..."):
                success = coordinator.initialize_analysis(selected_model)
                if success:
                    st.success(f"✅ Model {selected_model} loaded successfully!")
                else:
                    st.error("❌ Failed to load model")
                    return
        
        # Check if model is initialized
        if not hasattr(coordinator.embedding_engine, 'model') or coordinator.embedding_engine.model is None:
            st.info("Please initialize the model first.")
            return
        
        st.subheader("2. Generate Embeddings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Core Corpus**")
            if st.button("Process Core Corpus"):
                progress_placeholder = st.empty()
                
                def progress_callback(message):
                    progress_placeholder.info(message)
                
                with st.spinner("Processing core corpus..."):
                    success, message = coordinator.process_core_corpus(
                        st.session_state.core_corpus,
                        progress_callback
                    )
                
                progress_placeholder.empty()
                
                if success:
                    st.success(message)
                    st.session_state.core_embeddings_ready = True
                else:
                    st.error(message)
        
        with col2:
            st.markdown("**Target Corpus**")
            if st.button("Process Target Corpus"):
                progress_placeholder = st.empty()
                
                def progress_callback(message):
                    progress_placeholder.info(message)
                
                with st.spinner("Processing target corpus..."):
                    success, message = coordinator.process_target_corpus(
                        st.session_state.target_corpus,
                        progress_callback
                    )
                
                progress_placeholder.empty()
                
                if success:
                    st.success(message)
                    st.session_state.target_embeddings_ready = True
                else:
                    st.error(message)
        
        # Analysis summary
        if hasattr(coordinator, 'core_embeddings') and coordinator.core_embeddings is not None:
            st.subheader("3. Analysis Summary")
            summary = coordinator.get_analysis_summary()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model", summary['model_name'])
                st.metric("Embedding Dimension", summary['embedding_dimension'])
            
            with col2:
                if summary['core_corpus_processed']:
                    st.metric("Core Documents", summary['core_documents'])
                    st.metric("Core Paragraphs", summary['core_paragraphs'])
            
            with col3:
                if summary['target_corpus_processed']:
                    st.metric("Target Documents", summary['target_documents'])
                    st.metric("Target Paragraphs", summary['target_paragraphs'])
        
        # Similarity analysis
        if (hasattr(coordinator, 'core_embeddings') and coordinator.core_embeddings is not None and
            hasattr(coordinator, 'target_embeddings') and coordinator.target_embeddings is not None):
            
            st.subheader("4. Similarity Analysis")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button("Calculate Similarity Matrix"):
                    with st.spinner("Calculating similarities..."):
                        success, similarity_matrix, message = coordinator.calculate_core_target_similarity()
                    
                    if success:
                        st.success(message)
                        st.session_state.similarity_matrix = similarity_matrix
                        
                        # Show basic statistics
                        stats = coordinator.similarity_engine.get_similarity_statistics(similarity_matrix)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean Similarity", f"{stats['mean_similarity']:.3f}")
                            st.metric("Std Similarity", f"{stats['std_similarity']:.3f}")
                        
                        with col2:
                            st.metric("Min Similarity", f"{stats['min_similarity']:.3f}")
                            st.metric("Max Similarity", f"{stats['max_similarity']:.3f}")
                        
                        with col3:
                            st.metric("Median Similarity", f"{stats['median_similarity']:.3f}")
                            st.metric("75th Percentile", f"{stats['q75_similarity']:.3f}")
                    else:
                        st.error(message)
            
            with col2:
                if st.button("📖 How to Interpret Results", key="interpret_help"):
                    st.session_state.show_interpretation_modal = True
            
            # Modal for interpretation guidance
            if st.session_state.get('show_interpretation_modal', False):
                with st.container():
                    st.markdown("---")
                    st.markdown("### 📖 Interpreting Similarity Analysis Results")
                    
                    with st.expander("Understanding Similarity Scores", expanded=True):
                        st.markdown("""
                        **Similarity scores range from 0 to 1:**
                        - **0.0-0.3**: Low similarity - texts discuss different topics or concepts
                        - **0.3-0.5**: Moderate similarity - some shared themes or related concepts
                        - **0.5-0.7**: High similarity - substantial conceptual overlap
                        - **0.7-1.0**: Very high similarity - nearly identical meaning or very closely related content
                        
                        **Keep in mind:** These are semantic similarities based on meaning, not exact word matches.
                        """)
                    
                    with st.expander("Statistical Measures"):
                        st.markdown("""
                        **Mean Similarity**: Average influence level across your target corpus
                        - Higher mean suggests your target documents are generally influenced by your core texts
                        - Very low mean might indicate weak conceptual connections
                        
                        **Standard Deviation**: How much similarity varies across documents
                        - High std dev means some documents are very influenced, others not at all
                        - Low std dev suggests consistent influence levels
                        
                        **75th Percentile**: The similarity score that 75% of your comparisons fall below
                        - Useful for identifying the threshold for "highly influenced" content
                        """)
                    
                    with st.expander("Interpreting Results in Context"):
                        st.markdown("""
                        **Academic Research Context:**
                        - Look for clusters of high similarity to identify research lineages
                        - Low overall similarity might indicate novel or interdisciplinary work
                        - Consider your core corpus: very specialized texts will show lower similarity to general content
                        
                        **Influence Analysis:**
                        - High similarity doesn't always mean direct citation - it can indicate conceptual influence
                        - Look at the "Most Influential Texts" to see which core ideas are most prevalent
                        - Consider temporal patterns: do similarities increase over time periods?
                        
                        **Limitations to Consider:**
                        - Similarity is based on semantic meaning, not causal influence
                        - Short paragraphs may show lower similarity due to less context
                        - Domain-specific language can affect cross-disciplinary comparisons
                        """)
                    
                    col1, col2 = st.columns([1, 1])
                    with col2:
                        if st.button("Close", key="close_modal"):
                            st.session_state.show_interpretation_modal = False
                            st.rerun()
                    
                    st.markdown("---")
            
            # Most influential texts
            k = st.slider("Number of top texts to show:", 5, 50, 10)
            
            if st.button("Find Most Influential Core Texts"):
                with st.spinner("Finding most influential texts..."):
                    success, result, message = coordinator.find_most_influential_core_texts(k)
                
                if success:
                    st.success(message)
                    
                    # Display results
                    for text_info in result['influential_texts']:
                        # Get the actual paragraph text
                        para_meta = coordinator.core_metadata['paragraph_metadata'][text_info['paragraph_index']]
                        filename = text_info['filename']
                        para_idx = text_info['paragraph_in_doc']
                        
                        # Get the paragraph text from the core corpus
                        paragraph_text = ""
                        if filename in st.session_state.core_corpus['documents']:
                            paragraphs = st.session_state.core_corpus['documents'][filename]['paragraphs']
                            if para_idx < len(paragraphs):
                                paragraph_text = paragraphs[para_idx]
                        
                        # Extract first sentence (simple approach - split on period, exclamation, or question mark)
                        first_sentence = ""
                        if paragraph_text:
                            sentences = re.split(r'[.!?]+', paragraph_text)
                            if sentences:
                                first_sentence = sentences[0].strip()
                                if len(first_sentence) > 150:  # Truncate if too long
                                    first_sentence = first_sentence[:150] + "..."
                        
                        with st.expander(f"#{text_info['rank']} - {text_info['document_title']} (Similarity: {text_info['average_similarity']:.3f})"):
                            st.write(f"**Author:** {text_info['document_author']}")
                            st.write(f"**File:** {text_info['filename']}")
                            st.write(f"**Paragraph:** {text_info['paragraph_in_doc']}")
                            st.write(f"**Average Similarity:** {text_info['average_similarity']:.4f}")
                            if first_sentence:
                                st.write(f"**First sentence:** {first_sentence}")
                else:
                    st.error(message)
        
        # Navigation
        if (hasattr(coordinator, 'core_embeddings') and coordinator.core_embeddings is not None and
            hasattr(coordinator, 'target_embeddings') and coordinator.target_embeddings is not None):
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Continue to Vector Analysis"):
                    st.session_state.current_step = 'vectors'
                    st.rerun()
            with col2:
                if st.button("Continue to Results & Visualization"):
                    st.session_state.current_step = 'results'
                    st.rerun()
    
    def render_vector_analysis_page(self):
        """Render the vector analysis and custom dimensions page"""
        st.header("Vector Analysis & Custom Dimensions")
        
        # Check if analysis is ready
        if 'analysis_coordinator' not in st.session_state:
            st.warning("Please complete the analysis step first.")
            if st.button("Go to Analysis"):
                st.session_state.current_step = 'analysis'
                st.rerun()
            return
        
        coordinator = st.session_state.analysis_coordinator
        
        if coordinator.core_embeddings is None:
            st.warning("Please process the core corpus first.")
            return
        
        # Main tabs for vector analysis
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Suggested Terms", "🎯 Create Vectors", "📈 Vector Analysis", "🗺️ Vector Spaces", "📝 Topic Modeling"])

        with tab1:
            self._render_suggested_terms_tab(coordinator)
        
        with tab2:
            self._render_create_vectors_tab(coordinator)
        
        with tab3:
            self._render_vector_analysis_tab(coordinator)
        
        with tab4:
            self._render_vector_spaces_tab(coordinator)
        
        with tab5:
            self._render_topic_modeling_tab(coordinator)
    
    def _render_suggested_terms_tab(self, coordinator):
        """Render the suggested terms tab"""
        st.subheader("📊 Suggested Terms for Vector Creation")
        
        # Get suggested terms
        if coordinator.suggested_terms:
            st.success("✅ Term analysis complete!")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display suggested terms
                suggested_endpoints = coordinator.get_suggested_terms(top_n=30)
                
                if suggested_endpoints:
                    st.markdown("**Top suggested terms for creating vector endpoints:**")
                    
                    # Create columns for different sources
                    source_cols = st.columns(3)
                    
                    tfidf_terms = [t for t in suggested_endpoints if t['source'] == 'TF-IDF']
                    freq_terms = [t for t in suggested_endpoints if t['source'] == 'Frequency']
                    topic_terms = [t for t in suggested_endpoints if 'Topic' in t['source']]
                    
                    with source_cols[0]:
                        st.markdown("**TF-IDF Important Terms**")
                        for term in tfidf_terms[:10]:
                            st.write(f"• {term['term']} ({term['score']:.3f})")
                    
                    with source_cols[1]:
                        st.markdown("**Frequent Terms**")
                        for term in freq_terms[:10]:
                            st.write(f"• {term['term']} ({int(term['score'])})")
                    
                    with source_cols[2]:
                        st.markdown("**Topic Model Terms**")
                        for term in topic_terms[:10]:
                            st.write(f"• {term['term']} ({term['score']:.3f})")
                
                else:
                    st.info("No suggested terms available. Try reprocessing the core corpus.")
            
            with col2:
                st.markdown("### How to Use These Terms")
                st.markdown("""
                **Creating Effective Vectors:**
                
                1. **Choose contrasting concepts** - Pick terms that represent opposite ends of a spectrum
                
                2. **Use high-scoring terms** - Terms with higher TF-IDF or topic scores are more distinctive
                
                3. **Combine related terms** - Group similar concepts together for stronger vectors
                
                4. **Consider your research question** - What dimensions matter for your analysis?
                """)
                
                with st.expander("Example Vector Ideas"):
                    st.markdown("""
                    **Academic vs. Popular:**
                    - Positive: theoretical, methodology, empirical
                    - Negative: practical, everyday, common
                    
                    **Abstract vs. Concrete:**
                    - Positive: concept, theory, framework
                    - Negative: example, case, instance
                    
                    **Individual vs. Collective:**
                    - Positive: personal, individual, self
                    - Negative: social, community, group
                    """)
        
        else:
            st.info("Analyzing corpus for term suggestions...")
            if st.button("Refresh Term Analysis"):
                st.rerun()
    
    def _render_create_vectors_tab(self, coordinator):
        """Render the enhanced create vectors tab with improved mathematical methods"""
        st.subheader("🎯 Create Custom Vectors")
        
        # Display existing vectors with quality scores
        existing_vectors = coordinator.custom_vector_manager.list_vectors()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Create New Vector")
            
            # Get available methods
            methods = coordinator.custom_vector_manager.get_vector_creation_methods()
            
            # Method selection
            st.markdown("#### Method Selection")
            selected_method = st.selectbox(
                "Choose vector creation method:",
                options=[m['name'] for m in methods],
                format_func=lambda x: next(m['display_name'] for m in methods if m['name'] == x),
                help="Different methods for mathematically combining terms into vectors"
            )
            
            # Show method description
            method_info = next(m for m in methods if m['name'] == selected_method)
            
            # Color-code by complexity
            complexity_colors = {
                'Low': '🟢',
                'Medium': '🟡', 
                'High': '🔴'
            }
            complexity_indicator = complexity_colors.get(method_info['complexity'], '⚪')
            
            st.info(f"""
            **{method_info['display_name']}** {complexity_indicator}
            
            {method_info['description']}
            
            **Best for:** {method_info['best_for']}
            **Complexity:** {method_info['complexity']}
            """)
            
            # Vector creation form
            st.markdown("#### Vector Definition")
            vector_name = st.text_input("Vector Name", placeholder="e.g., Academic-Popular")
            description = st.text_area("Description (optional)", placeholder="Describe what this vector measures...")
            
            col_pos, col_neg = st.columns(2)
            
            with col_pos:
                st.markdown("**Positive Terms** (toward +)")
                positive_terms_text = st.text_area(
                    "Enter terms separated by commas", 
                    key="positive_terms",
                    placeholder="academic, theoretical, scholarly",
                    height=100,
                    help="Terms representing the positive direction of your vector"
                )
            
            with col_neg:
                st.markdown("**Negative Terms** (toward -)")
                negative_terms_text = st.text_area(
                    "Enter terms separated by commas", 
                    key="negative_terms", 
                    placeholder="popular, practical, everyday",
                    height=100,
                    help="Terms representing the negative direction (optional but recommended)"
                )
            
            # Parse terms
            positive_terms = [term.strip() for term in positive_terms_text.split(',') if term.strip()]
            negative_terms = [term.strip() for term in negative_terms_text.split(',') if term.strip()]
            
            # Real-time validation
            if positive_terms:
                with st.expander("Term Validation", expanded=False):
                    col_val1, col_val2 = st.columns(2)
                    
                    with col_val1:
                        st.markdown("**Positive Terms Validation**")
                        valid_pos, invalid_pos = coordinator.custom_vector_manager._validate_model_vocabulary(positive_terms)
                        for term in valid_pos:
                            st.success(f"✓ {term}")
                        for term in invalid_pos:
                            st.error(f"✗ {term} (not in model vocabulary)")
                    
                    with col_val2:
                        if negative_terms:
                            st.markdown("**Negative Terms Validation**")
                            valid_neg, invalid_neg = coordinator.custom_vector_manager._validate_model_vocabulary(negative_terms)
                            for term in valid_neg:
                                st.success(f"✓ {term}")
                            for term in invalid_neg:
                                st.error(f"✗ {term} (not in model vocabulary)")
                        else:
                            st.info("No negative terms provided")
                
                # Create vector button
                valid_pos, invalid_pos = coordinator.custom_vector_manager._validate_model_vocabulary(positive_terms)
                can_create = vector_name and valid_pos and len(valid_pos) > 0
                
                if st.button("Create Vector", disabled=not can_create, type="primary"):
                    if can_create:
                        with st.spinner("Creating vector..."):
                            success, message, vector = coordinator.custom_vector_manager.create_vector_from_terms(
                                vector_name, valid_pos, 
                                [term for term in negative_terms if term in coordinator.custom_vector_manager._validate_model_vocabulary(negative_terms)[0]] if negative_terms else [],
                                description, method=selected_method
                            )
                        
                        if success:
                            st.success(message)
                            
                            # Display quality metrics
                            vector_info = coordinator.custom_vector_manager.get_vector_info(vector_name)
                            if vector_info and 'quality_score' in vector_info:
                                quality_score = vector_info['quality_score']
                                quality_report = vector_info['quality_report']
                                
                                st.markdown("#### Vector Quality Assessment")
                                
                                # Quality score with color coding
                                if quality_score >= 0.8:
                                    st.success(f"**Quality Score: {quality_score:.2f}/1.0** - Excellent vector!")
                                elif quality_score >= 0.6:
                                    st.info(f"**Quality Score: {quality_score:.2f}/1.0** - Good vector")
                                elif quality_score >= 0.4:
                                    st.warning(f"**Quality Score: {quality_score:.2f}/1.0** - Moderate quality")
                                else:
                                    st.error(f"**Quality Score: {quality_score:.2f}/1.0** - Consider improving terms")
                                
                                # Show detailed metrics
                                with st.expander("Detailed Quality Metrics"):
                                    metrics = quality_report.get('metrics', {})
                                    
                                    col_m1, col_m2, col_m3 = st.columns(3)
                                    
                                    if 'separation' in metrics:
                                        with col_m1:
                                            st.metric("Term Separation", f"{metrics['separation']:.3f}")
                                    
                                    if 'stability' in metrics:
                                        with col_m2:
                                            st.metric("Vector Stability", f"{metrics['stability']:.3f}")
                                    
                                    if 'consistency' in metrics:
                                        with col_m3:
                                            st.metric("Term Consistency", f"{metrics['consistency']:.3f}")
                                
                                # Show recommendations
                                recommendations = quality_report.get('recommendations', [])
                                if recommendations:
                                    st.markdown("**Recommendations:**")
                                    for rec in recommendations:
                                        st.write(f"• {rec}")
                            
                            st.rerun()
                        else:
                            st.error(message)
            
            # Vector validation section
            if existing_vectors:
                st.markdown("---")
                st.markdown("#### Test Vector with Examples")
                
                # Select vector to validate
                vector_to_validate = st.selectbox(
                    "Select vector to test:",
                    options=[v['name'] for v in existing_vectors],
                    key="validation_vector_select"
                )
                
                if vector_to_validate:
                    col_pos_ex, col_neg_ex = st.columns(2)
                    
                    with col_pos_ex:
                        st.markdown("**Positive Examples**")
                        positive_examples_text = st.text_area(
                            "Enter example phrases/sentences (one per line)",
                            key="positive_examples",
                            placeholder="theoretical framework\nscholarly research\nacademic methodology",
                            height=80
                        )
                    
                    with col_neg_ex:
                        st.markdown("**Negative Examples**")
                        negative_examples_text = st.text_area(
                            "Enter example phrases/sentences (one per line)",
                            key="negative_examples",
                            placeholder="practical application\neveryday usage\ncommon sense",
                            height=80
                        )
                    
                    if st.button("Test Vector Performance"):
                        positive_examples = [ex.strip() for ex in positive_examples_text.split('\n') if ex.strip()]
                        negative_examples = [ex.strip() for ex in negative_examples_text.split('\n') if ex.strip()]
                        
                        if positive_examples:
                            with st.spinner("Testing vector..."):
                                success, validation_message = coordinator.custom_vector_manager.add_validation_examples(
                                    vector_to_validate, positive_examples, negative_examples
                                )
                            
                            if success:
                                st.success(validation_message)
                                
                                # Show validation results
                                validation_result = coordinator.custom_vector_manager.validation_examples.get(vector_to_validate)
                                if validation_result:
                                    col_v1, col_v2, col_v3 = st.columns(3)
                                    
                                    with col_v1:
                                        st.metric("Positive Score", f"{validation_result['positive_score']:.3f}")
                                    with col_v2:
                                        st.metric("Negative Score", f"{validation_result['negative_score']:.3f}")
                                    with col_v3:
                                        separation = validation_result['separation_score']
                                        st.metric("Separation", f"{separation:.3f}")
                                        
                                        if separation > 0.5:
                                            st.success("Good separation!")
                                        elif separation > 0.0:
                                            st.warning("Weak separation")
                                        else:
                                            st.error("Poor separation")
                            else:
                                st.error(validation_message)
                        else:
                            st.warning("Please provide at least some positive examples")
        
        with col2:
            st.markdown("### Existing Vectors")
            
            if existing_vectors:
                for vector in existing_vectors:
                    # Get additional info for display
                    vector_details = coordinator.custom_vector_manager.get_vector_info(vector['name'])
                    quality_score = vector_details.get('quality_score', 0.0) if vector_details else 0.0
                    method = vector_details.get('method', 'unknown') if vector_details else 'unknown'
                    
                    # Quality indicator
                    if quality_score >= 0.8:
                        quality_emoji = "🟢"
                    elif quality_score >= 0.6:
                        quality_emoji = "🟡"
                    else:
                        quality_emoji = "🔴"
                    
                    with st.expander(f"{quality_emoji} {vector['name']} ({vector['positive_terms_count']}+/{vector['negative_terms_count']}-)"):
                        st.write(f"**Description:** {vector['description'] or 'No description'}")
                        st.write(f"**Method:** {method}")
                        st.write(f"**Quality:** {quality_score:.2f}/1.0")
                        st.write(f"**Created:** {vector['created_at'][:10]}")
                        
                        # Show terms with a toggle button instead of nested expander
                        if vector_details:
                            terms_key = f"show_terms_{vector['name']}"
                            if st.button(f"{'Hide' if st.session_state.get(terms_key, False) else 'Show'} Terms", 
                                        key=f"toggle_terms_{vector['name']}"):
                                st.session_state[terms_key] = not st.session_state.get(terms_key, False)
                            
                            if st.session_state.get(terms_key, False):
                                pos_terms = vector_details.get('positive_terms', [])
                                neg_terms = vector_details.get('negative_terms', [])
                                
                                col_pos_display, col_neg_display = st.columns(2)
                                with col_pos_display:
                                    st.markdown("**Positive:**")
                                    for term in pos_terms:
                                        st.write(f"• {term}")
                                
                                with col_neg_display:
                                    st.markdown("**Negative:**")
                                    for term in neg_terms:
                                        st.write(f"• {term}")
                        
                        # Show validation status
                        if vector_details and 'validation' in vector_details:
                            validation = vector_details['validation']
                            separation = validation.get('separation_score', 0)
                            st.info(f"Validated: {separation:.3f} separation score")
                        
                        col_edit, col_delete = st.columns(2)
                        with col_edit:
                            if st.button("Edit", key=f"edit_{vector['name']}"):
                                st.session_state.editing_vector = vector['name']
                        
                        with col_delete:
                            if st.button("Delete", key=f"delete_{vector['name']}"):
                                success, message = coordinator.custom_vector_manager.delete_vector(vector['name'])
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
            else:
                st.info("No custom vectors created yet.")
            
            # Vector creation tips
            with st.expander("💡 Vector Creation Tips"):
                st.markdown("""
                **For better vectors:**
                
                1. **Use contrasting terms** - Pick opposites that clearly represent different concepts
                
                2. **Test with examples** - Validate your vectors with real sentences
                
                3. **Quality scores matter:**
                   - 🟢 0.8+: Excellent discrimination
                   - 🟡 0.6-0.8: Good performance  
                   - 🔴 <0.6: Consider revising terms
                
                4. **Method selection:**
                   - **Orthogonal Projection**: Most balanced, good default
                   - **Weighted Difference**: When terms have different importance
                   - **PCA Axis**: For clear conceptual opposites
                """)
                
            st.markdown(f"**Vectors created:** {len(existing_vectors)}/50")
    
    def _render_vector_analysis_tab(self, coordinator):
        """Render the vector analysis tab"""
        st.subheader("📈 Vector Analysis")
        
        existing_vectors = coordinator.custom_vector_manager.list_vectors()
        
        if not existing_vectors:
            st.info("Create some custom vectors first to analyze them.")
            return
        
        # Vector selection
        vector_names = [v['name'] for v in existing_vectors]
        selected_vector = st.selectbox("Select vector to analyze:", vector_names)
        
        # Corpus selection
        corpus_choice = st.radio("Analyze which corpus:", ["Target Corpus", "Core Corpus"])
        use_target = corpus_choice == "Target Corpus"
        
        if selected_vector:
            if st.button("Analyze Vector Projections"):
                with st.spinner("Analyzing vector projections..."):
                    success, results, message = coordinator.analyze_document_projections(
                        selected_vector, use_target=use_target
                    )
                
                if success:
                    st.success(message)
                    
                    # Display results
                    proj_results = results['projection_results']
                    performance = results['performance_analysis']
                    extremes = results['extreme_documents']
                    vector_info = results['vector_info']
                    
                    # Vector information
                    with st.expander("Vector Information", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Positive terms:** {', '.join(vector_info['positive_terms'])}")
                        with col2:
                            st.write(f"**Negative terms:** {', '.join(vector_info['negative_terms']) or 'None'}")
                        
                        if vector_info['description']:
                            st.write(f"**Description:** {vector_info['description']}")
                    
                    # Statistics
                    st.markdown("### Projection Statistics")
                    stats = proj_results['statistics']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{stats['mean_projection']:.3f}")
                        st.metric("Std Dev", f"{stats['std_projection']:.3f}")
                    with col2:
                        st.metric("Min", f"{stats['min_projection']:.3f}")
                        st.metric("Max", f"{stats['max_projection']:.3f}")
                    with col3:
                        st.metric("Range", f"{stats['max_projection'] - stats['min_projection']:.3f}")
                        st.metric("Median", f"{stats['median_projection']:.3f}")
                    with col4:
                        st.metric("Q25", f"{stats['q25_projection']:.3f}")
                        st.metric("Q75", f"{stats['q75_projection']:.3f}")
                    
                    # Performance analysis
                    if performance:
                        st.markdown("### Vector Performance")
                        st.write(f"**Performance Category:** {performance['performance_category']}")
                        
                        st.markdown("**Recommendations:**")
                        for rec in performance['recommendations']:
                            st.write(f"• {rec}")
                    
                    # Extreme documents
                    if extremes:
                        st.markdown("### Extreme Documents")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Most Positive**")
                            for doc in extremes['most_positive'][:3]:
                                st.write(f"• {doc['document_metadata']['document_metadata']['title'][:50]}... ({doc['projection_score']:.3f})")
                        
                        with col2:
                            st.markdown("**Most Negative**")
                            for doc in extremes['most_negative'][:3]:
                                st.write(f"• {doc['document_metadata']['document_metadata']['title'][:50]}... ({doc['projection_score']:.3f})")
                        
                        with col3:
                            st.markdown("**Most Neutral**")
                            for doc in extremes['most_neutral'][:3]:
                                st.write(f"• {doc['document_metadata']['document_metadata']['title'][:50]}... ({doc['projection_score']:.3f})")
                
                else:
                    st.error(message)
    
    def _render_vector_spaces_tab(self, coordinator):
        """Render the vector spaces tab with interactive plots"""
        st.subheader("🗺️ Multi-Dimensional Vector Spaces")
        
        existing_vectors = coordinator.custom_vector_manager.list_vectors()
        
        if len(existing_vectors) < 2:
            st.info("Create at least 2 custom vectors to create vector spaces.")
            return
        
        vector_names = [v['name'] for v in existing_vectors]
        
        # Space type selection
        space_type = st.radio("Select space type:", ["2D Space (2 vectors)", "3D Space (3 vectors)"])
        
        if space_type == "2D Space (2 vectors)":
            col1, col2 = st.columns(2)
            
            with col1:
                vector1 = st.selectbox("X-axis vector:", vector_names, key="2d_v1")
            with col2:
                vector2 = st.selectbox("Y-axis vector:", [v for v in vector_names if v != vector1], key="2d_v2")
            
            corpus_choice = st.radio("Analyze which corpus:", ["Target Corpus", "Core Corpus"], key="2d_corpus")
            use_target = corpus_choice == "Target Corpus"
            
            if vector1 and vector2 and st.button("Create 2D Space"):
                with st.spinner("Creating 2D vector space..."):
                    success, results, message = coordinator.create_2d_vector_space(
                        vector1, vector2, use_target=use_target
                    )
                
                if success:
                    st.success(message)
                    
                    # Display 2D space results
                    coordinates = results['coordinates']
                    quadrant_stats = results['quadrant_statistics']
                    
                    st.markdown("### 2D Vector Space")
                    st.write(f"**X-axis:** {vector1}")
                    st.write(f"**Y-axis:** {vector2}")
                    st.write(f"**Orthogonality Score:** {results['orthogonality_score']:.3f} (1.0 = perfectly orthogonal)")
                    
                    # Create the 2D scatter plot
                    try:
                        import plotly.express as px
                        import plotly.graph_objects as go
                        import pandas as pd
                        
                        # Prepare data for plotting
                        plot_data = []
                        for coord in coordinates:
                            plot_data.append({
                                'x': coord['x'],
                                'y': coord['y'],
                                'title': coord['document_metadata']['document_metadata']['title'][:50] + "...",
                                'quadrant': coord['quadrant'],
                                'filename': coord['document_metadata']['filename']
                            })
                        
                        df_plot = pd.DataFrame(plot_data)
                        
                        # Create scatter plot with quadrant coloring
                        fig = px.scatter(
                            df_plot, 
                            x='x', 
                            y='y',
                            color='quadrant',
                            hover_data=['title', 'filename'],
                            title=f"2D Vector Space: {vector1} vs {vector2}",
                            labels={
                                'x': f'{vector1} →',
                                'y': f'{vector2} →'
                            }
                        )
                        
                        # Add quadrant lines
                        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
                        
                        # Customize layout
                        fig.update_layout(
                            width=800,
                            height=600,
                            showlegend=True,
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            )
                        )
                        
                        # Add quadrant labels
                        x_range = [df_plot['x'].min(), df_plot['x'].max()]
                        y_range = [df_plot['y'].min(), df_plot['y'].max()]
                        
                        # Calculate label positions
                        x_offset = (x_range[1] - x_range[0]) * 0.1
                        y_offset = (y_range[1] - y_range[0]) * 0.1
                        
                        fig.add_annotation(x=x_range[1] - x_offset, y=y_range[1] - y_offset, 
                                         text="Q1 (+,+)", showarrow=False, font=dict(size=12, color="gray"))
                        fig.add_annotation(x=x_range[0] + x_offset, y=y_range[1] - y_offset, 
                                         text="Q2 (-,+)", showarrow=False, font=dict(size=12, color="gray"))
                        fig.add_annotation(x=x_range[0] + x_offset, y=y_range[0] + y_offset, 
                                         text="Q3 (-,-)", showarrow=False, font=dict(size=12, color="gray"))
                        fig.add_annotation(x=x_range[1] - x_offset, y=y_range[0] + y_offset, 
                                         text="Q4 (+,-)", showarrow=False, font=dict(size=12, color="gray"))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except ImportError:
                        st.error("Plotly not available. Install with: pip install plotly")
                        # Fall back to coordinate display
                        st.markdown("### Document Coordinates")
                        coord_df = pd.DataFrame([{
                            'Document': coord['document_metadata']['document_metadata']['title'][:50],
                            'X': f"{coord['x']:.3f}",
                            'Y': f"{coord['y']:.3f}",
                            'Quadrant': coord['quadrant']
                        } for coord in coordinates[:20]])  # Show first 20
                        st.dataframe(coord_df)
                    
                    # Quadrant statistics
                    st.markdown("### Quadrant Distribution")
                    quad_cols = st.columns(4)
                    
                    for i, (quad_name, quad_data) in enumerate(quadrant_stats.items()):
                        with quad_cols[i % 4]:
                            st.metric(quad_name, f"{quad_data['count']} docs", f"{quad_data['percentage']:.1f}%")
                    
                    # Sample documents from each quadrant (collapsed by default)
                    with st.expander("Sample Documents by Quadrant", expanded=False):
                        for quad_name, quad_data in quadrant_stats.items():
                            if quad_data['documents']:
                                st.markdown(f"**{quad_name} ({quad_data['count']} documents)**")
                                for doc in quad_data['documents'][:3]:
                                    title = doc['document_metadata']['document_metadata']['title']
                                    st.write(f"• {title[:60]}... (x:{doc['x']:.2f}, y:{doc['y']:.2f})")
                                st.markdown("---")
                
                else:
                    st.error(message)
        
        elif space_type == "3D Space (3 vectors)" and len(existing_vectors) >= 3:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                vector1 = st.selectbox("X-axis vector:", vector_names, key="3d_v1")
            with col2:
                vector2 = st.selectbox("Y-axis vector:", [v for v in vector_names if v != vector1], key="3d_v2")
            with col3:
                vector3 = st.selectbox("Z-axis vector:", [v for v in vector_names if v not in [vector1, vector2]], key="3d_v3")
            
            corpus_choice = st.radio("Analyze which corpus:", ["Target Corpus", "Core Corpus"], key="3d_corpus")
            use_target = corpus_choice == "Target Corpus"
            
            if vector1 and vector2 and vector3 and st.button("Create 3D Space"):
                with st.spinner("Creating 3D vector space..."):
                    success, results, message = coordinator.create_3d_vector_space(
                        vector1, vector2, vector3, use_target=use_target
                    )
                
                if success:
                    st.success(message)
                    
                    # Display 3D space results
                    coordinates = results['coordinates']
                    octant_stats = results['octant_statistics']
                    
                    st.markdown("### 3D Vector Space")
                    st.write(f"**X-axis:** {vector1}")
                    st.write(f"**Y-axis:** {vector2}")
                    st.write(f"**Z-axis:** {vector3}")
                    
                    # Create the 3D scatter plot
                    try:
                        import plotly.express as px
                        import plotly.graph_objects as go
                        import pandas as pd
                        
                        # Prepare data for plotting
                        plot_data = []
                        for coord in coordinates:
                            plot_data.append({
                                'x': coord['x'],
                                'y': coord['y'],
                                'z': coord['z'],
                                'title': coord['document_metadata']['document_metadata']['title'][:50] + "...",
                                'octant': coord['octant'],
                                'filename': coord['document_metadata']['filename']
                            })
                        
                        df_plot = pd.DataFrame(plot_data)
                        
                        # Create 3D scatter plot
                        fig = px.scatter_3d(
                            df_plot, 
                            x='x', 
                            y='y', 
                            z='z',
                            color='octant',
                            hover_data=['title', 'filename'],
                            title=f"3D Vector Space: {vector1} vs {vector2} vs {vector3}",
                            labels={
                                'x': f'{vector1} →',
                                'y': f'{vector2} →',
                                'z': f'{vector3} →'
                            }
                        )
                        
                        # Add reference lines at origin
                        # X-axis line
                        x_range = [df_plot['x'].min(), df_plot['x'].max()]
                        fig.add_trace(go.Scatter3d(
                            x=x_range, y=[0, 0], z=[0, 0],
                            mode='lines',
                            line=dict(color='gray', width=2, dash='dash'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        # Y-axis line
                        y_range = [df_plot['y'].min(), df_plot['y'].max()]
                        fig.add_trace(go.Scatter3d(
                            x=[0, 0], y=y_range, z=[0, 0],
                            mode='lines',
                            line=dict(color='gray', width=2, dash='dash'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        # Z-axis line
                        z_range = [df_plot['z'].min(), df_plot['z'].max()]
                        fig.add_trace(go.Scatter3d(
                            x=[0, 0], y=[0, 0], z=z_range,
                            mode='lines',
                            line=dict(color='gray', width=2, dash='dash'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        # Customize layout
                        fig.update_layout(
                            width=900,
                            height=700,
                            scene=dict(
                                xaxis_title=f'{vector1} →',
                                yaxis_title=f'{vector2} →',
                                zaxis_title=f'{vector3} →',
                                aspectmode='cube'
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except ImportError:
                        st.error("Plotly not available. Install with: pip install plotly")
                        # Fall back to coordinate display
                        st.markdown("### Document Coordinates")
                        coord_df = pd.DataFrame([{
                            'Document': coord['document_metadata']['document_metadata']['title'][:50],
                            'X': f"{coord['x']:.3f}",
                            'Y': f"{coord['y']:.3f}",
                            'Z': f"{coord['z']:.3f}",
                            'Octant': coord['octant']
                        } for coord in coordinates[:20]])  # Show first 20
                        st.dataframe(coord_df)
                    
                    # Octant statistics
                    st.markdown("### Octant Distribution")
                    
                    # Add octant explanation
                    with st.expander("Understanding Octants", expanded=False):
                        st.markdown("""
                        **Octant Layout (3D Quadrants):**
                        An octant divides 3D space into 8 regions using three axes.
                        
                        **What the signs mean:**
                        - **First sign**: Position on X-axis (your first vector)
                        - **Second sign**: Position on Y-axis (your second vector)  
                        - **Third sign**: Position on Z-axis (your third vector)
                        - **Positive (+)**: High on that vector dimension
                        - **Negative (-)**: Low on that vector dimension
                        
                        **Example interpretations:**
                        - **O(+,+,+)**: High on all three vectors
                        - **O(-,+,-)**: Low on X-axis, High on Y-axis, Low on Z-axis
                        - **O(-,-,-)**: Low on all three vectors
                        
                        Each octant represents a unique combination of high/low positions across your three chosen dimensions.
                        """)
                    
                    oct_cols = st.columns(4)
                    
                    for i, (oct_name, oct_data) in enumerate(octant_stats.items()):
                        with oct_cols[i % 4]:
                            st.metric(oct_name, f"{oct_data['count']} docs", f"{oct_data['percentage']:.1f}%")
                    
                    # Sample documents from each octant (collapsed by default)
                    with st.expander("Sample Documents by Octant", expanded=False):
                        for oct_name, oct_data in octant_stats.items():
                            if oct_data['documents']:
                                st.markdown(f"**{oct_name} ({oct_data['count']} documents)**")
                                for doc in oct_data['documents'][:2]:
                                    title = doc['document_metadata']['document_metadata']['title']
                                    st.write(f"• {title[:50]}... (x:{doc['x']:.2f}, y:{doc['y']:.2f}, z:{doc['z']:.2f})")
                                st.markdown("---")
                
                else:
                    st.error(message)
    
    # Add this new tab to your vector analysis page
    def _render_topic_modeling_tab(self, coordinator):
        """Render topic modeling tab with BERTopic configuration"""
        st.subheader("📝 Topic Modeling with BERTopic")

        # Check full prerequisite chain
        if 'core_corpus' not in st.session_state:
            st.warning("Please upload core corpus first.")
            return
        
        if 'analysis_coordinator' not in st.session_state:
            st.warning("Please complete the preprocessing/analysis step first.")
            return
            
        if not hasattr(coordinator.embedding_engine, 'model') or coordinator.embedding_engine.model is None:
            st.warning("Please initialize the model in the Preprocessing/Analysis step first.")
            return
            
        if coordinator.core_embeddings is None:
            st.warning("Please process the core corpus embeddings first.")
            return
        
        st.markdown("""
        Extract topics from your core corpus using BERTopic, with full control over preprocessing 
        parameters to get the most relevant topics for your analysis. Then you can see which documents in your target contain similar topics.
        """)
        
        # Topic modeling configuration
        with st.expander("Topic Modeling Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Basic Parameters")
                
                # Number of topics
                n_topics = st.selectbox(
                    "Number of topics:",
                    options=["auto", 5, 10, 15, 20, 25, 30],
                    help="'auto' lets BERTopic determine optimal number of topics"
                )
                
                # Minimum topic size
                min_topic_size = st.number_input(
                    "Minimum topic size:",
                    min_value=2,
                    max_value=50,
                    value=10,
                    help="Minimum number of documents required to form a topic"
                )
                
                # Language model for representation
                representation_model = st.selectbox(
                    "Topic representation model:",
                    options=[
                        "KeyBERT",
                        "MaximalMarginalRelevance", 
                        "PartOfSpeech",
                        "TextGeneration"
                    ],
                    help="Method for generating topic word representations"
                )
            
            with col2:
                st.markdown("#### Stopwords Configuration")
                
                # Stopwords selection
                stopword_option = st.radio(
                    "Stopwords strategy:",
                    options=[
                        "Default English stopwords",
                        "No stopwords", 
                        "Custom stopwords",
                        "Extended academic stopwords"
                    ]
                )
                
                if stopword_option == "Custom stopwords":
                    custom_stopwords_text = st.text_area(
                        "Enter custom stopwords (one per line):",
                        placeholder="the\nand\nof\nto\na\nin\nis\nit\nyou\nthat\nhe\nwas\nfor\non\nare\nas\nwith\nhis\nthey\nat\nbe\nthis\nhave\nfrom\nor\none\nhad\nby\nword\nbut\nnot\nwhat\nall\nwere\nwe\nwhen\nyour\ncan\nsaid\nthere\neach\nwhich\ndo\nhow\ntheir\nif\nwill\nup\nother\nabout\nout\nmany\nthen\nthem\nthese\nso\nsome\nher\nwould\nmake\nlike\ninto\nhim\nhas\ntwo\nmore\nher\ngo\nno\nway\ncould\nmy\nthan\nfirst\nbeen\ncall\nwho\nits\nnow\nfind\nlong\ndown\nday\ndid\nget\ncome\nmade\nmay\npart",
                        height=150
                    )
                    
                    # Parse custom stopwords
                    if custom_stopwords_text:
                        custom_stopwords = [word.strip().lower() for word in custom_stopwords_text.split('\n') if word.strip()]
                        st.info(f"Using {len(custom_stopwords)} custom stopwords")
                
                elif stopword_option == "Extended academic stopwords":
                    st.info("Using extended academic stopwords (includes common academic terms)")
                
                # Advanced preprocessing options
                st.markdown("#### Text Preprocessing")
                
                remove_punctuation = st.checkbox("Remove punctuation", value=True)
                min_word_length = st.number_input("Minimum word length:", min_value=1, max_value=10, value=2)
                lowercase = st.checkbox("Convert to lowercase", value=True)
                remove_numbers = st.checkbox("Remove numbers", value=False)
        
        # Topic modeling execution
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("Run Topic Modeling", type="primary"):
                # Prepare stopwords based on selection
                stopwords = None
                if stopword_option == "Default English stopwords":
                    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
                    stopwords = list(ENGLISH_STOP_WORDS)
                elif stopword_option == "No stopwords":
                    stopwords = []
                elif stopword_option == "Custom stopwords":
                    stopwords = custom_stopwords if 'custom_stopwords' in locals() else []
                elif stopword_option == "Extended academic stopwords":
                    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
                    academic_stopwords = [
                        'study', 'research', 'analysis', 'paper', 'article', 'journal', 
                        'author', 'authors', 'university', 'department', 'conclusion',
                        'abstract', 'introduction', 'methodology', 'results', 'discussion',
                        'figure', 'table', 'section', 'chapter', 'page', 'pages',
                        'et', 'al', 'etc', 'ie', 'eg', 'via', 'versus', 'vs'
                    ]
                    stopwords = list(ENGLISH_STOP_WORDS) + academic_stopwords
                
                # Prepare configuration
                topic_config = {
                    'n_topics': None if n_topics == "auto" else n_topics,
                    'min_topic_size': min_topic_size,
                    'stopwords': stopwords,
                    'representation_model': representation_model,
                    'preprocessing': {
                        'remove_punctuation': remove_punctuation,
                        'min_word_length': min_word_length,
                        'lowercase': lowercase,
                        'remove_numbers': remove_numbers
                    }
                }
                
                with st.spinner("Running topic modeling... This may take a few minutes."):
                    success, topic_results, message = coordinator.run_enhanced_topic_modeling(
                        st.session_state.core_corpus, topic_config
                    )
                
                if success:
                    st.success(message)
                    
                    # Store results in session state
                    st.session_state.topic_modeling_results = topic_results
                    
                    # Display topic overview
                    st.markdown("### Topic Modeling Results")
                    
                    topics_df = topic_results['topics_dataframe']
                    
                    # Topic summary metrics
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        st.metric("Topics Found", len(topics_df))
                    with col_m2:
                        st.metric("Documents Processed", topic_results['total_documents'])
                    with col_m3:
                        st.metric("Coherence Score", f"{topic_results.get('coherence_score', 0):.3f}")
                    with col_m4:
                        st.metric("Outliers", topic_results.get('outlier_count', 0))
                    
                else:
                    st.error(message)
        
        with col2:
            st.markdown("### Stopwords Preview")
            
            # Show preview of selected stopwords
            if stopword_option == "Default English stopwords":
                from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
                preview_words = sorted(list(ENGLISH_STOP_WORDS))[:20]
                st.write(f"Using {len(ENGLISH_STOP_WORDS)} default English stopwords:")
                st.text(", ".join(preview_words) + "...")
                
            elif stopword_option == "Extended academic stopwords":
                st.write("Includes default English stopwords plus academic terms:")
                academic_preview = ["study", "research", "analysis", "methodology", "authors", "university", "conclusion", "abstract"]
                st.text(", ".join(academic_preview) + "...")
                
            elif stopword_option == "Custom stopwords" and 'custom_stopwords' in locals():
                st.write(f"Your {len(custom_stopwords)} custom stopwords:")
                st.text(", ".join(custom_stopwords[:10]) + ("..." if len(custom_stopwords) > 10 else ""))
                
            elif stopword_option == "No stopwords":
                st.info("No stopwords will be removed")
        
        # Display existing results if available
        if 'topic_modeling_results' in st.session_state:
            st.markdown("---")
            results = st.session_state.topic_modeling_results
            
            # Topic details
            st.markdown("### Topic Details")
            
            # Topic selection for detailed view
            topic_options = results['topics_dataframe']['Topic'].unique()
            selected_topic = st.selectbox(
                "Select topic to examine:",
                options=topic_options,
                format_func=lambda x: f"Topic {x}: {results['topic_labels'].get(x, 'Unknown')}"
            )
            
            if selected_topic is not None:
                topic_info = results['topics_dataframe'][results['topics_dataframe']['Topic'] == selected_topic]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"#### Topic {selected_topic}")
                    st.write(f"**Label:** {results['topic_labels'].get(selected_topic, 'Unknown')}")
                    st.write(f"**Document Count:** {len(topic_info)}")
                    
                    # Top words for this topic
                    if 'topic_words' in results and selected_topic in results['topic_words']:
                        st.markdown("**Top Words:**")
                        for word, score in results['topic_words'][selected_topic][:10]:
                            st.write(f"• {word} ({score:.3f})")
                
                with col2:
                    st.markdown("#### Sample Documents")
                    sample_docs = topic_info.head(5)
                    for _, doc in sample_docs.iterrows():
                        st.write(f"• {doc.get('Document', 'Unknown')[:60]}...")
            
            # Topic similarity to core texts
            # Topic similarity to core texts
            if st.button("Analyze Topic-Target Similarity"):
                with st.spinner("Calculating how target documents relate to core topics..."):
                    target_similarities = coordinator.calculate_topic_target_similarities(results)
                    
                    if target_similarities:
                        # Store in session state
                        st.session_state.target_similarities = target_similarities
                        st.success("Analysis complete!")
                    else:
                        st.error("Could not analyze topic-target similarities. Make sure both corpora are processed.")

            # Display results if they exist in session state
            if 'target_similarities' in st.session_state:
                target_similarities = st.session_state.target_similarities
                
                st.markdown("### Target Corpus Topic Analysis")
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Target Documents", target_similarities['total_target_documents'])
                with col2:
                    st.metric("Topics Found", target_similarities['topics_found_in_target'])
                with col3:
                    outliers = target_similarities['topic_distribution'].get(-1, 0)
                    st.metric("Outlier Documents", outliers)
                
                # Network visualization checkbox (avoids button conflicts)
                show_network = st.checkbox("Show Network Visualization")
                
                if show_network:
                    with st.spinner("Creating network visualization..."):
                        network_data = coordinator.create_topic_network_visualization(target_similarities)
                        
                        if network_data and 'figure' in network_data:
                            st.plotly_chart(network_data['figure'], use_container_width=True)
                            
                            # Network statistics
                            stats = network_data['network_stats']
                            st.markdown("#### Network Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Nodes", stats['nodes'])
                            with col2:
                                st.metric("Connections", stats['edges'])
                            with col3:
                                st.metric("Density", f"{stats['density']:.3f}")
                            with col4:
                                st.metric("Components", stats['components'])
                        
                        elif network_data and 'error' in network_data:
                            st.error(f"Network visualization error: {network_data['error']}")
                
                # Document analysis with paragraph numbers
                st.markdown("#### Document Topic Composition")
                
                for doc in target_similarities['target_analysis'][:15]:  # Show first 15 documents
                    # Create more informative title with paragraph info
                    doc_title = f"{doc['target_document']['title']} (Paragraph {doc['target_document']['paragraph_number']})"
                    primary_topic = doc['primary_topic']['label']
                    
                    with st.expander(f"{doc_title} - Primary: {primary_topic}"):
                        
                        # Document info
                        st.write(f"**Document:** {doc['target_document']['filename']}")
                        st.write(f"**Author:** {doc['target_document']['author']}")
                        st.write(f"**Paragraph:** {doc['target_document']['paragraph_number']}")
                        
                        # Show topic composition as percentages
                        if doc['topic_composition']:
                            st.markdown("**Topic Composition:**")
                            for topic_id, topic_info in list(doc['topic_composition'].items())[:3]:
                                percentage = topic_info['probability'] * 100
                                st.write(f"• {topic_info['label']}: {percentage:.1f}%")
                        
                        # Most similar core document with paragraph number
                        if doc['most_similar_core']:
                            core_doc = doc['most_similar_core']
                            st.write(f"**Most similar core text:** {core_doc['title']} (Paragraph {core_doc['paragraph_number']}) - Similarity: {core_doc['similarity']:.3f}")
                            st.write(f"**Core file:** {core_doc['filename']}")
                        
                        # Text preview
                        st.write(f"**Text preview:** {doc['text_preview']}")

        # Help section
        with st.expander("Topic Modeling Help"):
            st.markdown("""
            **Understanding the Parameters:**
            
            **Number of Topics:**
            - 'Auto': Let BERTopic find the optimal number
            - Fixed number: Force specific number of topics
            - The topic # `-1` is a special topic for 'outlier' documents
            
            **Minimum Topic Size:**
            - Higher values = fewer, more coherent topics
            - Lower values = more topics, potentially noisier
            
            **Stopwords Strategy:**
            - **Default**: Standard English stopwords
            - **Extended Academic**: Includes common academic terms
            - **Custom**: Your domain-specific stopwords
            - **None**: Keep all words (may include noise)
            
            **Representation Models:**
            - **KeyBERT**: Keyword-based topic representation
            - **MaximalMarginalRelevance**: Diverse, relevant keywords
            - **PartOfSpeech**: Focus on specific word types
            - **TextGeneration**: AI-generated topic descriptions
            """)

    def render_comprehensive_results(self):
        """Render comprehensive results page with all analyses and export options"""
        st.header("📊 Comprehensive Analysis Results")
        
        # Check prerequisites
        if 'analysis_coordinator' not in st.session_state:
            st.warning("No analysis data available. Please complete the analysis workflow first.")
            self._render_analysis_navigation_buttons()
            return
        
        coordinator = st.session_state.analysis_coordinator
        
        if coordinator.core_embeddings is None or coordinator.target_embeddings is None:
            st.warning("Both core and target corpora must be processed to view results.")
            self._render_analysis_navigation_buttons()
            return
        
        # Executive Summary at the top
        self._render_executive_summary(coordinator)
        
        # Main results tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Similarity Analysis", 
            "🎯 Vector Analysis", 
            "📝 Topic Analysis", 
            "🌐 Network View",
            "📋 Detailed Reports",
            "💾 Export Data"
        ])
        
        with tab1:
            self._render_similarity_results(coordinator)
        
        with tab2:
            self._render_vector_results(coordinator)
        
        with tab3:
            self._render_topic_results(coordinator)
        
        with tab4:
            self._render_network_results(coordinator)
        
        with tab5:
            self._render_detailed_reports(coordinator)
        
        with tab6:
            self._render_export_section(coordinator)

    def _render_executive_summary(self, coordinator):
        """Render executive summary of all analyses"""
        st.subheader("Executive Summary")
        
        # Get basic stats
        summary = coordinator.get_analysis_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Core Documents", summary.get('core_documents', 0))
            st.metric("Core Paragraphs", summary.get('core_paragraphs', 0))
        
        with col2:
            st.metric("Target Documents", summary.get('target_documents', 0))
            st.metric("Target Paragraphs", summary.get('target_paragraphs', 0))
        
        with col3:
            st.metric("Custom Vectors", summary.get('custom_vectors_count', 0))
            st.metric("Model Used", summary.get('model_name', 'Unknown'))
        
        with col4:
            # Calculate overall influence score if similarity matrix exists
            if hasattr(st.session_state, 'similarity_matrix'):
                overall_influence = float(np.mean(st.session_state.similarity_matrix))
                st.metric("Overall Influence Score", f"{overall_influence:.3f}")
            else:
                st.metric("Overall Influence Score", "Not calculated")
        
        # Key findings summary
        st.markdown("### Key Findings")
        findings = self._generate_key_findings(coordinator)
        for finding in findings:
            st.write(f"• {finding}")

    def _generate_key_findings(self, coordinator):
        """Generate key findings from all analyses"""
        findings = []
        
        # Similarity analysis findings
        if hasattr(st.session_state, 'similarity_matrix'):
            similarity_matrix = st.session_state.similarity_matrix
            stats = coordinator.similarity_engine.get_similarity_statistics(similarity_matrix)
            
            findings.append(f"Average semantic similarity between target and core texts: {stats['mean_similarity']:.3f}")
            
            if stats['mean_similarity'] > 0.5:
                findings.append("High overall influence detected - target corpus shows strong alignment with core concepts")
            elif stats['mean_similarity'] > 0.3:
                findings.append("Moderate influence detected - target corpus partially reflects core concepts")
            else:
                findings.append("Low overall influence - target corpus appears largely independent of core concepts")
        
        # Vector analysis findings
        if coordinator.custom_vector_manager.custom_vectors:
            findings.append(f"Created {len(coordinator.custom_vector_manager.custom_vectors)} custom analytical dimensions")
        
        # Topic analysis findings
        if 'topic_modeling_results' in st.session_state:
            topic_results = st.session_state.topic_modeling_results
            findings.append(f"Identified {len(topic_results['topics_dataframe'])} distinct topics in core corpus")
        
        if 'target_similarities' in st.session_state:
            target_sims = st.session_state.target_similarities
            findings.append(f"Analyzed {target_sims['total_target_documents']} target documents against core topics")
            
            outliers = target_sims['topic_distribution'].get(-1, 0)
            if outliers > 0:
                outlier_pct = (outliers / target_sims['total_target_documents']) * 100
                findings.append(f"{outlier_pct:.1f}% of target documents show novel themes not found in core corpus")
        
        return findings

    def _render_similarity_results(self, coordinator):
        """Render similarity analysis results"""
        st.subheader("Core-Target Similarity Analysis")
        
        # Check if similarity matrix exists
        if not hasattr(st.session_state, 'similarity_matrix'):
            st.info("Similarity analysis not yet performed.")
            if st.button("Run Similarity Analysis"):
                with st.spinner("Calculating similarities..."):
                    success, similarity_matrix, message = coordinator.calculate_core_target_similarity()
                    if success:
                        st.session_state.similarity_matrix = similarity_matrix
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            return
        
        similarity_matrix = st.session_state.similarity_matrix
        stats = coordinator.similarity_engine.get_similarity_statistics(similarity_matrix)
        
        # Statistics display
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Similarity", f"{stats['mean_similarity']:.3f}")
            st.metric("Std Similarity", f"{stats['std_similarity']:.3f}")
        with col2:
            st.metric("Min Similarity", f"{stats['min_similarity']:.3f}")
            st.metric("Max Similarity", f"{stats['max_similarity']:.3f}")
        with col3:
            st.metric("Median Similarity", f"{stats['median_similarity']:.3f}")
            st.metric("75th Percentile", f"{stats['q75_similarity']:.3f}")
        with col4:
            # Distribution analysis
            high_sim_count = np.sum(similarity_matrix > 0.7)
            total_comparisons = similarity_matrix.shape[0] * similarity_matrix.shape[1]
            high_sim_pct = (high_sim_count / total_comparisons) * 100
            st.metric("High Similarity Pairs", f"{high_sim_pct:.1f}%")
        
        # Similarity distribution visualization
        st.markdown("### Similarity Distribution")
        try:
            import plotly.express as px
            import pandas as pd
            
            # Create histogram of similarities
            sim_flat = similarity_matrix.flatten()
            fig = px.histogram(
                x=sim_flat, 
                nbins=50,
                title="Distribution of Similarity Scores",
                labels={'x': 'Similarity Score', 'y': 'Frequency'}
            )
            fig.add_vline(x=stats['mean_similarity'], line_dash="dash", 
                         line_color="red", annotation_text="Mean")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("Install plotly for similarity distribution visualization")
        
        # Most influential texts
        if st.button("Show Most Influential Core Texts"):
            success, result, message = coordinator.find_most_influential_core_texts(15)
            if success:
                st.markdown("### Most Influential Core Texts")
                for text_info in result['influential_texts']:
                    with st.expander(f"#{text_info['rank']} - {text_info['document_title']} (Similarity: {text_info['average_similarity']:.3f})"):
                        st.write(f"**Author:** {text_info['document_author']}")
                        st.write(f"**File:** {text_info['filename']}")
                        st.write(f"**Paragraph:** {text_info['paragraph_in_doc'] + 1}")
                        st.write(f"**Average Similarity:** {text_info['average_similarity']:.4f}")

    def _render_vector_results(self, coordinator):
        """Render vector analysis results"""
        st.subheader("Custom Vector Analysis Results")
        
        existing_vectors = coordinator.custom_vector_manager.list_vectors()
        
        if not existing_vectors:
            st.info("No custom vectors created yet.")
            if st.button("Go to Vector Creation"):
                st.session_state.current_step = 'vectors'
                st.rerun()
            return
        
        # Vector summary
        st.markdown("### Vector Summary")
        vector_df_data = []
        for vector in existing_vectors:
            vector_info = coordinator.custom_vector_manager.get_vector_info(vector['name'])
            quality_score = vector_info.get('quality_score', 0.0) if vector_info else 0.0
            
            vector_df_data.append({
                'Vector Name': vector['name'],
                'Description': vector['description'] or 'No description',
                'Positive Terms': len(vector_info.get('positive_terms', [])) if vector_info else 0,
                'Negative Terms': len(vector_info.get('negative_terms', [])) if vector_info else 0,
                'Quality Score': f"{quality_score:.2f}",
                'Method': vector_info.get('method', 'Unknown') if vector_info else 'Unknown'
            })
        
        if vector_df_data:
            vector_df = pd.DataFrame(vector_df_data)
            st.dataframe(vector_df, use_container_width=True)
        
        # Vector projection analysis
        st.markdown("### Vector Projection Analysis")
        selected_vector = st.selectbox("Select vector for detailed analysis:", 
                                      [v['name'] for v in existing_vectors])
        
        col1, col2 = st.columns(2)
        with col1:
            analyze_target = st.button("Analyze Target Corpus Projections")
        with col2:
            analyze_core = st.button("Analyze Core Corpus Projections")
        
        if analyze_target or analyze_core:
            use_target = analyze_target
            corpus_name = "Target" if use_target else "Core"
            
            with st.spinner(f"Analyzing {corpus_name} corpus projections..."):
                success, results, message = coordinator.analyze_document_projections(
                    selected_vector, use_target=use_target
                )
            
            if success:
                st.success(f"{corpus_name} corpus analysis complete!")
                
                # Display projection statistics
                proj_results = results['projection_results']
                stats = proj_results['statistics']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Projection", f"{stats['mean_projection']:.3f}")
                with col2:
                    st.metric("Std Deviation", f"{stats['std_projection']:.3f}")
                with col3:
                    st.metric("Range", f"{stats['max_projection'] - stats['min_projection']:.3f}")
                with col4:
                    st.metric("Documents Analyzed", stats['total_documents'])
                
                # Extreme documents
                extremes = results['extreme_documents']
                if extremes:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Most Positive**")
                        for doc in extremes['most_positive'][:3]:
                            title = doc['document_metadata']['document_metadata']['title']
                            st.write(f"• {title[:40]}... ({doc['projection_score']:.3f})")
                    
                    with col2:
                        st.markdown("**Most Negative**")
                        for doc in extremes['most_negative'][:3]:
                            title = doc['document_metadata']['document_metadata']['title']
                            st.write(f"• {title[:40]}... ({doc['projection_score']:.3f})")
                    
                    with col3:
                        st.markdown("**Most Neutral**")
                        for doc in extremes['most_neutral'][:3]:
                            title = doc['document_metadata']['document_metadata']['title']
                            st.write(f"• {title[:40]}... ({doc['projection_score']:.3f})")
            else:
                st.error(message)

    def _render_topic_results(self, coordinator):
        """Render topic modeling results"""
        st.subheader("Topic Analysis Results")
        
        if 'topic_modeling_results' not in st.session_state:
            st.info("Topic modeling not yet performed.")
            if st.button("Go to Topic Modeling"):
                st.session_state.current_step = 'vectors'
                st.rerun()
            return
        
        results = st.session_state.topic_modeling_results
        
        # Topic overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Topics Found", len(results['topics_dataframe']))
        with col2:
            st.metric("Documents Processed", results['total_documents'])
        with col3:
            st.metric("Coherence Score", f"{results.get('coherence_score', 0):.3f}")
        with col4:
            st.metric("Outliers", results.get('outlier_count', 0))
        
        # Topic details table
        st.markdown("### Core Corpus Topics")
        topic_table_data = []
        for topic_id, label in results['topic_labels'].items():
            if topic_id != -1:  # Skip outlier topic
                topic_words = results['topic_words'].get(topic_id, [])
                top_words = [word for word, _ in topic_words[:5]]
                
                # Count documents in this topic
                doc_count = sum(1 for t in results['topics'] if t == topic_id)
                
                topic_table_data.append({
                    'Topic ID': topic_id,
                    'Label': label,
                    'Top Words': ', '.join(top_words),
                    'Document Count': doc_count,
                    'Percentage': f"{(doc_count / results['total_documents']) * 100:.1f}%"
                })
        
        if topic_table_data:
            topic_df = pd.DataFrame(topic_table_data)
            st.dataframe(topic_df, use_container_width=True)
        
        # Target corpus topic analysis
        if 'target_similarities' in st.session_state:
            st.markdown("### Target Corpus Topic Assignment")
            target_sims = st.session_state.target_similarities
            
            # Topic distribution in target corpus
            topic_dist = target_sims['topic_distribution']
            
            if topic_dist:
                st.markdown("**Topic Distribution in Target Corpus:**")
                dist_data = []
                for topic_id, count in topic_dist.items():
                    if topic_id == -1:
                        label = "Outlier (Novel Themes)"
                    else:
                        label = results['topic_labels'].get(topic_id, f"Topic {topic_id}")
                    
                    percentage = (count / target_sims['total_target_documents']) * 100
                    dist_data.append({
                        'Topic': label,
                        'Documents': count,
                        'Percentage': f"{percentage:.1f}%"
                    })
                
                dist_df = pd.DataFrame(dist_data)
                st.dataframe(dist_df, use_container_width=True)
            
            # Sample target documents by topic
            st.markdown("### Sample Target Documents by Topic")
            topic_samples = {}
            for analysis in target_sims['target_analysis'][:50]:  # Limit for performance
                topic_id = analysis['primary_topic']['id']
                topic_label = analysis['primary_topic']['label']
                
                if topic_label not in topic_samples:
                    topic_samples[topic_label] = []
                
                if len(topic_samples[topic_label]) < 3:  # Max 3 samples per topic
                    topic_samples[topic_label].append({
                        'title': analysis['target_document']['title'],
                        'paragraph': analysis['target_document']['paragraph_number'],
                        'preview': analysis['text_preview']
                    })
            
            for topic_label, samples in topic_samples.items():
                with st.expander(f"{topic_label} ({len(samples)} samples shown)"):
                    for sample in samples:
                        st.write(f"**{sample['title']}** (Paragraph {sample['paragraph']})")
                        st.write(f"*{sample['preview']}*")
                        st.write("---")

    def _render_network_results(self, coordinator):
        """Render network visualization results"""
        st.subheader("Network Analysis")
        
        if 'target_similarities' not in st.session_state:
            st.info("Target-topic analysis not yet performed.")
            return
        
        target_similarities = st.session_state.target_similarities
        
        # Show network if it exists
        if st.button("Generate Network Visualization"):
            with st.spinner("Creating network visualization..."):
                network_data = coordinator.create_topic_network_visualization(target_similarities)
            
            if network_data and 'figure' in network_data:
                st.plotly_chart(network_data['figure'], use_container_width=True)
                
                # Network statistics
                stats = network_data['network_stats']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Nodes", stats['nodes'])
                with col2:
                    st.metric("Connections", stats['edges'])
                with col3:
                    st.metric("Density", f"{stats['density']:.3f}")
                with col4:
                    st.metric("Components", stats['components'])
            
            elif network_data and 'error' in network_data:
                st.error(f"Network visualization error: {network_data['error']}")
        
        # Edge list preview
        if target_similarities.get('network_edges'):
            st.markdown("### Network Connections (Sample)")
            edge_sample = target_similarities['network_edges'][:20]  # Show first 20 edges
            
            edge_data = []
            for edge in edge_sample:
                edge_data.append({
                    'Source Document': edge['source'][:50] + "...",
                    'Target Document': edge['target'][:50] + "...",
                    'Similarity': f"{edge['weight']:.3f}",
                    'Topic': edge['topic']
                })
            
            if edge_data:
                edge_df = pd.DataFrame(edge_data)
                st.dataframe(edge_df, use_container_width=True)

    def _render_detailed_reports(self, coordinator):
        """Render detailed analysis reports"""
        st.subheader("Detailed Analysis Reports")
        
        # Document-level analysis report
        if st.button("Generate Document-Level Analysis Report"):
            with st.spinner("Generating detailed report..."):
                report_data = self._generate_document_report(coordinator)
            
            if report_data is not None and not report_data.empty:  # Check for empty DataFrame
                st.markdown("### Document-Level Analysis")
                st.dataframe(report_data, use_container_width=True)
            else:
                st.warning("No data available for document report")
        
        # Influence patterns report
        if st.button("Generate Influence Patterns Report"):
            with st.spinner("Analyzing influence patterns..."):
                influence_report = self._generate_influence_report(coordinator)
            
            if influence_report:  # This is a list, so regular boolean check is fine
                st.markdown("### Influence Patterns Analysis")
                for pattern in influence_report:
                    st.write(f"• {pattern}")
            else:
                st.warning("No influence patterns detected")

    def _generate_document_report(self, coordinator):
        """Generate comprehensive document-level analysis report"""
        try:
            report_data = []
            
            # Get all target documents with their various scores
            if 'target_corpus' not in st.session_state:
                return None
            
            target_corpus = st.session_state.target_corpus
            
            for filename, doc_data in target_corpus['documents'].items():
                for para_idx, paragraph in enumerate(doc_data['paragraphs']):
                    row = {
                        'Document': doc_data['metadata']['title'],
                        'Author': doc_data['metadata']['author'],
                        'Filename': filename,
                        'Paragraph': para_idx + 1,
                        'Word Count': len(paragraph.split())
                    }
                    
                    # Add similarity scores if available
                    if hasattr(st.session_state, 'similarity_matrix'):
                        # Calculate average similarity for this paragraph
                        target_idx = sum(len(d['paragraphs']) for d in list(target_corpus['documents'].values())[:list(target_corpus['documents'].keys()).index(filename)]) + para_idx
                        
                        if target_idx < st.session_state.similarity_matrix.shape[0]:
                            avg_similarity = np.mean(st.session_state.similarity_matrix[target_idx])
                            max_similarity = np.max(st.session_state.similarity_matrix[target_idx])
                            row['Avg Similarity'] = f"{avg_similarity:.3f}"
                            row['Max Similarity'] = f"{max_similarity:.3f}"
                    
                    # Add topic assignment if available
                    if 'target_similarities' in st.session_state:
                        target_sims = st.session_state.target_similarities
                        for analysis in target_sims['target_analysis']:
                            if (analysis['target_document']['filename'] == filename and
                                analysis['target_document']['paragraph_number'] == para_idx + 1):
                                row['Primary Topic'] = analysis['primary_topic']['label']
                                break
                    
                    report_data.append(row)
            
            return pd.DataFrame(report_data) if report_data else None
            
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            return None

    def _generate_influence_report(self, coordinator):
        """Generate influence patterns analysis"""
        patterns = []
        
        try:
            # Analyze similarity patterns
            if hasattr(st.session_state, 'similarity_matrix'):
                similarity_matrix = st.session_state.similarity_matrix
                stats = coordinator.similarity_engine.get_similarity_statistics(similarity_matrix)
                
                # Overall influence level
                if stats['mean_similarity'] > 0.6:
                    patterns.append("STRONG INFLUENCE: Target corpus shows high semantic alignment with core corpus")
                elif stats['mean_similarity'] > 0.4:
                    patterns.append("MODERATE INFLUENCE: Target corpus partially reflects core corpus themes")
                else:
                    patterns.append("WEAK INFLUENCE: Target corpus appears largely independent of core corpus")
                
                # Distribution patterns
                if stats['std_similarity'] > 0.2:
                    patterns.append("HIGH VARIABILITY: Some target texts highly influenced, others independent")
                else:
                    patterns.append("CONSISTENT INFLUENCE: Relatively uniform influence levels across target corpus")
            
            # Topic-based patterns
            if 'target_similarities' in st.session_state:
                target_sims = st.session_state.target_similarities
                outlier_pct = (target_sims['topic_distribution'].get(-1, 0) / target_sims['total_target_documents']) * 100
                
                if outlier_pct > 30:
                    patterns.append(f"HIGH NOVELTY: {outlier_pct:.1f}% of target content represents novel themes")
                elif outlier_pct > 10:
                    patterns.append(f"MODERATE NOVELTY: {outlier_pct:.1f}% of target content shows new themes")
                else:
                    patterns.append(f"LOW NOVELTY: Most target content aligns with core corpus themes")
            
            # Vector analysis patterns
            if coordinator.custom_vector_manager.custom_vectors:
                patterns.append(f"DIMENSIONAL ANALYSIS: {len(coordinator.custom_vector_manager.custom_vectors)} custom analytical dimensions created")
            
            return patterns
            
        except Exception as e:
            patterns.append(f"Error analyzing patterns: {str(e)}")
            return patterns

    def _render_export_section(self, coordinator):
        """Render comprehensive export section"""
        st.subheader("Export Analysis Data")
        
        # Export options
        export_options = st.multiselect(
            "Select data to export:",
            [
                "Similarity Matrix",
                "Document Metadata",
                "Custom Vectors",
                "Topic Analysis Results",
                "Network Edge List",
                "Network Node List",
                "Detailed Document Report",
                "Analysis Summary"
            ],
            default=["Analysis Summary", "Document Metadata"]
        )
        
        export_format = st.radio("Export format:", ["JSON", "CSV (where applicable)", "Excel"])
        
        # Generate and store export data in session state
        if st.button("Prepare Export Files"):
            with st.spinner("Preparing export data..."):
                export_data = self._prepare_export_data(coordinator, export_options)
                
                if export_data:
                    # Store in session state to persist across reruns
                    st.session_state.prepared_export_data = export_data
                    st.session_state.export_format = export_format
                    st.session_state.export_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.success("Export files prepared! Download buttons will appear below.")
                else:
                    st.error("Failed to prepare export data")
        
        # Display download buttons if export data exists
        if hasattr(st.session_state, 'prepared_export_data') and st.session_state.prepared_export_data:
            export_data = st.session_state.prepared_export_data
            export_format = st.session_state.export_format
            timestamp = st.session_state.export_timestamp
            
            st.markdown("### Download Files")
            
            if export_format == "JSON":
                # Single JSON file with all data
                json_data = json.dumps(export_data, indent=2, default=str)
                st.download_button(
                    label="📄 Download Complete Analysis (JSON)",
                    data=json_data,
                    file_name=f"corpus_analysis_{timestamp}.json",
                    mime="application/json",
                    key="json_download"
                )
            
            elif export_format == "CSV (where applicable)":
                # Individual CSV files for tabular data
                st.markdown("**Individual CSV Downloads:**")
                
                # Create columns for organized layout
                col1, col2 = st.columns(2)
                
                button_count = 0
                for data_type, data in export_data.items():
                    if isinstance(data, list) and data:
                        # Convert to DataFrame if possible
                        try:
                            df = pd.DataFrame(data)
                            csv_data = df.to_csv(index=False)
                            
                            # Alternate between columns
                            with col1 if button_count % 2 == 0 else col2:
                                st.download_button(
                                    label=f"📊 {data_type.replace('_', ' ').title()}",
                                    data=csv_data,
                                    file_name=f"{data_type.lower().replace(' ', '_')}_{timestamp}.csv",
                                    mime="text/csv",
                                    key=f"csv_{data_type}_{timestamp}"
                                )
                            button_count += 1
                        except Exception as e:
                            st.warning(f"Cannot convert {data_type} to CSV format: {str(e)}")
            
            elif export_format == "Excel":
                # Single Excel file with multiple sheets
                try:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        sheets_created = 0
                        for data_type, data in export_data.items():
                            try:
                                if isinstance(data, list) and data:
                                    df = pd.DataFrame(data)
                                    sheet_name = data_type.replace('_', ' ')[:30]  # Excel sheet name limit
                                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                                    sheets_created += 1
                                elif isinstance(data, dict) and data_type == "similarity_matrix":
                                    # Handle similarity matrix specially
                                    matrix_data = data.get('matrix', [])
                                    if matrix_data:
                                        df = pd.DataFrame(matrix_data)
                                        df.to_excel(writer, sheet_name="Similarity Matrix", index=False)
                                        sheets_created += 1
                            except Exception as e:
                                st.warning(f"Could not add {data_type} to Excel: {str(e)}")
                    
                    if sheets_created > 0:
                        excel_data = output.getvalue()
                        st.download_button(
                            label="📗 Download Complete Analysis (Excel)",
                            data=excel_data,
                            file_name=f"corpus_analysis_{timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="excel_download"
                        )
                    else:
                        st.error("No data could be converted to Excel format")
                        
                except Exception as e:
                    st.error(f"Error creating Excel file: {str(e)}")
            
            # Clear export data button
            if st.button("🗑️ Clear Prepared Files"):
                if hasattr(st.session_state, 'prepared_export_data'):
                    del st.session_state.prepared_export_data
                if hasattr(st.session_state, 'export_format'):
                    del st.session_state.export_format
                if hasattr(st.session_state, 'export_timestamp'):
                    del st.session_state.export_timestamp
                st.rerun()

    def _prepare_export_data(self, coordinator, export_options):
        """Prepare comprehensive export data package"""
        export_data = {}
        
        try:
            # Analysis Summary
            if "Analysis Summary" in export_options:
                summary = coordinator.get_analysis_summary()
                summary['export_timestamp'] = datetime.now().isoformat()
                summary['key_findings'] = self._generate_key_findings(coordinator)
                export_data['analysis_summary'] = summary
            
            # Similarity Matrix
            if "Similarity Matrix" in export_options and hasattr(st.session_state, 'similarity_matrix'):
                similarity_matrix = st.session_state.similarity_matrix
                
                # Convert to list format for JSON serialization
                export_data['similarity_matrix'] = {
                    'matrix': similarity_matrix.tolist(),
                    'shape': similarity_matrix.shape,
                    'statistics': coordinator.similarity_engine.get_similarity_statistics(similarity_matrix)
                }
            
            # Document Metadata
            if "Document Metadata" in export_options:
                doc_metadata = []
                
                # Core corpus metadata
                if coordinator.core_metadata:
                    for meta in coordinator.core_metadata['paragraph_metadata']:
                        doc_metadata.append({
                            'corpus_type': 'core',
                            'filename': meta['filename'],
                            'paragraph_index': meta['paragraph_index'],
                            'title': meta['document_metadata']['title'],
                            'author': meta['document_metadata']['author'],
                            'date': meta['document_metadata']['date'],
                            'source': meta['document_metadata']['source'],
                            'text_length': meta['text_length']
                        })
                
                # Target corpus metadata
                if coordinator.target_metadata:
                    for meta in coordinator.target_metadata['paragraph_metadata']:
                        doc_metadata.append({
                            'corpus_type': 'target',
                            'filename': meta['filename'],
                            'paragraph_index': meta['paragraph_index'],
                            'title': meta['document_metadata']['title'],
                            'author': meta['document_metadata']['author'],
                            'date': meta['document_metadata']['date'],
                            'source': meta['document_metadata']['source'],
                            'text_length': meta['text_length']
                        })
                
                export_data['document_metadata'] = doc_metadata
            
            # Custom Vectors
            if "Custom Vectors" in export_options:
                vectors_data = []
                for vector_name, vector_info in coordinator.custom_vector_manager.custom_vectors.items():
                    vector_export = {
                        'name': vector_name,
                        'description': vector_info['description'],
                        'positive_terms': vector_info['positive_terms'],
                        'negative_terms': vector_info['negative_terms'],
                        'method': vector_info['method'],
                        'created_at': vector_info['created_at'],
                        'quality_score': vector_info.get('quality_score', 0),
                        'vector_norm': vector_info['norm']
                    }
                    vectors_data.append(vector_export)
                
                export_data['custom_vectors'] = vectors_data
            
            # Topic Analysis Results
            if "Topic Analysis Results" in export_options and 'topic_modeling_results' in st.session_state:
                topic_results = st.session_state.topic_modeling_results
                
                topics_export = {
                    'total_documents': topic_results['total_documents'],
                    'coherence_score': topic_results.get('coherence_score', 0),
                    'topics': []
                }
                
                for topic_id, label in topic_results['topic_labels'].items():
                    if topic_id != -1:  # Skip outlier topic
                        topic_words = topic_results['topic_words'].get(topic_id, [])
                        doc_count = sum(1 for t in topic_results['topics'] if t == topic_id)
                        
                        topics_export['topics'].append({
                            'topic_id': topic_id,
                            'label': label,
                            'top_words': [word for word, score in topic_words[:10]],
                            'word_scores': topic_words[:10],
                            'document_count': doc_count,
                            'percentage': (doc_count / topic_results['total_documents']) * 100
                        })
                
                # Add target corpus topic assignments if available
                if 'target_similarities' in st.session_state:
                    target_sims = st.session_state.target_similarities
                    topics_export['target_analysis'] = {
                        'total_target_documents': target_sims['total_target_documents'],
                        'topic_distribution': target_sims['topic_distribution'],
                        'document_assignments': []
                    }
                    
                    # Add sample document assignments (limit for export size)
                    for analysis in target_sims['target_analysis'][:100]:  # Limit to first 100
                        doc_assignment = {
                            'document_title': analysis['target_document']['title'],
                            'document_author': analysis['target_document']['author'],
                            'filename': analysis['target_document']['filename'],
                            'paragraph_number': analysis['target_document']['paragraph_number'],
                            'primary_topic_id': analysis['primary_topic']['id'],
                            'primary_topic_label': analysis['primary_topic']['label'],
                            'topic_probabilities': analysis.get('topic_composition', {}),
                            'most_similar_core': analysis.get('most_similar_core'),
                            'text_preview': analysis['text_preview']
                        }
                        topics_export['target_analysis']['document_assignments'].append(doc_assignment)
                
                export_data['topic_analysis'] = topics_export
            
            # Network Edge List
            if "Network Edge List" in export_options and 'target_similarities' in st.session_state:
                target_sims = st.session_state.target_similarities
                if 'network_edges' in target_sims:
                    # Clean up edge data for export
                    clean_edges = []
                    for edge in target_sims['network_edges']:
                        clean_edges.append({
                            'source_document': edge['source'],
                            'target_document': edge['target'],
                            'similarity_weight': float(edge['weight']),
                            'shared_topic': edge['topic']
                        })
                    export_data['network_edges'] = clean_edges
            
            # Network Node List
            if "Network Node List" in export_options and 'target_similarities' in st.session_state:
                target_sims = st.session_state.target_similarities
                
                # Extract unique nodes from edges with metadata
                nodes_dict = {}
                if 'network_edges' in target_sims:
                    for edge in target_sims['network_edges']:
                        for node_key in ['source', 'target']:
                            node = edge[node_key]
                            if node not in nodes_dict:
                                node_type = 'target' if node.startswith('Target:') else 'core'
                                display_name = node.split(':', 1)[1] if ':' in node else node
                                
                                nodes_dict[node] = {
                                    'node_id': node,
                                    'node_type': node_type,
                                    'display_name': display_name,
                                    'degree': 0  # Will count connections
                                }
                            nodes_dict[node]['degree'] += 1
                
                export_data['network_nodes'] = list(nodes_dict.values())
            
            # Detailed Document Report
            if "Detailed Document Report" in export_options:
                report_data = self._generate_document_report(coordinator)
                if report_data is not None:
                    export_data['detailed_document_report'] = report_data.to_dict('records')
            
            # Influence Patterns
            if "Analysis Summary" in export_options:
                influence_patterns = self._generate_influence_report(coordinator)
                if 'analysis_summary' in export_data:
                    export_data['analysis_summary']['influence_patterns'] = influence_patterns
            
            return export_data
            
        except Exception as e:
            st.error(f"Error preparing export data: {str(e)}")
            return {}
    def _render_analysis_navigation_buttons(self):
        """Render navigation buttons to help users get to analysis steps"""
        st.markdown("### Quick Navigation")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📁 Upload Core Corpus"):
                st.session_state.current_step = 'core_corpus'
                st.rerun()
        
        with col2:
            if st.button("📁 Upload Target Corpus"):
                st.session_state.current_step = 'target_corpus'
                st.rerun()
        
        with col3:
            if st.button("🔬 Run Preprocessing/Analysis"):
                st.session_state.current_step = 'analysis'
                st.rerun()
        
        with col4:
            if st.button("🎯 Create Vectors"):
                st.session_state.current_step = 'vectors'
                st.rerun()