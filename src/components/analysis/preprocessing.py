import streamlit as st
import re
from ..base import BaseComponent

class PreprocessingComponent(BaseComponent):

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