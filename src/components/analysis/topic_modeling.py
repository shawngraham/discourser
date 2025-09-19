import streamlit as st
from ..base import BaseComponent

class TopicModelingComponent(BaseComponent):

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
                        "MaximalMarginalRelevance",
                        "KeyBERT",
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