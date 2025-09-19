import streamlit as st
from ..base import BaseComponent
from ..results.export_utilities import ExportUtilitiesComponent
import numpy as np
import pandas as pd
import datetime

class ComprehensiveResultsComponent(BaseComponent):
    def __init__(self):
        super().__init__()
        
        self.export_utilities = ExportUtilitiesComponent()  # Add this
   

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
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📈 Similarity Analysis", 
            "🎯 Vector Analysis", 
            "📝 Topic Analysis", 
            "🌐 Network View",
            "🧮 PCA Analysis",
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
            self._render_pca_results(coordinator)
        with tab6:
            self._render_detailed_reports(coordinator)
        
        with tab7:
            self.export_utilities._render_export_section(coordinator)

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

    def _render_pca_results(self, coordinator):
        """Render PCA analysis results (display only, no analysis execution)"""
        st.subheader("🧮 PCA Analysis Results")
        
        # Check if any PCA results exist
        pca_result_keys = [key for key in st.session_state.keys() if key.startswith('pca_results_')]
        
        if not pca_result_keys:
            st.info("No PCA analysis results available. Run PCA analysis in the Vector Analysis section first.")
            if st.button("Go to Vector Analysis"):
                st.session_state.current_step = 'vectors'
                st.rerun()
            return
        
        # Display results for each corpus analyzed
        for result_key in pca_result_keys:
            corpus_name = result_key.replace('pca_results_', '')
            pca_results = st.session_state[result_key]
            metadata_key = f'pca_metadata_{corpus_name}'
            
            if metadata_key in st.session_state:
                metadata = st.session_state[metadata_key]
                
                # Display corpus name
                corpus_display = corpus_name.replace('_', ' ').title()
                st.markdown(f"### 📈 {corpus_display} Corpus Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Original Dimensions", pca_results['original_dimensions'])
                with col2:
                    st.metric("Components Analyzed", pca_results['reduced_dimensions'])
                with col3:
                    st.metric("Total Variance Explained", f"{pca_results['total_variance_explained']:.1%}")
                with col4:
                    cumulative_var = [comp['cumulative_variance'] for comp in pca_results['components']]
                    effective_dims = len([v for v in cumulative_var if v < 0.9]) + 1
                    st.metric("Effective Dimensions (90%)", effective_dims)
                
                # Show insights and recommendations
                if corpus_name == "combined":
                    insights = self._generate_combined_corpus_insights(pca_results, coordinator)
                    recommendations = self._generate_combined_corpus_recommendations(pca_results, coordinator)
                else:
                    corpus_choice = corpus_name.replace('_', ' ').title() + " Corpus"
                    insights = self._generate_pca_insights(pca_results, corpus_choice)
                    recommendations = self._generate_pca_recommendations(pca_results, coordinator)
                
                if insights:
                    with st.expander("Key Insights"):
                        for insight in insights:
                            st.write(f"• {insight}")
                
                if recommendations:
                    with st.expander("Recommendations"):
                        for rec in recommendations:
                            st.write(f"• {rec}")
                
                st.markdown("---")