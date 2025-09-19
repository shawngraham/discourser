import streamlit as st
from ..base import BaseComponent
from .topic_modeling import TopicModelingComponent
import numpy as np
import pandas as pd  

class VectorAnalysisComponent(BaseComponent):

    def __init__(self):
        super().__init__()
        self.topic_modeling = TopicModelingComponent()


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
        
        # Main tabs for vector analysis including new PCA Analysis tab
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Suggested Terms", 
            "🎯 Create Vectors", 
            "📈 Vector Analysis", 
            "🗺️ Vector Spaces", 
            "🔍 Topic Modeling",
            "🧮 PCA Analysis"  
        ])
        
        with tab1:
            self._render_suggested_terms_tab(coordinator)
        
        with tab2:
            self._render_create_vectors_tab(coordinator)
        
        with tab3:
            self._render_vector_analysis_tab(coordinator)
        
        with tab4:
            self._render_vector_spaces_tab(coordinator)
        
        with tab5:
            self.topic_modeling._render_topic_modeling_tab(coordinator)

        with tab6:
            self._render_pca_analysis_tab(coordinator)

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

    def _render_pca_analysis_tab(self, coordinator):
        """Render the PCA analysis tab with embedding space analysis"""
        st.subheader("🧮 Principal Component Analysis of Embedding Space")
        
        st.markdown("""
        Explore the mathematical structure of your embedding space using Principal Component Analysis (PCA).
        This helps you understand which dimensions capture the most variation in your corpus.
        """)
        
        # PCA Configuration
        with st.expander("PCA Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                corpus_choice = st.radio(
                    "Analyze which corpus:",
                    ["Core Corpus", "Target Corpus", "Combined View"],
                    help="Choose which corpus embeddings to analyze. Combined View shows both corpora in the same space."
                )
                
                n_components = st.slider(
                    "Number of principal components:",
                    min_value=2,
                    max_value=min(50, len(coordinator.core_embeddings) if coordinator.core_embeddings is not None else 50),
                    value=10,
                    help="How many principal components to compute and display"
                )
            
            with col2:
                st.markdown("### What is PCA?")
                st.markdown("""
                **Principal Component Analysis** finds the directions of maximum variance in your data.
                
                - **Component 1**: Direction with most variation
                - **Component 2**: Second most variation (orthogonal to first)
                - **Explained Variance**: How much of the total variation each component captures
                
                This helps identify the most important conceptual dimensions in your corpus.
                """)
        
        # Run PCA Analysis
        use_target = corpus_choice == "Target Corpus"
        use_combined = corpus_choice == "Combined View"
        
        if use_combined:
            corpus_name = "combined"
        else:
            corpus_name = "target" if use_target else "core"
        
        if st.button(f"🔬 Analyze {corpus_choice} Embedding Space", type="primary"):
            with st.spinner(f"Running PCA analysis on {corpus_choice.lower()}..."):
                
                # Get embeddings to analyze based on selection
                if use_combined:
                    # Check both embeddings are available
                    if coordinator.target_embeddings is None:
                        st.error("Target corpus not processed yet. Both corpora needed for combined view.")
                        return
                    
                    # Combine embeddings
                    combined_embeddings = np.vstack([coordinator.core_embeddings, coordinator.target_embeddings])
                    
                    # Create combined metadata
                    core_metadata = coordinator.core_metadata['paragraph_metadata']
                    target_metadata = coordinator.target_metadata['paragraph_metadata']
                    
                    # Add corpus type to metadata
                    combined_metadata = []
                    for meta in core_metadata:
                        meta_copy = meta.copy()
                        meta_copy['corpus_type'] = 'Core'
                        combined_metadata.append(meta_copy)
                    
                    for meta in target_metadata:
                        meta_copy = meta.copy()
                        meta_copy['corpus_type'] = 'Target'
                        combined_metadata.append(meta_copy)
                    
                    embeddings = combined_embeddings
                    metadata = combined_metadata
                    
                elif use_target:
                    if coordinator.target_embeddings is None:
                        st.error("Target corpus not processed yet.")
                        return
                    embeddings = coordinator.target_embeddings
                    metadata = coordinator.target_metadata['paragraph_metadata']
                else:
                    embeddings = coordinator.core_embeddings
                    metadata = coordinator.core_metadata['paragraph_metadata']
                
                # Perform PCA analysis
                pca_results = coordinator.vector_analysis_engine.analyze_embedding_space(
                    embeddings, n_components=n_components
                )
                
                if pca_results:
                    # Store results in session state
                    st.session_state[f'pca_results_{corpus_name}'] = pca_results
                    st.session_state[f'pca_metadata_{corpus_name}'] = metadata
                    
                    # Store additional info for combined view
                    if use_combined:
                        st.session_state[f'pca_corpus_split_{corpus_name}'] = {
                            'core_count': len(coordinator.core_embeddings),
                            'target_count': len(coordinator.target_embeddings)
                        }
                    
                    st.success(f"✅ PCA analysis complete for {corpus_choice}!")
                else:
                    st.error("❌ PCA analysis failed. Please try again.")
        
        # Display results if they exist
        results_key = f'pca_results_{corpus_name}'
        metadata_key = f'pca_metadata_{corpus_name}'
        
        if results_key in st.session_state and st.session_state[results_key]:
            pca_results = st.session_state[results_key]
            metadata = st.session_state[metadata_key]
            
            st.markdown("---")
            st.markdown(f"### 📈 PCA Results for {corpus_choice}")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original Dimensions", pca_results['original_dimensions'])
            with col2:
                st.metric("Components Analyzed", pca_results['reduced_dimensions'])
            with col3:
                st.metric("Total Variance Explained", f"{pca_results['total_variance_explained']:.1%}")
            with col4:
                # Calculate effective dimensionality (components needed for 90% variance)
                cumulative_var = [comp['cumulative_variance'] for comp in pca_results['components']]
                effective_dims = len([v for v in cumulative_var if v < 0.9]) + 1
                st.metric("Effective Dimensions (90%)", effective_dims)
            
            # Variance explained visualization
            st.markdown("### 📊 Variance Explained by Components")
            
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                
                # Create DataFrame for plotting
                components_data = []
                for i, comp in enumerate(pca_results['components']):
                    components_data.append({
                        'Component': f'PC{i+1}',
                        'Component_Number': i+1,
                        'Individual_Variance': comp['explained_variance'] * 100,
                        'Cumulative_Variance': comp['cumulative_variance'] * 100
                    })
                
                df_components = pd.DataFrame(components_data)
                
                # Create subplot with secondary y-axis
                fig = go.Figure()
                
                # Bar chart for individual variance
                fig.add_trace(go.Bar(
                    x=df_components['Component'],
                    y=df_components['Individual_Variance'],
                    name='Individual Variance',
                    marker_color='lightblue',
                    yaxis='y1'
                ))
                
                # Line chart for cumulative variance
                fig.add_trace(go.Scatter(
                    x=df_components['Component'],
                    y=df_components['Cumulative_Variance'],
                    name='Cumulative Variance',
                    line=dict(color='red', width=3),
                    marker=dict(size=8),
                    yaxis='y2'
                ))
                
                # Add 90% variance line
                fig.add_hline(y=90, line_dash="dash", line_color="green", 
                             annotation_text="90% Variance Threshold")
                
                # Update layout
                fig.update_layout(
                    title="Principal Components: Individual vs Cumulative Variance Explained",
                    xaxis=dict(title="Principal Component"),
                    yaxis=dict(title="Individual Variance (%)", side="left"),
                    yaxis2=dict(title="Cumulative Variance (%)", side="right", overlaying="y"),
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except ImportError:
                st.warning("Install plotly for interactive visualizations: `pip install plotly`")
                
                # Fallback table display
                st.markdown("**Component Variance Table:**")
                variance_data = []
                for i, comp in enumerate(pca_results['components']):
                    variance_data.append({
                        'Component': f'PC{i+1}',
                        'Individual Variance': f"{comp['explained_variance']:.1%}",
                        'Cumulative Variance': f"{comp['cumulative_variance']:.1%}"
                    })
                
                st.dataframe(pd.DataFrame(variance_data), use_container_width=True)
            
            # Component analysis
            st.markdown("### 🔍 Component Analysis")
            
            # Select component to examine
            selected_component = st.selectbox(
                "Select component to examine in detail:",
                options=range(len(pca_results['components'])),
                format_func=lambda x: f"PC{x+1} ({pca_results['components'][x]['explained_variance']:.1%} variance)"
            )
            
            if selected_component is not None:
                comp_info = pca_results['components'][selected_component]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"#### Principal Component {selected_component + 1}")
                    st.write(f"**Individual Variance Explained:** {comp_info['explained_variance']:.1%}")
                    st.write(f"**Cumulative Variance Explained:** {comp_info['cumulative_variance']:.1%}")
                    
                    # Show component vector visualization if possible
                    component_vector = np.array(comp_info['component_vector'])
                    
                    # Find the strongest dimensions in this component
                    abs_weights = np.abs(component_vector)
                    top_dimensions = np.argsort(abs_weights)[-20:][::-1]  # Top 20 dimensions
                    
                    st.markdown("**Strongest Dimensions in this Component:**")
                    
                    dim_data = []
                    for dim_idx in top_dimensions:
                        dim_data.append({
                            'Dimension': f'Dim_{dim_idx}',
                            'Weight': f"{component_vector[dim_idx]:.4f}",
                            'Absolute Weight': f"{abs_weights[dim_idx]:.4f}"
                        })
                    
                    dim_df = pd.DataFrame(dim_data)
                    st.dataframe(dim_df, use_container_width=True)
                
                with col2:
                    st.markdown("#### Interpretation Guide")
                    
                    variance_pct = comp_info['explained_variance']
                    
                    if variance_pct > 0.1:
                        st.success("🟢 **High Importance**")
                        st.write("This component captures major variation in your corpus.")
                    elif variance_pct > 0.05:
                        st.info("🟡 **Moderate Importance**")
                        st.write("This component captures meaningful but secondary variation.")
                    else:
                        st.warning("🔴 **Low Importance**")
                        st.write("This component captures minor variation - may represent noise.")
                    
                    st.markdown("**What this means:**")
                    st.write(f"• Explains {variance_pct:.1%} of total variation")
                    st.write(f"• Ranking: {selected_component + 1} out of {len(pca_results['components'])}")
                    
                    if selected_component == 0:
                        st.write("• **Primary conceptual dimension** in your corpus")
                    elif selected_component == 1:
                        st.write("• **Secondary conceptual dimension** in your corpus")
                    else:
                        st.write("• Captures more specialized patterns")
            
            # Document projections onto components
            st.markdown("### 📍 Document Positions in Principal Component Space")
            
            if 'pca_embeddings' in pca_results:
                pca_embeddings = pca_results['pca_embeddings']
                
                # Allow user to select which components to visualize
                col1, col2 = st.columns(2)
                with col1:
                    pc_x = st.selectbox("X-axis (PC):", range(len(pca_results['components'])), 
                                       format_func=lambda x: f"PC{x+1}")
                with col2:
                    pc_y = st.selectbox("Y-axis (PC):", range(len(pca_results['components'])), 
                                       index=1 if len(pca_results['components']) > 1 else 0,
                                       format_func=lambda x: f"PC{x+1}")
                
                if pc_x != pc_y and st.button("🗺️ Create PCA Scatter Plot"):
                    try:
                        import plotly.express as px
                        
                        # Check if this is a combined view
                        is_combined = corpus_name == "combined"
                        
                        # Prepare data for plotting
                        plot_data = []
                        for i, meta in enumerate(metadata):
                            if i < len(pca_embeddings):
                                data_point = {
                                    'x': pca_embeddings[i, pc_x],
                                    'y': pca_embeddings[i, pc_y],
                                    'title': meta['document_metadata']['title'][:50] + "...",
                                    'author': meta['document_metadata']['author'],
                                    'filename': meta['filename'],
                                    'paragraph': meta['paragraph_index'] + 1
                                }
                                
                                # Add corpus type for combined view
                                if is_combined:
                                    data_point['corpus'] = meta['corpus_type']
                                    data_point['hover_info'] = f"{meta['corpus_type']}: {meta['document_metadata']['title'][:40]}... (Para {meta['paragraph_index'] + 1})"
                                else:
                                    data_point['corpus'] = corpus_choice.replace(" Corpus", "")
                                    data_point['hover_info'] = f"{meta['document_metadata']['title'][:40]}... (Para {meta['paragraph_index'] + 1})"
                                
                                plot_data.append(data_point)
                        
                        df_plot = pd.DataFrame(plot_data)
                        
                        # Create scatter plot with proper color coding
                        if is_combined:
                            # Combined view with distinct colors for each corpus
                            fig = px.scatter(
                                df_plot,
                                x='x',
                                y='y',
                                color='corpus',
                                hover_data=['title', 'author', 'filename', 'paragraph'],
                                title=f"Combined Corpus View: PC{pc_x+1} vs PC{pc_y+1}",
                                labels={
                                    'x': f'PC{pc_x+1} ({pca_results["components"][pc_x]["explained_variance"]:.1%} variance)',
                                    'y': f'PC{pc_y+1} ({pca_results["components"][pc_y]["explained_variance"]:.1%} variance)',
                                    'corpus': 'Corpus Type'
                                },
                                color_discrete_map={
                                    'Core': '#FF6B6B',      # Distinct red for core
                                    'Target': '#4ECDC4'     # Distinct teal for target
                                }
                            )
                            
                            # Add analysis insights for combined view
                            st.markdown("### 📊 Combined Corpus Analysis")
                            
                            # Calculate clustering metrics
                            core_points = df_plot[df_plot['corpus'] == 'Core'][['x', 'y']].values
                            target_points = df_plot[df_plot['corpus'] == 'Target'][['x', 'y']].values
                            
                            if len(core_points) > 0 and len(target_points) > 0:
                                # Calculate average distance between core and target centroids
                                core_centroid = np.mean(core_points, axis=0)
                                target_centroid = np.mean(target_points, axis=0)
                                centroid_distance = np.linalg.norm(core_centroid - target_centroid)
                                
                                # Calculate overlap using convex hull or simple bounding box
                                core_spread = np.std(core_points, axis=0).mean()
                                target_spread = np.std(target_points, axis=0).mean()
                                
                                # Display clustering insights
                                col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                                with col_c1:
                                    st.metric("Centroid Distance", f"{centroid_distance:.3f}")
                                with col_c2:
                                    st.metric("Core Spread", f"{core_spread:.3f}")
                                with col_c3:
                                    st.metric("Target Spread", f"{target_spread:.3f}")
                                with col_c4:
                                    if max(core_spread, target_spread) > 0:
                                        overlap_ratio = min(core_spread, target_spread) / max(core_spread, target_spread)
                                    else:
                                        overlap_ratio = 0
                                    st.metric("Relative Overlap", f"{overlap_ratio:.3f}")
                                
                                # Provide interpretation
                                st.markdown("#### Spatial Relationship Analysis")
                                if centroid_distance < 1.0:
                                    st.success("🔗 **Strong Spatial Alignment**: Core and target corpora occupy similar conceptual space")
                                elif centroid_distance < 2.0:
                                    st.info("🔄 **Moderate Alignment**: Some conceptual overlap with distinct regions")
                                else:
                                    st.warning("📍 **Distinct Territories**: Core and target occupy different conceptual spaces")
                                
                                # Find mixed clusters - areas with both core and target documents
                                mixed_regions = self._find_mixed_clusters(df_plot)
                                if mixed_regions:
                                    st.markdown("#### 🎯 High Influence Regions")
                                    st.write(f"Found {len(mixed_regions)} regions with mixed core-target content, indicating direct conceptual influence.")
                                
                        else:
                            # Single corpus view
                            fig = px.scatter(
                                df_plot,
                                x='x',
                                y='y',
                                hover_data=['title', 'author', 'filename', 'paragraph'],
                                title=f"Documents in PC{pc_x+1} vs PC{pc_y+1} Space",
                                labels={
                                    'x': f'PC{pc_x+1} ({pca_results["components"][pc_x]["explained_variance"]:.1%} variance)',
                                    'y': f'PC{pc_y+1} ({pca_results["components"][pc_y]["explained_variance"]:.1%} variance)'
                                }
                            )
                        
                        # Add origin lines
                        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
                        
                        # Enhanced layout for combined view
                        if is_combined:
                            fig.update_layout(
                                height=700,
                                legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.01,
                                    bgcolor="rgba(255,255,255,0.8)"
                                )
                            )
                            
                            # Add quadrant analysis for combined view
                            quadrant_analysis = self._analyze_quadrant_distribution(df_plot)
                            if quadrant_analysis:
                                with st.expander("📊 Quadrant Distribution Analysis", expanded=False):
                                    st.write("**Distribution of Core vs Target documents across PCA quadrants:**")
                                    
                                    quad_cols = st.columns(4)
                                    for i, (quad_name, quad_data) in enumerate(quadrant_analysis.items()):
                                        with quad_cols[i]:
                                            st.write(f"**{quad_name}**")
                                            st.write(f"Core: {quad_data['core_count']}")
                                            st.write(f"Target: {quad_data['target_count']}")
                                            
                                            if quad_data['core_count'] > 0 and quad_data['target_count'] > 0:
                                                st.success("Mixed")
                                            elif quad_data['core_count'] > quad_data['target_count']:
                                                st.info("Core-dominated")
                                            else:
                                                st.warning("Target-dominated")
                        else:
                            fig.update_layout(height=600)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except ImportError:
                        st.warning("Install plotly for PCA scatter plots")
            
            # Show insights and recommendations
            if corpus_name == "combined":
                # Use the combined corpus insights methods
                insights = self._generate_combined_corpus_insights(pca_results, coordinator)
                recommendations = self._generate_combined_corpus_recommendations(pca_results, coordinator)
            else:
                # Use regular PCA insights methods
                insights = self._generate_pca_insights(pca_results, corpus_choice)
                recommendations = self._generate_pca_recommendations(pca_results, coordinator)
            
            if insights:
                st.markdown("### 💡 Key Insights")
                for insight in insights:
                    st.write(f"• {insight}")
            
            if recommendations:
                st.markdown("### 📋 Recommendations")
                for rec in recommendations:
                    st.write(f"• {rec}")