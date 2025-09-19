import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

def generate_key_findings(coordinator):
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

def generate_document_report(coordinator):
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

def generate_influence_report(coordinator):
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

def find_mixed_clusters(df_plot):
        """Find regions with both core and target documents (indicating influence zones)"""
        try:
            # Simple grid-based approach to find mixed regions
            mixed_regions = []
            
            # Create a grid and count core/target in each cell
            x_min, x_max = df_plot['x'].min(), df_plot['x'].max()
            y_min, y_max = df_plot['y'].min(), df_plot['y'].max()
            
            grid_size = 5  # 5x5 grid
            x_step = (x_max - x_min) / grid_size
            y_step = (y_max - y_min) / grid_size
            
            for i in range(grid_size):
                for j in range(grid_size):
                    x_start, x_end = x_min + i * x_step, x_min + (i + 1) * x_step
                    y_start, y_end = y_min + j * y_step, y_min + (j + 1) * y_step
                    
                    # Find documents in this grid cell
                    cell_docs = df_plot[
                        (df_plot['x'] >= x_start) & (df_plot['x'] < x_end) &
                        (df_plot['y'] >= y_start) & (df_plot['y'] < y_end)
                    ]
                    
                    if len(cell_docs) > 2:  # Need minimum documents
                        core_count = len(cell_docs[cell_docs['corpus'] == 'Core'])
                        target_count = len(cell_docs[cell_docs['corpus'] == 'Target'])
                        
                        # Mixed region if both core and target present
                        if core_count > 0 and target_count > 0:
                            mixed_regions.append({
                                'x_center': (x_start + x_end) / 2,
                                'y_center': (y_start + y_end) / 2,
                                'core_count': core_count,
                                'target_count': target_count,
                                'total_count': len(cell_docs)
                            })
            
            return mixed_regions
        except Exception:
            return []

def analyze_quadrant_distribution(df_plot):
        """Analyze distribution of core vs target documents across quadrants"""
        try:
            # Determine quadrants based on origin (0,0)
            quadrants = {
                'Q1 (+,+)': {'core_count': 0, 'target_count': 0},
                'Q2 (-,+)': {'core_count': 0, 'target_count': 0},
                'Q3 (-,-)': {'core_count': 0, 'target_count': 0},
                'Q4 (+,-)': {'core_count': 0, 'target_count': 0}
            }
            
            for _, row in df_plot.iterrows():
                x, y = row['x'], row['y']
                corpus_type = row['corpus']
                
                # Determine quadrant
                if x >= 0 and y >= 0:
                    quad = 'Q1 (+,+)'
                elif x < 0 and y >= 0:
                    quad = 'Q2 (-,+)'
                elif x < 0 and y < 0:
                    quad = 'Q3 (-,-)'
                else:  # x >= 0 and y < 0
                    quad = 'Q4 (+,-)'
                
                # Count by corpus type
                if corpus_type == 'Core':
                    quadrants[quad]['core_count'] += 1
                else:
                    quadrants[quad]['target_count'] += 1
            
            return quadrants
        except Exception:
            return None

def generate_pca_insights(pca_results: Dict, corpus_choice: str) -> List[str]:
        """Generate insights based on PCA results"""
        insights = []
        
        total_variance = pca_results['total_variance_explained']
        components = pca_results['components']
        
        # Overall structure insights
        if total_variance > 0.8:
            insights.append(f"Your {corpus_choice.lower()} has well-defined structure - the top {len(components)} components capture {total_variance:.1%} of variation")
        elif total_variance > 0.6:
            insights.append(f"Your {corpus_choice.lower()} has moderate structure - may benefit from more components for full representation")
        else:
            insights.append(f"Your {corpus_choice.lower()} is highly complex - variation is spread across many dimensions")
        
        # Component concentration insights
        if len(components) > 0:
            first_component_var = components[0]['explained_variance']
            if first_component_var > 0.3:
                insights.append(f"Strong primary conceptual dimension (PC1 explains {first_component_var:.1%})")
            elif first_component_var < 0.1:
                insights.append("No dominant conceptual dimension - corpus covers diverse themes")
        
        # Effective dimensionality
        effective_dims = len([comp for comp in components if comp['cumulative_variance'] < 0.9])
        if effective_dims <= 3:
            insights.append(f"Corpus can be effectively represented in {effective_dims} dimensions")
        elif effective_dims > 10:
            insights.append("High-dimensional corpus - many independent themes present")
        
        return insights

def generate_pca_recommendations(pca_results: Dict, coordinator) -> List[str]:
        """Generate recommendations based on PCA results"""
        recommendations = []
        
        components = pca_results['components']
        
        if len(components) >= 2:
            # Vector creation recommendations
            recommendations.append("Consider creating custom vectors aligned with your top principal components")
            
            first_var = components[0]['explained_variance']
            second_var = components[1]['explained_variance']
            
            if first_var / second_var > 3:
                recommendations.append("PC1 is much stronger than PC2 - focus vector creation on the primary dimension")
            else:
                recommendations.append("PC1 and PC2 are comparable - consider 2D vector spaces for analysis")
        
        # Analysis recommendations based on variance distribution
        total_variance = pca_results['total_variance_explained']
        if total_variance < 0.7:
            recommendations.append("Consider increasing the number of components or examining specific sub-themes")
        
        # Custom vector alignment recommendation
        if hasattr(coordinator, 'custom_vector_manager') and coordinator.custom_vector_manager.custom_vectors:
            recommendations.append("Compare your custom vectors to principal components to validate vector quality")
        else:
            recommendations.append("Use PCA insights to inform custom vector creation - align vectors with natural data structure")
        
        return recommendations

def generate_combined_corpus_insights(pca_results: Dict, coordinator) -> List[str]:
        """Generate insights specific to combined corpus analysis"""
        insights = []
        
        # Check if core/target metadata is available
        metadata_key = 'pca_metadata_combined'
        if metadata_key in st.session_state:
            metadata = st.session_state[metadata_key]
            core_count = len([m for m in metadata if m['corpus_type'] == 'Core'])
            target_count = len([m for m in metadata if m['corpus_type'] == 'Target'])
            
            insights.append(f"Analyzed {core_count} core documents and {target_count} target documents in unified space")
            
            # Variance concentration
            first_comp_var = pca_results['components'][0]['explained_variance']
            if first_comp_var > 0.4:
                insights.append("Strong primary conceptual dimension dominates both corpora")
            elif first_comp_var < 0.15:
                insights.append("Conceptual complexity is distributed across multiple dimensions")
            
            # Total variance captured
            total_var = pca_results['total_variance_explained']
            if total_var > 0.8:
                insights.append("The selected components capture most conceptual variation")
            else:
                insights.append("Additional components may be needed to fully represent the conceptual space")
        
        return insights

def generate_combined_corpus_recommendations(pca_results: Dict, coordinator) -> List[str]:
        """Generate recommendations for combined corpus analysis"""
        recommendations = []
        
        components = pca_results['components']
        
        if len(components) >= 2:
            first_var = components[0]['explained_variance']
            second_var = components[1]['explained_variance']
            
            if first_var > 0.3:
                recommendations.append("Focus custom vector creation on the primary dimension (PC1)")
            
            if second_var > 0.15:
                recommendations.append("Consider 2D vector spaces using PC1 and PC2 as axes")
        
        # Similarity analysis recommendations
        if hasattr(st.session_state, 'similarity_matrix'):
            recommendations.append("Cross-reference PCA clusters with similarity analysis results")
        else:
            recommendations.append("Run similarity analysis to complement the spatial relationship findings")
        
        # Vector creation recommendations
        if not hasattr(coordinator, 'custom_vector_manager') or not coordinator.custom_vector_manager.custom_vectors:
            recommendations.append("Create custom vectors that align with the discovered principal components")
        
        recommendations.append("Use the spatial separation insights to inform interpretation of influence patterns")
        
        return recommendations

def render_analysis_navigation_buttons():
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