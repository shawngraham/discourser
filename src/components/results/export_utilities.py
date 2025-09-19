import streamlit as st
from ..base import BaseComponent
import json
import io
import datetime

class ExportUtilitiesComponent(BaseComponent):

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