import streamlit as st
from datetime import datetime
from ..base import BaseComponent

class CorpusUploadComponent(BaseComponent):

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