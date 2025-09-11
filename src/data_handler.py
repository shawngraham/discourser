import pandas as pd
import streamlit as st
from pathlib import Path
import zipfile
import io
from typing import Dict, List, Tuple, Optional
import re

class DataHandler:
    """Handles all data input, validation, and preprocessing"""
    
    def __init__(self):
        self.required_columns = ['filename', 'title', 'author', 'date', 'source']
        self.optional_columns = ['document_type']
        
    def validate_csv_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate CSV has required columns"""
        errors = []
        
        # Check required columns
        missing_required = set(self.required_columns) - set(df.columns)
        if missing_required:
            errors.append(f"Missing required columns: {', '.join(missing_required)}")
        
        # Check for empty required fields
        for col in self.required_columns:
            if col in df.columns:
                if df[col].isna().any():
                    errors.append(f"Column '{col}' contains empty values")
        
        # Validate date format
        if 'date' in df.columns:
            try:
                pd.to_datetime(df['date'])
            except:
                errors.append("Date column contains invalid date formats. Use YYYY-MM-DD format.")
        
        # Check for duplicate filenames
        if 'filename' in df.columns:
            if df['filename'].duplicated().any():
                errors.append("Duplicate filenames found in CSV")
        
        return len(errors) == 0, errors
    
    def validate_file_correspondence(self, df: pd.DataFrame, uploaded_files: Dict) -> Tuple[bool, List[str]]:
        """Validate that all CSV filenames have corresponding uploaded files"""
        errors = []
        
        if 'filename' not in df.columns:
            return False, ["CSV missing filename column"]
        
        csv_filenames = set(df['filename'].tolist())
        uploaded_filenames = set(uploaded_files.keys())
        
        # Check for missing files
        missing_files = csv_filenames - uploaded_filenames
        if missing_files:
            errors.append(f"Files referenced in CSV but not uploaded: {', '.join(missing_files)}")
        
        # Check for extra files
        extra_files = uploaded_filenames - csv_filenames
        if extra_files:
            errors.append(f"Uploaded files not referenced in CSV: {', '.join(extra_files)}")
        
        return len(errors) == 0, errors
    
    def process_text_file(self, file_content: str, filename: str) -> Tuple[bool, List[str], List[str]]:
        """Process a single text file and extract paragraphs"""
        errors = []
        
        try:
            # Check encoding
            if not file_content.strip():
                errors.append(f"File {filename} is empty")
                return False, errors, []
            
            # Split by double blank lines to get paragraphs
            paragraphs = re.split(r'\n\s*\n\s*\n', file_content.strip())
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            if not paragraphs:
                errors.append(f"No paragraphs found in {filename}. Ensure paragraphs are separated by double blank lines.")
                return False, errors, []
            
            return True, [], paragraphs
            
        except Exception as e:
            errors.append(f"Error processing {filename}: {str(e)}")
            return False, errors, []
    
    def process_corpus(self, csv_file, text_files: Dict) -> Tuple[bool, Dict, List[str]]:
        """Process entire corpus upload"""
        errors = []
        
        try:
            # Read and validate CSV
            df = pd.read_csv(csv_file)
            csv_valid, csv_errors = self.validate_csv_schema(df)
            if not csv_valid:
                errors.extend(csv_errors)
                return False, {}, errors
            
            # Validate file correspondence
            files_valid, file_errors = self.validate_file_correspondence(df, text_files)
            if not files_valid:
                errors.extend(file_errors)
                return False, {}, errors
            
            # Process each text file
            processed_data = {
                'metadata': df,
                'documents': {},
                'total_paragraphs': 0
            }
            
            for _, row in df.iterrows():
                filename = row['filename']
                if filename in text_files:
                    # Decode file content
                    file_content = text_files[filename].decode('utf-8')
                    
                    success, file_errors, paragraphs = self.process_text_file(file_content, filename)
                    if not success:
                        errors.extend(file_errors)
                        continue
                    
                    processed_data['documents'][filename] = {
                        'metadata': row.to_dict(),
                        'paragraphs': paragraphs,
                        'paragraph_count': len(paragraphs)
                    }
                    processed_data['total_paragraphs'] += len(paragraphs)
            
            if errors:
                return False, {}, errors
            
            return True, processed_data, []
            
        except Exception as e:
            errors.append(f"Unexpected error processing corpus: {str(e)}")
            return False, {}, errors
