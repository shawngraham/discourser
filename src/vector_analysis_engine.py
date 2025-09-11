import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from collections import Counter
import re
from typing import Dict, List, Tuple, Optional, Union
import logging
from bertopic import BERTopic

logger = logging.getLogger(__name__)

class VectorAnalysisEngine:
    """Handles automatic vector extraction and corpus analysis for term suggestions"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.pca_model = None
        self.topic_model = None
        self.corpus_vocabulary = None
        
    def extract_corpus_terms(self, corpus_data: Dict, max_features: int = 1000) -> Dict:
        """Extract important terms from corpus using multiple methods"""
        try:
            # Extract all text
            all_texts = []
            for filename, doc_data in corpus_data['documents'].items():
                all_texts.extend(doc_data['paragraphs'])
            
            if not all_texts:
                return {}
            
            # TF-IDF analysis
            tfidf_terms = self._extract_tfidf_terms(all_texts, max_features)
            
            # Frequency analysis
            frequency_terms = self._extract_frequency_terms(all_texts, max_features)
            
            # Topic modeling for term suggestions
            topic_terms = self._extract_topic_terms(all_texts, n_topics=10)
            
            # Combine results
            suggested_terms = {
                'tfidf_terms': tfidf_terms,
                'frequency_terms': frequency_terms,
                'topic_terms': topic_terms,
                'corpus_stats': {
                    'total_documents': len(corpus_data['documents']),
                    'total_paragraphs': len(all_texts),
                    'avg_paragraph_length': np.mean([len(text.split()) for text in all_texts])
                }
            }
            
            logger.info(f"Extracted terms from corpus: {len(tfidf_terms)} TF-IDF, {len(frequency_terms)} frequency, {len(topic_terms)} topic terms")
            return suggested_terms
            
        except Exception as e:
            logger.error(f"Error extracting corpus terms: {str(e)}")
            return {}
    
    def _extract_tfidf_terms(self, texts: List[str], max_features: int) -> List[Dict]:
        """Extract important terms using TF-IDF"""
        try:
            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
                min_df=2,  # Must appear in at least 2 documents
                max_df=0.8  # Don't include terms in more than 80% of documents
            )
            
            # Fit and transform
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get mean TF-IDF scores
            mean_scores = np.mean(self.tfidf_matrix.toarray(), axis=0)
            
            # Create sorted list of terms with scores
            tfidf_terms = []
            for idx, score in enumerate(mean_scores):
                tfidf_terms.append({
                    'term': feature_names[idx],
                    'score': float(score),
                    'type': 'tfidf'
                })
            
            # Sort by score descending
            tfidf_terms.sort(key=lambda x: x['score'], reverse=True)
            
            # Keep vocabulary for later use
            self.corpus_vocabulary = set(feature_names)
            
            return tfidf_terms[:50]  # Return top 50
            
        except Exception as e:
            logger.error(f"Error in TF-IDF extraction: {str(e)}")
            return []
    
    def _extract_frequency_terms(self, texts: List[str], max_features: int) -> List[Dict]:
        """Extract frequent terms"""
        try:
            # Combine all texts and tokenize
            all_text = ' '.join(texts).lower()
            
            # Simple tokenization (improve this for production)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
            
            # Remove common stop words
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'between'}
            filtered_words = [word for word in words if word not in stop_words]
            
            # Count frequencies
            word_counts = Counter(filtered_words)
            
            # Convert to format consistent with other methods
            frequency_terms = []
            for word, count in word_counts.most_common(50):
                frequency_terms.append({
                    'term': word,
                    'score': count,
                    'type': 'frequency'
                })
            
            return frequency_terms
            
        except Exception as e:
            logger.error(f"Error in frequency extraction: {str(e)}")
            return []
    
    def _extract_topic_terms(self, texts: List[str], n_topics: int = 10) -> List[Dict]:
        """Extract terms from topic modeling"""
        try:
            # Use BERTopic for topic modeling
            self.topic_model = BERTopic(
                nr_topics=n_topics,
                verbose=False,
                calculate_probabilities=False  # Faster
            )
            
            # Fit the model
            topics, _ = self.topic_model.fit_transform(texts)
            
            # Extract topic terms
            topic_terms = []
            topic_info = self.topic_model.get_topic_info()
            
            for topic_id in range(min(n_topics, len(topic_info))):
                if topic_id == -1:  # Skip outlier topic
                    continue
                    
                topic_words = self.topic_model.get_topic(topic_id)
                if topic_words:
                    for word, score in topic_words[:5]:  # Top 5 words per topic
                        topic_terms.append({
                            'term': word,
                            'score': float(score),
                            'type': f'topic_{topic_id}',
                            'topic_id': topic_id
                        })
            
            return topic_terms
            
        except Exception as e:
            logger.error(f"Error in topic modeling: {str(e)}")
            return []
    
    def analyze_embedding_space(self, embeddings: np.ndarray, n_components: int = 10) -> Dict:
        """Analyze embedding space and extract principal components"""
        try:
            # Fit PCA
            self.pca_model = PCA(n_components=n_components, random_state=42)
            pca_embeddings = self.pca_model.fit_transform(embeddings)
            
            # Calculate component statistics
            components_info = []
            for i, (component, variance) in enumerate(zip(self.pca_model.components_, self.pca_model.explained_variance_ratio_)):
                components_info.append({
                    'component_id': i,
                    'explained_variance': float(variance),
                    'cumulative_variance': float(np.sum(self.pca_model.explained_variance_ratio_[:i+1])),
                    'component_vector': component.tolist()
                })
            
            analysis_results = {
                'components': components_info,
                'total_variance_explained': float(np.sum(self.pca_model.explained_variance_ratio_)),
                'pca_embeddings': pca_embeddings,
                'original_dimensions': embeddings.shape[1],
                'reduced_dimensions': n_components
            }
            
            logger.info(f"PCA analysis complete: {n_components} components explain {analysis_results['total_variance_explained']:.3f} of variance")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in embedding space analysis: {str(e)}")
            return {}
    
    def run_enhanced_topic_modeling(self, corpus_data: Dict, config: Dict) -> Tuple[bool, Dict, str]:
        """Run topic modeling with enhanced configuration"""
        try:
            from bertopic import BERTopic
            from sklearn.feature_extraction.text import CountVectorizer
            from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
            import re
            
            # Extract all text
            all_texts = []
            doc_metadata = []
            
            for filename, doc_data in corpus_data['documents'].items():
                for para_idx, paragraph in enumerate(doc_data['paragraphs']):
                    all_texts.append(paragraph)
                    doc_metadata.append({
                        'filename': filename,
                        'paragraph_index': para_idx,
                        'title': doc_data['metadata']['title'],
                        'author': doc_data['metadata']['author']
                    })
            
            if not all_texts:
                return False, {}, "No text found for topic modeling"
            
            # Preprocess texts
            processed_texts = []
            preprocessing = config.get('preprocessing', {})
            
            for text in all_texts:
                processed_text = text
                
                if preprocessing.get('lowercase', True):
                    processed_text = processed_text.lower()
                
                if preprocessing.get('remove_punctuation', True):
                    processed_text = re.sub(r'[^\w\s]', ' ', processed_text)
                
                if preprocessing.get('remove_numbers', False):
                    processed_text = re.sub(r'\d+', '', processed_text)
                
                # Filter by word length
                min_length = preprocessing.get('min_word_length', 2)
                words = processed_text.split()
                words = [word for word in words if len(word) >= min_length]
                processed_text = ' '.join(words)
                
                processed_texts.append(processed_text)
            
            # Set up vectorizer with stopwords
            vectorizer_model = CountVectorizer(
                stop_words=config.get('stopwords'),
                min_df=2,
                max_df=0.95,
                ngram_range=(1, 2)
            )
            
            # Set up representation model
            representation_model = None
            rep_type = config.get('representation_model', 'KeyBERT')
            
            if rep_type == "KeyBERT" or rep_type == "KeyBERTInspired":
                representation_model = KeyBERTInspired()
            elif rep_type == "MaximalMarginalRelevance":
                representation_model = MaximalMarginalRelevance(diversity=0.3)
            
            # Initialize BERTopic
            topic_model = BERTopic(
                nr_topics=config.get('n_topics'),
                min_topic_size=config.get('min_topic_size', 10),
                vectorizer_model=vectorizer_model,
                representation_model=representation_model,
                verbose=False,
                calculate_probabilities=True
            )
            
            # Fit the model
            topics, probabilities = topic_model.fit_transform(processed_texts)
            
            # Get topic information
            topic_info = topic_model.get_topic_info()
            
            # Create results dictionary
            results = {
                'topic_model': topic_model,
                'topics': topics,
                'probabilities': probabilities,
                'topics_dataframe': topic_info,
                'total_documents': len(all_texts),
                'outlier_count': sum(1 for t in topics if t == -1),
                'topic_labels': {},
                'topic_words': {},
                'document_metadata': doc_metadata
            }
            
            # Extract topic words and labels
            for topic_id in topic_info['Topic'].unique():
                topic_id = int(topic_id)
                if topic_id != -1:  # Skip outlier topic
                    try:
                        topic_words = topic_model.get_topic(topic_id)
                        if topic_words:
                            results['topic_words'][topic_id] = topic_words
                    except Exception as e:
                        logger.warning(f"Could not get topic {topic_id}: {e}")
                        continue
                    # Create readable label from top words
                    top_words = [word for word, _ in topic_words[:3]]
                    results['topic_labels'][topic_id] = " + ".join(top_words)
            
            # Calculate topic coherence if possible
            try:
                from gensim.models import CoherenceModel
                from gensim.corpora import Dictionary
                
                # Prepare for coherence calculation
                texts_for_coherence = [text.split() for text in processed_texts]
                dictionary = Dictionary(texts_for_coherence)
                
                # Get topic words for coherence
                topic_words_for_coherence = []
                for topic_id in results['topic_words']:
                    words = [word for word, _ in results['topic_words'][topic_id][:10]]
                    topic_words_for_coherence.append(words)
                
                if topic_words_for_coherence:
                    coherence_model = CoherenceModel(
                        topics=topic_words_for_coherence,
                        texts=texts_for_coherence,
                        dictionary=dictionary,
                        coherence='c_v'
                    )
                    coherence_score = coherence_model.get_coherence()
                    results['coherence_score'] = coherence_score
            except ImportError:
                results['coherence_score'] = 0.0
            
            return True, results, f"Successfully extracted {len(topic_info)} topics from {len(all_texts)} documents"
            
        except Exception as e:
            return False, {}, f"Error in topic modeling: {str(e)}"                


    def get_suggested_vector_endpoints(self, suggested_terms: Dict, top_n: int = 20) -> List[Dict]:
        """Get suggested terms for vector endpoint creation"""
        try:
            all_suggestions = []
            
            # Add top TF-IDF terms
            if 'tfidf_terms' in suggested_terms:
                for term in suggested_terms['tfidf_terms'][:top_n//3]:
                    all_suggestions.append({
                        'term': term['term'],
                        'score': term['score'],
                        'source': 'TF-IDF',
                        'description': f"High importance term (TF-IDF: {term['score']:.3f})"
                    })
            
            # Add top frequency terms
            if 'frequency_terms' in suggested_terms:
                for term in suggested_terms['frequency_terms'][:top_n//3]:
                    all_suggestions.append({
                        'term': term['term'],
                        'score': term['score'],
                        'source': 'Frequency',
                        'description': f"Frequent term (appears {int(term['score'])} times)"
                    })
            
            # Add topic terms
            if 'topic_terms' in suggested_terms:
                for term in suggested_terms['topic_terms'][:top_n//3]:
                    all_suggestions.append({
                        'term': term['term'],
                        'score': term['score'],
                        'source': f"Topic {term.get('topic_id', '')}",
                        'description': f"Topic term (score: {term['score']:.3f})"
                    })
            
            # Remove duplicates and sort by score
            seen_terms = set()
            unique_suggestions = []
            for suggestion in all_suggestions:
                if suggestion['term'] not in seen_terms:
                    seen_terms.add(suggestion['term'])
                    unique_suggestions.append(suggestion)
            
            # Sort by score (descending)
            unique_suggestions.sort(key=lambda x: x['score'], reverse=True)
            
            return unique_suggestions[:top_n]
            
        except Exception as e:
            logger.error(f"Error getting suggested endpoints: {str(e)}")
            return []

