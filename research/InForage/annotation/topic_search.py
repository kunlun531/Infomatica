import json
import os
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import numpy as np

class TopicSearcher:
    def __init__(self, data_path: str = "target_topic.jsonl"):
        """
        Initialize the TopicSearcher with BM25 for searching topics and targets.
        
        Args:
            data_path: Path to the target_topic.jsonl file
        """
        self.data_path = data_path
        self.topics = []
        self.corpus = []
        self.load_data()
        self.bm25 = self._create_bm25_index()
        
    def load_data(self) -> None:
        """Load data from the jsonl file."""
        self.topics = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    topic_data = json.loads(line.strip())
                    self.topics.append(topic_data)
                    # Create a document combining topic and target for search
                    self.corpus.append(f"{topic_data['topic']} {topic_data['target']}")
                except json.JSONDecodeError:
                    continue
    
    def _create_bm25_index(self) -> BM25Okapi:
        """Create BM25 index from the corpus."""
        # Tokenize the corpus
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        return BM25Okapi(tokenized_corpus)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for topics and targets related to the query.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of topic data dictionaries
        """
        # Tokenize the query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_k_indices = np.argsort(doc_scores)[::-1][:k]
        
        # Return the corresponding topic data
        results = [self.topics[idx] for idx in top_k_indices]
        
        return results

def search_topics(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Convenience function to search topics without creating a TopicSearcher instance.
    
    Args:
        query: The search query
        k: Number of results to return
        
    Returns:
        List of topic data dictionaries
    """
    searcher = TopicSearcher()
    return searcher.search(query, k)


if __name__ == "__main__":
    print(search_topics("test"))