```python
import re
import string
from typing import List, Dict, Any, Optional

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.util import ngrams
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.util import ngrams
except ImportError:
    print("NLTK is required for some quality metrics. Please install it: pip install nltk")
    # Fallback / dummy implementations if NLTK is not available
    def word_tokenize(text): return text.split()
    def sent_tokenize(text): return text.split('.')
    def ngrams(tokens, n): return []

try:
    import textstat
except ImportError:
    print("Textstat is required for readability metrics. Please install it: pip install textstat")
    # Fallback / dummy implementations if textstat is not available
    class TextStatDummy:
        def __getattr__(self, name):
            return lambda *args, **kwargs: 0.0 # Return 0 for any readability score
    textstat = TextStatDummy()


class QualityMetrics:
    """
    Evaluates the linguistic diversity, semantic coverage, fluency, and relevance
    of generated descriptions for novel tokens.

    This module provides various quantitative metrics to provide objective feedback
    on description quality, crucial for prioritizing candidates for human review
    and informing active learning strategies.
    """

    def __init__(self):
        """
        Initializes the QualityMetrics module.
        No heavy external model loading in constructor for flexibility,
        can be extended later with e.g. embedding models.
        """
        pass

    def _preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing: lowercase and remove punctuation.
        """
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def _calculate_readability(self, description: str) -> Dict[str, float]:
        """
        Calculates various readability scores for a given description.
        Requires the 'textstat' library.
        """
        if not description.strip():
            return {
                "flesch_reading_ease": 0.0,
                "flesch_kincaid_grade": 0.0,
                "dale_chall_readability": 0.0,
                "difficult_words": 0,
            }
        try:
            return {
                "flesch_reading_ease": textstat.flesch_reading_ease(description),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(description),
                "dale_chall_readability": textstat.dale_chall_readability_score(description),
                "difficult_words": textstat.difficult_words(description),
            }
        except Exception as e:
            # Handle potential errors from textstat for malformed input
            print(f"Error calculating readability scores: {e}")
            return {
                "flesch_reading_ease": 0.0,
                "flesch_kincaid_grade": 0.0,
                "dale_chall_readability": 0.0,
                "difficult_words": 0,
            }

    def _calculate_linguistic_diversity(self, description: str) -> Dict[str, float]:
        """
        Calculates linguistic diversity metrics like Type-Token Ratio and N-gram diversity.
        Requires the 'nltk' library.
        """
        if not description.strip():
            return {
                "word_count": 0,
                "sentence_count": 0,
                "type_token_ratio": 0.0,
                "unique_bigram_ratio": 0.0,
                "unique_trigram_ratio": 0.0,
            }

        preprocessed_desc = self._preprocess_text(description)
        words = word_tokenize(preprocessed_desc)
        sentences = sent_tokenize(description)

        word_count = len(words)
        sentence_count = len(sentences)

        # Type-Token Ratio (TTR)
        ttr = len(set(words)) / word_count if word_count > 0 else 0.0

        # N-gram diversity
        bigrams = list(ngrams(words, 2))
        trigrams = list(ngrams(words, 3))

        unique_bigram_ratio = len(set(bigrams)) / len(bigrams) if len(bigrams) > 0 else 0.0
        unique_trigram_ratio = len(set(trigrams)) / len(trigrams) if len(trigrams) > 0 else 0.0

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "type_token_ratio": ttr,
            "unique_bigram_ratio": unique_bigram_ratio,
            "unique_trigram_ratio": unique_trigram_ratio,
        }

    def _calculate_semantic_relevance(self, token: str, description: str, context: Optional[str] = None) -> Dict[str, float]:
        """
        Calculates basic semantic relevance by checking for keyword overlap
        between the description and the token/context.
        This is a simple proxy for 'semantic coverage' without embeddings.
        """
        if not description.strip():
            return {
                "token_keyword_overlap": 0.0,
                "context_keyword_overlap": 0.0,
            }

        desc_words = set(word_tokenize(self._preprocess_text(description)))
        
        # Token keyword overlap
        token_words = set(word_tokenize(self._preprocess_text(token)))
        token_overlap = len(desc_words.intersection(token_words))
        token_overlap_ratio = token_overlap / len(token_words) if len(token_words) > 0 else 0.0
        
        # Context keyword overlap (if context is provided)
        context_overlap_ratio = 0.0
        if context and context.strip():
            context_words = set(word_tokenize(self._preprocess_text(context)))
            context_overlap = len(desc_words.intersection(context_words))
            context_overlap_ratio = context_overlap / len(context_words) if len(context_words) > 0 else 0.0

        return {
            "token_keyword_overlap": token_overlap_ratio,
            "context_keyword_overlap": context_overlap_ratio,
        }

    def _compare_to_references(self, description: str, reference_descriptions: List[str]) -> Dict[str, float]:
        """
        Compares the generated description to a list of reference descriptions
        for similar or related existing tokens.
        This helps assess distinctiveness or potential redundancy.
        Currently, uses Jaccard similarity as a simple proxy.
        """
        if not description.strip() or not reference_descriptions:
            return {
                "max_jaccard_similarity_to_references": 0.0,
                "avg_jaccard_similarity_to_references": 0.0,
            }

        desc_tokens = set(word_tokenize(self._preprocess_text(description)))
        if not desc_tokens:
            return {
                "max_jaccard_similarity_to_references": 0.0,
                "avg_jaccard_similarity_to_references": 0.0,
            }

        similarities = []
        for ref_desc in reference_descriptions:
            if not ref_desc.strip():
                continue
            ref_tokens = set(word_tokenize(self._preprocess_text(ref_desc)))
            if not ref_tokens:
                continue

            intersection = len(desc_tokens.intersection(ref_tokens))
            union = len(desc_tokens.union(ref_tokens))
            jaccard = intersection / union if union > 0 else 0.0
            similarities.append(jaccard)

        if not similarities:
            return {
                "max_jaccard_similarity_to_references": 0.0,
                "avg_jaccard_similarity_to_references": 0.0,
            }

        return {
            "max_jaccard_similarity_to_references": max(similarities),
            "avg_jaccard_similarity_to_references": sum(similarities) / len(similarities),
        }

    def calculate_scores(self, 
                         token: str, 
                         description: str, 
                         context: Optional[str] = None, 
                         reference_descriptions: Optional[List[str]] = None
                         ) -> Dict[str, Any]:
        """
        Calculates a comprehensive set of quality scores for a generated description.

        Args:
            token (str): The new vocabulary token for which the description was generated.
            description (str): The generated linguistic supervision description.
            context (Optional[str]): Additional context that was used during generation
                                     (e.g., from Knowledge Integrator).
            reference_descriptions (Optional[List[str]]): A list of descriptions for
                                                           similar/related existing tokens,
                                                           used for comparative analysis.

        Returns:
            Dict[str, Any]: A dictionary containing various quality scores.
        """
        if not description or not description.strip():
            return {
                "error": "Description is empty or invalid.",
                "word_count": 0,
                "sentence_count": 0,
                "flesch_reading_ease": 0.0,
                "flesch_kincaid_grade": 0.0,
                "type_token_ratio": 0.0,
                "unique_bigram_ratio": 0.0,
                "token_keyword_overlap": 0.0,
                "context_keyword_overlap": 0.0,
                "max_jaccard_similarity_to_references": 0.0,
                "avg_jaccard_similarity_to_references": 0.0,
            }

        scores = {}
        
        # Linguistic Diversity and Fluency
        diversity_scores = self._calculate_linguistic_diversity(description)
        scores.update(diversity_scores)

        # Readability
        readability_scores = self._calculate_readability(description)
        scores.update(readability_scores)

        # Semantic Relevance (proxy)
        relevance_scores = self._calculate_semantic_relevance(token, description, context)
        scores.update(relevance_scores)

        # Comparison to references
        if reference_descriptions:
            comparison_scores = self._compare_to_references(description, reference_descriptions)
            scores.update(comparison_scores)
        else:
            scores.update({
                "max_jaccard_similarity_to_references": 0.0,
                "avg_jaccard_similarity_to_references": 0.0,
            })
            
        return scores

# Example Usage (for testing purposes, remove in final production code if not needed)
if __name__ == "__main__":
    metrics_calculator = QualityMetrics()

    new_token_1 = "QuantumEntanglement"
    description_1 = "Quantum entanglement is a physical phenomenon that occurs when a group of particles are generated, interact, or share spatial proximity in a way that the quantum state of each particle cannot be described independently of the others, even when the particles are separated by a large distance."
    context_1 = "Physics, quantum mechanics, non-locality, spooky action at a distance."
    
    new_token_2 = "NeuralNetwork"
    description_2 = "A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. Neural networks can adapt to changing input; so the network generates the best possible result without needing to redesign the output criteria."
    context_2 = "Artificial intelligence, machine learning, deep learning, algorithms, brain."

    new_token_3 = "Blockchain"
    description_3 = "Blockchain is a decentralized, distributed ledger technology that records transactions across many computers so that any involved block cannot be altered retroactively, without the alteration of all subsequent blocks."
    context_3 = "Cryptocurrency, distributed ledger, security, decentralization, ledger."

    # Example reference descriptions for 'similar' concepts to Neural Network
    reference_descriptions_nn = [
        "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
        "An algorithm is a set of rules to be followed in calculations or other problem-solving operations, especially by a computer.",
        "Deep learning is a subset of machine learning in artificial intelligence that has networks capable of learning unsupervised from data that is unstructured or unlabeled.",
    ]

    print(f"--- Scores for Token: '{new_token_1}' ---")
    scores_1 = metrics_calculator.calculate_scores(new_token_1, description_1, context_1)
    for key, value in scores_1.items():
        print(f"  {key}: {value}")

    print(f"\n--- Scores for Token: '{new_token_2}' ---")
    scores_2 = metrics_calculator.calculate_scores(new_token_2, description_2, context_2, reference_descriptions_nn)
    for key, value in scores_2.items():
        print(f"  {key}: {value}")
        
    print(f"\n--- Scores for Token: '{new_token_3}' (no references) ---")
    scores_3 = metrics_calculator.calculate_scores(new_token_3, description_3, context_3)
    for key, value in scores_3.items():
        print(f"  {key}: {value}")

    # Test with empty description
    print(f"\n--- Scores for Empty Description ---")
    empty_token = "EmptyConcept"
    empty_desc = ""
    scores_empty = metrics_calculator.calculate_scores(empty_token, empty_desc)
    for key, value in scores_empty.items():
        print(f"  {key}: {value}")

    # Test with very short description
    print(f"\n--- Scores for Short Description ---")
    short_token = "Dog"
    short_desc = "A furry friend."
    scores_short = metrics_calculator.calculate_scores(short_token, short_desc)
    for key, value in scores_short.items():
        print(f"  {key}: {value}")
```