import os
from typing import List
from keybert import KeyBERT
from utils.utils import new_cd
parent_dir = os.path.dirname(__file__)


class KeyBertModel:
    def __init__(self, args):
        with new_cd(parent_dir):
            self.model = KeyBERT(model=f"../checkpoints/{args.keybert_version}")

    @staticmethod
    def extract_similar_high_score_keywords(input_list: list, score_difference_threshold:float = 0.1) -> List[str]:
        """find truly keywords from query"""
        filtered_list = [item for item in input_list if item[0] != 'video']
        sorted_list = sorted(filtered_list, key=lambda x: x[1], reverse=True)
        similar_high_score_keywords = []
        if sorted_list:
            highest_score = sorted_list[0][1]  
            for item in sorted_list:
                if highest_score - item[1] <= score_difference_threshold:
                    similar_high_score_keywords.append(item[0])
                else:
                    break  
        return similar_high_score_keywords
    
    def extract_keywords(self, text, top_n=5, keyphrase_ngram_range=(1, 1), stop_words='english') -> List[str]:
        """Extract keywords from query."""
        keywords = self.model.extract_keywords(text, keyphrase_ngram_range=keyphrase_ngram_range, stop_words=stop_words, top_n=top_n)
        return self.extract_similar_high_score_keywords(keywords)