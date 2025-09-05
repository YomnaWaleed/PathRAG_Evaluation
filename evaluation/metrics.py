# Evaluation metrics implementation

from typing import List, Dict, Set
from sklearn.metrics import precision_score, recall_score
import numpy as np


class EvaluationMetrics:
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Calculate Recall@K"""
        if not relevant_ids:
            return 0.0

        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)

        if not relevant_set:
            return 0.0

        return len(retrieved_at_k.intersection(relevant_set)) / len(relevant_set)

    @staticmethod
    def precision_at_k(
        retrieved_ids: List[str], relevant_ids: List[str], k: int
    ) -> float:
        """Calculate Precision@K"""
        if not retrieved_ids:
            return 0.0

        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)

        if not retrieved_at_k:
            return 0.0

        return len(retrieved_at_k.intersection(relevant_set)) / len(retrieved_at_k)

    @staticmethod
    def calculate_groundeness(answer: str, context: str, embedding_model) -> float:
        """
        Calculate how well the answer is grounded in the context
        using embedding similarity
        """
        answer_embedding = embedding_model.embed_content(answer)
        context_embedding = embedding_model.embed_content(context)

        similarity = EvaluationMetrics.cosine_similarity(
            answer_embedding, context_embedding
        )
        return similarity

    @staticmethod
    def detect_hallucination(
        answer: str, context: str, llm_model, threshold: float = 0.8
    ) -> float:
        """
        Detect hallucination by checking if answer claims are supported by context
        Returns hallucination rate (0-1)
        """
        # Use LLM to verify each claim in the answer against context
        claims = EvaluationMetrics.extract_claims(answer)
        supported_claims = 0

        for claim in claims:
            if EvaluationMetrics.is_claim_supported(claim, context, llm_model):
                supported_claims += 1

        if not claims:
            return 0.0

        hallucination_rate = 1 - (supported_claims / len(claims))
        return hallucination_rate

    @staticmethod
    def extract_claims(text: str) -> List[str]:
        """Extract individual claims from text"""
        # Simple implementation - split by sentences
        import re

        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def is_claim_supported(claim: str, context: str, llm_model) -> bool:
        """Check if a clain is supported by a context using LLM"""
        prompt = f"""
        Determine if the following claim is supported by the context.
        
        Claim: {claim}
        
        Context: {context}
        
        Respond only with "SUPPORTED" or "NOT_SUPPORTED".
        """
        response = llm_model.generate_text(prompt)
        return "SUPPORTED" in response.text.upper()

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 and norm2 else 0
