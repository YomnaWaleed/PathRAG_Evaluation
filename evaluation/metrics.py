# Evaluation metrics implementation

from typing import List, Dict, Set
from sklearn.metrics import precision_score, recall_score
import numpy as np


class EvaluationMetrics:
    @staticmethod
    def recall_at_k(
        retrieved_chunks: List[Dict], relevant_chunks: List[Dict], k: int
    ) -> float:
        """Calculate Recall@K based on content similarity"""
        if not relevant_chunks:
            return 0.0

        # Take top-K retrieved chunks
        retrieved_at_k = retrieved_chunks[:k]

        # Make a set of relevant content strings
        relevant_set = set(chunk.get("content", "") for chunk in relevant_chunks)

        # Count matches
        matches = 0
        for chunk in retrieved_at_k:
            chunk_content = chunk.get("content", "")
            for relevant_text in relevant_set:
                if relevant_text.lower() in chunk_content.lower():
                    matches += 1
                    break

        return matches / len(relevant_set) if relevant_set else 0.0

    @staticmethod
    def precision_at_k(
        retrieved_chunks: List[Dict], relevant_chunks: List[Dict], k: int
    ) -> float:
        """Calculate Precision@K based on chunk_ids"""
        if not retrieved_chunks:
            return 0.0

        retrieved_at_k = set(chunk.get("chunk_id") for chunk in retrieved_chunks[:k])
        relevant_set = set(chunk.get("chunk_id") for chunk in relevant_chunks)

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
