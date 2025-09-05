# Multi-hop query simulation

import asyncio
import networkx as nx
from typing import List, Tuple, Dict
from google.generativeai import embed_content
import google.generativeai as genai
from config.settings import LLM_MODEL
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.settings import EMBEDDING_MODEL, GOOGLE_API_KEY


class MultiHopSimulator:
    def __init__(self, documents: Dict[str, List[Dict]]):
        self.documents = documents
        self.graph = nx.DiGraph()
        self.entity_embeddings = {}

    async def perform_reasoning(
        self,
        query_entities: List[str],
        relevant_chunks: Dict[str, List[Dict]],
        max_hops: int,
    ) -> List[Dict]:
        """
        Perform multi-hop reasoning over the graph and relevant chunks.
        For now: just collect top chunks for each query entity up to max_hops.
        """
        results = []
        try:
            for hop in range(max_hops):
                for entity in query_entities:
                    if entity in relevant_chunks:
                        for chunk in relevant_chunks[entity]:
                            results.append(
                                {
                                    "hop": hop + 1,
                                    "entity": entity,
                                    "content": chunk.get("content", ""),
                                    "doc": chunk.get("doc_name", "unknown"),
                                    "score": chunk.get("score", 0),
                                }
                            )
            return results
        except Exception as e:
            print(f"[Error in perform_reasoning]: {e}")
            return []

    async def extract_entities(self, text: str) -> List[str]:
        """Extract entities from text using Gemini LLM"""
        prompt = f"""
        Extract the main entities (technical terms, standards, processes, tools, or concepts)
        from the following text. 
        Return them as a comma-separated list only, without explanations.

        Text: {text}
        """

        try:
            model = genai.GenerativeModel(model_name=LLM_MODEL)
            response = model.generate_content(prompt)

            # Gemini output as plain text
            entities_text = response.text.strip()

            # Split by comma and clean
            entities = [e.strip() for e in entities_text.split(",") if e.strip()]
            return entities if entities else []
        except Exception as e:
            print(f"[Error in extract_entities]: {e}")
            return []

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Gemini"""
        result = embed_content(
            model=EMBEDDING_MODEL, content=text, task_type="retrieval_document"
        )
        return result["embedding"]

    async def build_entity_graph(self):
        """Build a graph of entities and their relationships"""
        try:
            for doc_id, text in self.documents.items():
                entities = await self.extract_entities(text)

                for ent in entities:
                    if ent not in self.graph:
                        self.graph.add_node(ent, doc_id=doc_id)

                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        e1, e2 = entities[i], entities[j]

                        if self.graph.has_edge(e1, e2):
                            self.graph[e1][e2]["weight"] += 1
                        else:
                            self.graph.add_edge(e1, e2, weight=1)

            print(
                f" Entity graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges."
            )

        except Exception as e:
            print(f"[Error in build_entity_graph]: {e}")

    async def simulate_multi_hop(self, query: str, max_hops: int = 3) -> List[Dict]:
        """Simulate multi-hop query processing"""
        # 1. Extract entities from query
        query_entities = await self.extract_entities(query)

        # 2. Find relevant chunks for each entity
        relevant_chunks = await self.find_relevant_chunks(query_entities)

        # 3. Perform multi-hop reasoning
        results = await self.perform_reasoning(
            query_entities, relevant_chunks, max_hops
        )

        return results

    async def find_relevant_chunks(self, entities: List[str]) -> Dict[str, List[Dict]]:
        """Find chunks relevant to each entity"""
        relevant_chunks = {}

        for entity in entities:
            entity_embedding = await self.get_embedding(entity)
            relevant_chunks[entity] = await self.find_similar_chunks(entity_embedding)

        return relevant_chunks

    async def find_similar_chunks(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict]:
        """Find chunks similar to the query embedding"""
        similarities = []

        for doc_name, chunks in self.documents.items():
            for chunk in chunks:
                if "embedding" not in chunk:
                    chunk["embedding"] = await self.get_embedding(chunk["content"])

                similarity = self.cosine_similarity(query_embedding, chunk["embedding"])
                similarities.append((similarity, chunk))

        # Return top_k most similar chunks
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in similarities[:top_k]]

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 and norm2 else 0
