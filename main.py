# Main evaluation script
import asyncio
import json
from datetime import datetime
from typing import List, Dict
import os
import pandas as pd

from config.settings import GOOGLE_API_KEY, TOP_K_VALUES
from utils.data_loader import load_and_process_documents
from evaluation.multi_hop_simulator import MultiHopSimulator
from evaluation.metrics import EvaluationMetrics
from evaluation.test_cases import TEST_QUERIES
from utils.vizualization import ResultVisualizer
from dotenv import load_dotenv


async def main():
    load_dotenv()  # Load environment variables

    print("loading and processing documents...")
    documents = load_and_process_documents("data")

    print("initializing multi-hop query simulator...")
    simulator = MultiHopSimulator(documents)
    await simulator.build_entity_graph()

    print("starting evaluation...")
    results = {}

    # evaluate single-hop queries
    print("evaluating single-hop queries...")
    single_hop_results = await evaluate_queries(
        TEST_QUERIES["single_hop"], simulator, "single_hop", documents
    )
    results["single_hop"] = single_hop_results

    # evaluate multi-hop queries
    print("evaluating multi-hop queries...")
    multi_hop_results = await evaluate_queries(
        TEST_QUERIES["multi_hop"], simulator, "multi_hop", documents
    )
    results["multi_hop"] = multi_hop_results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    with open(f"results/evaluation_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Generating visualizations...")
    visualize_results(results)

    print("Evaluation completed!")


async def evaluate_queries(
    queries: List[Dict],
    simulator: MultiHopSimulator,
    query_type: str,
    documents: Dict,
) -> Dict:
    """Evaluate a set of queries"""
    results = {
        "recall_at_k": {f"recall@{k}": [] for k in TOP_K_VALUES},
        "precision_at_k": {f"precision@{k}": [] for k in TOP_K_VALUES},
        "groundedness_scores": [],
        "hallucination_rates": [],
        "query_details": [],
    }

    for i, query_data in enumerate(queries):
        print(
            f"Processing {query_type} query {i+1}/{len(queries)}: {query_data['query']}"
        )

        # Run the query
        query_results = await simulator.simulate_multi_hop(
            query_data["query"], max_hops=query_data.get("reasoning_steps", 2)
        )

        # Make sure retrieved_chunks are dicts, not strings
        retrieved_chunks = [chunk for chunk in query_results if isinstance(chunk, dict)]
        retrieved_ids = [chunk.get("chunk_id") for chunk in retrieved_chunks]

        # Get actual content from relevant chunks
        relevant_chunks = []
        for doc_name, chunks in documents.items():
            for chunk in chunks:
                if chunk.get("chunk_id") in query_data["relevant_chunk_ids"]:
                    relevant_chunks.append(chunk)

        # Calculate metrics
        for k in TOP_K_VALUES:
            recall = EvaluationMetrics.recall_at_k(retrieved_chunks, relevant_chunks, k)
            precision = EvaluationMetrics.precision_at_k(
                retrieved_chunks, relevant_chunks, k
            )

            results["recall_at_k"][f"recall@{k}"].append(recall)
            results["precision_at_k"][f"precision@{k}"].append(precision)

        # Calculate groundedness and hallucination (placeholders)
        groundedness = 0.8
        hallucination_rate = 0.1

        results["groundedness_scores"].append(groundedness)
        results["hallucination_rates"].append(hallucination_rate)
        retrieved_contents = [chunk.get("content", "") for chunk in retrieved_chunks]

        # Store query details
        results["query_details"].append(
            {
                "query": query_data["query"],
                "retrieved_ids": retrieved_ids,
                "retrieved_contents": retrieved_contents,
                "relevant_ids": query_data["relevant_chunk_ids"],
                "groundedness": groundedness,
                "hallucination_rate": hallucination_rate,
            }
        )

    # Calculate averages
    for k in TOP_K_VALUES:
        results["recall_at_k"][f"recall@{k}"] = sum(
            results["recall_at_k"][f"recall@{k}"]
        ) / len(queries)
        results["precision_at_k"][f"precision@{k}"] = sum(
            results["precision_at_k"][f"precision@{k}"]
        ) / len(queries)

    results["avg_groundedness"] = sum(results["groundedness_scores"]) / len(queries)
    results["avg_hallucination_rate"] = sum(results["hallucination_rates"]) / len(
        queries
    )

    return results


def visualize_results(results: Dict):
    """Generate visualizations from results"""
    # Plot Recall@k
    recall_data = {
        "single_hop": [
            results["single_hop"]["recall_at_k"][f"recall@{k}"] for k in TOP_K_VALUES
        ],
        "multi_hop": [
            results["multi_hop"]["recall_at_k"][f"recall@{k}"] for k in TOP_K_VALUES
        ],
    }
    ResultVisualizer.plot_recall_at_k(recall_data, TOP_K_VALUES)

    # Plot metrics comparison
    metrics_data = {
        "single_hop": {
            "avg_groundedness": results["single_hop"]["avg_groundedness"],
            "avg_hallucination": 1 - results["single_hop"]["avg_hallucination_rate"],
        },
        "multi_hop": {
            "avg_groundedness": results["multi_hop"]["avg_groundedness"],
            "avg_hallucination": 1 - results["multi_hop"]["avg_hallucination_rate"],
        },
    }
    ResultVisualizer.plot_metrics_comparison(metrics_data)

    # Create metrics table
    table_data = {
        "single_hop": {
            **{
                f"recall@{k}": results["single_hop"]["recall_at_k"][f"recall@{k}"]
                for k in TOP_K_VALUES
            },
            "avg_groundedness": results["single_hop"]["avg_groundedness"],
            "avg_hallucination_rate": results["single_hop"]["avg_hallucination_rate"],
        },
        "multi_hop": {
            **{
                f"recall@{k}": results["multi_hop"]["recall_at_k"][f"recall@{k}"]
                for k in TOP_K_VALUES
            },
            "avg_groundedness": results["multi_hop"]["avg_groundedness"],
            "avg_hallucination_rate": results["multi_hop"]["avg_hallucination_rate"],
        },
    }
    df_table = pd.DataFrame(
        {
            "Metric": [f"recall@{k}" for k in TOP_K_VALUES]
            + ["avg_groundedness", "avg_hallucination_rate"],
            "Single Hop": [
                results["single_hop"]["recall_at_k"][f"recall@{k}"]
                for k in TOP_K_VALUES
            ]
            + [
                results["single_hop"]["avg_groundedness"],
                results["single_hop"]["avg_hallucination_rate"],
            ],
            "Multi Hop": [
                results["multi_hop"]["recall_at_k"][f"recall@{k}"] for k in TOP_K_VALUES
            ]
            + [
                results["multi_hop"]["avg_groundedness"],
                results["multi_hop"]["avg_hallucination_rate"],
            ],
        }
    )

    print("Metrics Table:")
    print(df_table.to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
