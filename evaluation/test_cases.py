# Test cases and ground truth data

from typing import Dict, List

# Ground truth data for evaluation
TEST_QUERIES = {
    "single_hop": [
        {
            "query": "What is AUTOSAR ECU State Manager?",
            "relevant_chunk_ids": ["chunk_0", "chunk_1", "chunk_5"],
            "expected_entities": ["AUTOSAR", "ECU", "State Manager"],
        },
        {
            "query": "What are the key processes in Automotive SPICE?",
            "relevant_chunk_ids": ["chunk_10", "chunk_11", "chunk_15"],
            "expected_entities": ["Automotive SPICE", "processes", "assessment"],
        },
    ],
    "multi_hop": [
        {
            "query": "How does AUTOSAR ECU State Manager relate to Automotive SPICE processes?",
            "relevant_chunk_ids": ["chunk_1", "chunk_5", "chunk_10", "chunk_15"],
            "expected_entities": [
                "AUTOSAR",
                "ECU State Manager",
                "Automotive SPICE",
                "processes",
            ],
            "reasoning_steps": 2,
        },
        {
            "query": "What assessment methods are recommended for both AUTOSAR and Automotive SPICE?",
            "relevant_chunk_ids": ["chunk_3", "chunk_7", "chunk_12", "chunk_18"],
            "expected_entities": [
                "AUTOSAR",
                "Automotive SPICE",
                "assessment",
                "methods",
            ],
            "reasoning_steps": 3,
        },
    ],
}

# Manual evaluation criteria
EVALUATION_CRITERIA = {
    "relevance": {
        "description": "How relevant are the retrieved results to the query?",
        "scale": "1-5 (1=irrelevant, 5=highly relevant)",
    },
    "completeness": {
        "description": "Does the response cover all aspects of the query?",
        "scale": "1-5 (1=incomplete, 5=comprehensive)",
    },
    "accuracy": {
        "description": "Is the information in the response accurate?",
        "scale": "1-5 (1=inaccurate, 5=completely accurate)",
    },
    "groundedness": {
        "description": "Is the response well-supported by the source material?",
        "scale": "1-5 (1=unsupported, 5=fully supported)",
    },
}
