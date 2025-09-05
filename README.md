# PathRAG Evaluation Framework

A comprehensive evaluation framework for multi-hop Retrieval Augmented Generation (RAG) systems, specifically designed for technical documentation analysis with AUTOSAR and Automotive SPICE standards.

## Overview

This project implements a sophisticated evaluation harness for assessing the performance of multi-hop RAG systems. It provides metrics for measuring retrieval accuracy, groundedness, and hallucination rates across both single-hop and multi-hop queries.

## Features

- **Multi-hop Query Simulation**: Simulates complex reasoning across interconnected entities
- **Comprehensive Evaluation Metrics**: 
  - Recall@K and Precision@K for retrieval performance
  - Groundedness scoring for answer quality
  - Hallucination rate detection
- **Entity Graph Construction**: Builds knowledge graphs from technical documents
- **Automated Visualization**: Generates interactive charts and metrics tables
- **PDF Document Processing**: Handles technical documentation extraction and chunking

## Architecture

PathRAG_Evaluation

├── config

│ └── settings.py # Configuration and API settings

├── evaluation

│ ├── metrics.py # Evaluation metrics implementation

│ ├── multi_hop_simulator.py # Multi-hop reasoning engine

│ └── test_cases.py # Test queries and ground truth

├── utils

│ ├── data_loader.py # PDF processing and chunking

│ └── visualization.py # Results visualization

├── data/ # PDF documents directory

├── results/ # Evaluation results storage

└── main.py # Main evaluation script

## Evaluation Metrics

### Core Metrics:

- **Recall@K**: Measures retrieval completeness at different cutoff points (K=1,3,5,10)
- **Precision@K**: Measures retrieval accuracy at different cutoff points
- **Groundedness**: Evaluates how well answers are supported by source material (0-1 scale)
- **Hallucination Rate**: Detects fabricated or unsupported information (0-1 scale)

### Query Types:

- **Single-hop Queries**: Direct information retrieval
- **Multi-hop Queries**: Complex reasoning across multiple concepts