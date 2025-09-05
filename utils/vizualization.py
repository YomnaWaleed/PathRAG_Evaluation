# Results visualization

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict


class ResultVisualizer:
    @staticmethod
    def plot_recall_at_k(recall_results: Dict[str, List[float]], k_values: List[int]):
        """Plot Recall@k for different queries"""
        plt.figure(figsize=(10, 6))

        for query_type, recalls in recall_results.items():
            plt.plot(k_values, recalls, marker="o", label=query_type)

        plt.xlabel("k")
        plt.ylabel("Recall@k")
        plt.title("Recall@k for Different Query Types")
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_metrics_comparison(metrics_data: Dict[str, Dict[str, float]]):
        """Plot comparison of different evaluation metrics"""
        metrics_df = pd.DataFrame(metrics_data).T

        plt.figure(figsize=(12, 6))
        metrics_df.plot(kind="bar")
        plt.title("Evaluation Metrics Comparison")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def create_metrics_table(metrics_data: Dict[str, Dict[str, float]]):
        """Create a formatted table of evaluation metrics"""
        df = pd.DataFrame(metrics_data).T
        return df.style.background_gradient(cmap="viridis")
