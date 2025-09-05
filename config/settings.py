# configuration and environment variables

import os
from dotenv import load_dotenv
from pathlib import Path
from google.generativeai import configure

load_dotenv()
# configure Gemeini
GEMINIT_API_KEY = os.getenv("GEMINIT_API_KEY")

# Model configurations
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-2.5-pro"
LLM_TEMPERATURE = 0.2
MAX_RETRIES = 2

# Evaluation parameters
TOP_K_VALUES = [1, 3, 5, 10]  # for Recall@K
SIMILARITY_THRESHOLD = 0.8  # for groundness
