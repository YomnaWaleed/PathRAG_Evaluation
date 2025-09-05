# PDF loading and processing

import PyPDF2
import re
from typing import List, Dict
from pathlib import Path


def extract_text_from_pdf(pdf_path: str):
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def clean_text(text: str) -> str:
    """Clean extracted text."""
    text = re.sub(r"\s+", " ", text)
    # Remove special characters but keep technical terms
    text = re.sub(r"[^\w\s.,;:!?()-]", "", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(
            {
                "content": chunk,
                "start_index": i,
                "end_index": i + len(chunk.split()),
                "chunk_id": f"chunk_{i}",
            }
        )

    return chunks


def load_and_process_documents(data_dir: str = "data") -> Dict[str, List[Dict]]:
    """Load and process all PDF documents"""
    documents = {}
    data_path = Path(data_dir)

    for pdf_file in data_path.glob("*.pdf"):
        raw_text = extract_text_from_pdf(str(pdf_file))
        cleaned_text = clean_text(raw_text)
        chunks = chunk_text(cleaned_text)
        documents[pdf_file.stem] = chunks

    return documents
