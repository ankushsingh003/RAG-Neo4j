"""
Document Loader and Semantic Text Chunker
Prepares documents, chunks them with sliding windows and overlap,
and enriches each chunk with deterministic metadata.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    title: str
    content: str
    chunk_index: int
    char_count: int
    token_count: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DocumentLoader:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, file_path: Path) -> List[DocumentChunk]:
        """Loads a single text or markdown file and returns chunked documents."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

        doc_id = file_path.stem
        # Extract title from first markdown header or filename
        title_match = re.search(r"^#\s+(.+)$", raw_text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else file_path.name

        return self.chunk_text(raw_text=raw_text, doc_id=doc_id, title=title, source_path=str(file_path))

    def load_directory(self, dir_path: Path) -> List[DocumentChunk]:
        """Loads all .txt and .md files in the specified directory."""
        dir_path = Path(dir_path)
        all_chunks: List[DocumentChunk] = []
        if not dir_path.exists():
            return all_chunks

        for file_path in sorted(dir_path.glob("*")):
            if file_path.suffix.lower() in [".txt", ".md"]:
                chunks = self.load_file(file_path)
                all_chunks.extend(chunks)

        return all_chunks

    def chunk_text(self, raw_text: str, doc_id: str, title: str, source_path: str = "") -> List[DocumentChunk]:
        """Splits raw text into sliding window chunks respecting section boundaries."""
        # Split by double newline or markdown headers first to respect section semantics
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
        
        chunks: List[DocumentChunk] = []
        current_chunk_words: List[str] = []
        current_word_count = 0
        chunk_idx = 0

        for para in paragraphs:
            para_words = para.split()
            if not para_words:
                continue

            if current_word_count + len(para_words) <= self.chunk_size:
                current_chunk_words.extend(para_words)
                current_word_count += len(para_words)
            else:
                # Flush existing chunk if non-empty
                if current_chunk_words:
                    chunk_text = " ".join(current_chunk_words)
                    chunk_id = f"{doc_id}_chunk_{chunk_idx:02d}"
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        title=title,
                        content=chunk_text,
                        chunk_index=chunk_idx,
                        char_count=len(chunk_text),
                        token_count=len(current_chunk_words),
                        metadata={"source": source_path, "title": title}
                    ))
                    chunk_idx += 1

                    # Keep overlap words
                    overlap_words = current_chunk_words[-self.chunk_overlap:] if self.chunk_overlap > 0 else []
                    current_chunk_words = overlap_words + para_words
                    current_word_count = len(current_chunk_words)
                else:
                    # Paragraph itself is larger than chunk size, chunk it directly
                    for i in range(0, len(para_words), max(1, self.chunk_size - self.chunk_overlap)):
                        slice_words = para_words[i:i + self.chunk_size]
                        chunk_text = " ".join(slice_words)
                        chunk_id = f"{doc_id}_chunk_{chunk_idx:02d}"
                        chunks.append(DocumentChunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            title=title,
                            content=chunk_text,
                            chunk_index=chunk_idx,
                            char_count=len(chunk_text),
                            token_count=len(slice_words),
                            metadata={"source": source_path, "title": title}
                        ))
                        chunk_idx += 1
                    current_chunk_words = []
                    current_word_count = 0

        # Flush final remainder
        if current_chunk_words:
            chunk_text = " ".join(current_chunk_words)
            chunk_id = f"{doc_id}_chunk_{chunk_idx:02d}"
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                title=title,
                content=chunk_text,
                chunk_index=chunk_idx,
                char_count=len(chunk_text),
                token_count=len(current_chunk_words),
                metadata={"source": source_path, "title": title}
            ))

        return chunks
