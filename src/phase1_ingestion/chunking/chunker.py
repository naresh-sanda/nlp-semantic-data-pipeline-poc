import os
import time
from utils.logger import log_step

class Chunker:
    def __init__(self, strategy="fixed"):
        self.strategy = strategy

    def chunk_text(self, text, metadata=None):
        start_time = time.time()
        strategy = self.strategy
        log_step(f"Splitting text using strategy '{strategy}'...")
        
        if strategy == "fixed":
            chunks = self._fixed_chunking(text, chunk_size=200)
        elif strategy == "recursive":
            chunks = self._recursive_chunking(text)
        elif strategy == "metadata-aware":
            chunks = self._metadata_aware_chunking(text, metadata)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        log_step(f"Split into {len(chunks)} chunks", start_time)
        return chunks

    def _fixed_chunking(self, text, chunk_size=200):
        """Splits text into fixed character chunks."""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
        return chunks

    def _recursive_chunking(self, text):
        """Simulates recursive chunking by splitting on double newlines then single newlines."""
        chunks = []
        paragraphs = text.split("\n\n")
        for p in paragraphs:
            if len(p) > 300:
                chunks.extend(p.split("\n"))
            else:
                chunks.append(p)
        return [c for c in chunks if c.strip()]

    def _metadata_aware_chunking(self, text, metadata):
        """Chunking that keeps semantic blocks together based on terms."""
        # Simple implementation: split by 'Term:' or 'Rule:'
        lines = text.split("\n")
        chunks = []
        current_chunk = []
        for line in lines:
            if line.startswith("Term:") or line.startswith("Rule:") or line.startswith("Rule "):
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        return [c for c in chunks if c.strip()]
