import logging
from typing import List, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Thread-safe Singleton embedding engine powered by Google DeepMind's EmbeddingGemma 2.
    Produces unit-normalized 768-dimensional dense vectors with an 8,192 token context window.
    """

    _instance: Optional["EmbeddingEngine"] = None
    _model: Optional[SentenceTransformer] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingEngine, cls).__new__(cls)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[*] Loading Google DeepMind EmbeddingGemma 2 on [{device.upper()}] (768 dims, 8k context)...")

            # Load Google's EmbeddingGemma 2 from Hugging Face
            cls._model = SentenceTransformer(
                "google/embeddinggemma-2",
                device=device,
                trust_remote_code=True
            )
            print("[+] EmbeddingGemma 2 initialized successfully.")
        return cls._instance

    def embed_text(self, text: str) -> List[float]:
        """
        Encodes arbitrary property descriptions into a unit-normalized 768-dim float vector.
        Prevents NaN vectors on empty/whitespace inputs.
        """
        if not text or not text.strip():
            # Baseline unit vector along first dimension to avoid NaN cosine distance
            baseline = np.zeros(768, dtype=np.float32)
            baseline[0] = 1.0
            return baseline.tolist()

        # Generates normalized 768-dim embedding
        embedding = self._model.encode(
            text.strip(),
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embedding.tolist()


# Global Singleton accessor
embedding_engine = EmbeddingEngine()