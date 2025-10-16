from typing import List, Iterable
import os
from sentence_transformers import SentenceTransformer
import torch, numpy as np
_MODEL_NAME_Q = os.getenv("RAG_QUERY_ENCODER", "intfloat/multilingual-e5-base")
_MODEL_NAME_P = os.getenv("RAG_PASSAGE_ENCODER", _MODEL_NAME_Q)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_q = SentenceTransformer(_MODEL_NAME_Q, device=_DEVICE)
_p = SentenceTransformer(_MODEL_NAME_P, device=_DEVICE)
def _encode(texts: Iterable[str], model: SentenceTransformer) -> np.ndarray:
    embs = model.encode(list(texts), batch_size=int(os.getenv("RAG_BATCH", "32")),
                        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return embs.astype(np.float32)
def encode_query(text: str) -> List[float]:
    return _encode([text], _q)[0].tolist()
def encode_passage(text: str) -> List[float]:
    return _encode([text], _p)[0].tolist()
