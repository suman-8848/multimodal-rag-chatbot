"""Retrieval-augmented generation over the MR Tips & Tricks ChromaDB index.

Retrieves relevant text chunks (MiniLM) and images (CLIP, queried via its text
encoder into the same embedding space as the indexed images) from the
collections built by ingest/build_index.py, then generates a grounded answer
using one of two interchangeable LLM backends (see RAG_LLM_BACKEND below):

- "ollama"  — a local Ollama server. Used for local development.
- "zerogpu" — a small open model (Qwen2.5-3B-Instruct) run in-process via
  transformers, GPU-accelerated for free on HF Spaces' ZeroGPU when deployed
  there (falls back to CPU when run elsewhere, e.g. local testing). Chosen
  over Ollama-on-CPU (too slow on Spaces' free 2 vCPU hardware) and over HF's
  paid Inference Providers API (free tier is a thin $0.10/month credit) —
  ZeroGPU is genuinely free, just capped at 5 minutes/day of GPU time on a
  free HF account, shared across all visitors. See README for details.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import ollama
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

try:
    import spaces

    _HAS_SPACES = True
except ImportError:
    _HAS_SPACES = False

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "data" / "chroma_db"

TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
TEXT_COLLECTION = "text_chunks"
IMAGE_COLLECTION = "image_chunks"

# "llama3" (8B, Q4_0, ~4.7GB) was already pulled locally from an earlier prototype,
# so it's used as-is to avoid another large download on this connection. It's a
# heavier model than ideal for CPU-only inference (i7-1355U, 13.6GB RAM, no
# discrete GPU, ~3-8 tok/s expected) — swap to a smaller quantized model (e.g.
# "llama3.2:3b", pulled via `ollama pull llama3.2:3b`) if latency matters more
# than answer quality. Only used by the "ollama" backend.
OLLAMA_MODEL = "llama3"

# Ungated (Apache-2.0) so it downloads with no license click-through or
# HF_TOKEN needed — small enough to be fast on ZeroGPU's shared hardware.
# Only used by the "zerogpu" backend.
ZEROGPU_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

DEFAULT_LLM_BACKEND = os.environ.get("RAG_LLM_BACKEND", "ollama")

TOP_K_TEXT = 4
TOP_K_IMAGES = 2

SYSTEM_PROMPT = (
    "You are an assistant answering questions about a utility pole make-ready "
    "reference document (MR Tips & Tricks). Answer ONLY using the provided "
    "context. If the context doesn't contain the answer, say you don't know "
    "rather than guessing."
)


@dataclass
class TextSource:
    text: str
    section: str
    subsection: str
    distance: float


@dataclass
class ImageSource:
    image_path: str
    description: str
    section: str
    subsection: str
    distance: float


@dataclass
class RagResult:
    answer: str
    text_sources: list[TextSource] = field(default_factory=list)
    image_sources: list[ImageSource] = field(default_factory=list)


def _pooled(output):
    """Both CLIP get_*_features methods return BaseModelOutputWithPooling on
    this transformers version, with the projected embedding in .pooler_output
    (a bare tensor on older versions)."""
    return output.pooler_output if hasattr(output, "pooler_output") else output


_zerogpu_pipeline = None


def _get_zerogpu_pipeline():
    """Lazily load+cache the local generation model, placed on 'cuda' outside
    the @spaces.GPU-decorated call as HF's docs recommend (a CUDA emulation
    layer makes that work even without a real GPU attached yet)."""
    global _zerogpu_pipeline
    if _zerogpu_pipeline is None:
        from transformers import pipeline as hf_pipeline

        device = "cuda" if (_HAS_SPACES or torch.cuda.is_available()) else "cpu"
        _zerogpu_pipeline = hf_pipeline(
            "text-generation", model=ZEROGPU_MODEL_NAME, device=device, torch_dtype="auto"
        )
    return _zerogpu_pipeline


def _zerogpu_generate_impl(system_prompt: str, user_prompt: str) -> str:
    pipe = _get_zerogpu_pipeline()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    output = pipe(messages, max_new_tokens=512, do_sample=False)
    return output[0]["generated_text"][-1]["content"]


# @spaces.GPU is a documented no-op outside a real ZeroGPU Space, so this is
# safe to wrap unconditionally whenever the `spaces` package is installed.
_zerogpu_generate = spaces.GPU(_zerogpu_generate_impl) if _HAS_SPACES else _zerogpu_generate_impl


class RagPipeline:
    """Loads embedding models + Chroma collections once, then answers queries."""

    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        llm_backend: str = DEFAULT_LLM_BACKEND,
        ollama_model: str = OLLAMA_MODEL,
        top_k_text: int = TOP_K_TEXT,
        top_k_images: int = TOP_K_IMAGES,
    ):
        if llm_backend not in ("ollama", "zerogpu"):
            raise ValueError(f"Unknown llm_backend {llm_backend!r}, expected 'ollama' or 'zerogpu'")
        self.llm_backend = llm_backend
        self.ollama_model = ollama_model
        self.top_k_text = top_k_text
        self.top_k_images = top_k_images

        client = chromadb.PersistentClient(path=str(chroma_dir))
        self.text_collection = client.get_collection(TEXT_COLLECTION)
        self.image_collection = client.get_collection(IMAGE_COLLECTION)

        self.text_model = SentenceTransformer(TEXT_MODEL_NAME)

        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        self.clip_model.eval()

    def _embed_text_query(self, query: str) -> list[float]:
        return self.text_model.encode([query], convert_to_numpy=True)[0].tolist()

    def _embed_image_query(self, query: str) -> list[float]:
        inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            output = self.clip_model.get_text_features(**inputs)
        features = _pooled(output)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features[0].tolist()

    def retrieve(self, query: str) -> tuple[list[TextSource], list[ImageSource]]:
        """Embed the query for each modality and pull the top-k nearest records."""
        text_embedding = self._embed_text_query(query)
        text_results = self.text_collection.query(
            query_embeddings=[text_embedding], n_results=self.top_k_text
        )
        text_sources = [
            TextSource(
                text=doc,
                section=meta.get("section", ""),
                subsection=meta.get("subsection", ""),
                distance=dist,
            )
            for doc, meta, dist in zip(
                text_results["documents"][0],
                text_results["metadatas"][0],
                text_results["distances"][0],
            )
        ]

        image_embedding = self._embed_image_query(query)
        image_results = self.image_collection.query(
            query_embeddings=[image_embedding], n_results=self.top_k_images
        )
        image_sources = [
            ImageSource(
                image_path=meta.get("image_path", ""),
                description=doc,
                section=meta.get("section", ""),
                subsection=meta.get("subsection", ""),
                distance=dist,
            )
            for doc, meta, dist in zip(
                image_results["documents"][0],
                image_results["metadatas"][0],
                image_results["distances"][0],
            )
        ]
        return text_sources, image_sources

    def _build_prompt(self, query: str, text_sources, image_sources) -> str:
        blocks = []
        for i, src in enumerate(text_sources, 1):
            header = " > ".join(b for b in (src.section, src.subsection) if b)
            blocks.append(f"[Text {i} | {header}]\n{src.text}")
        for i, src in enumerate(image_sources, 1):
            header = " > ".join(b for b in (src.section, src.subsection) if b)
            blocks.append(f"[Image {i} | {header} | file: {src.image_path}]\n{src.description}")
        context = "\n\n".join(blocks) if blocks else "(no relevant context found)"
        return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    def _generate(self, user_prompt: str) -> str:
        if self.llm_backend == "zerogpu":
            return _zerogpu_generate(SYSTEM_PROMPT, user_prompt)

        response = ollama.chat(
            model=self.ollama_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["message"]["content"]

    def answer(self, query: str) -> RagResult:
        """Retrieve context for `query` and generate a grounded answer."""
        text_sources, image_sources = self.retrieve(query)
        prompt = self._build_prompt(query, text_sources, image_sources)
        answer_text = self._generate(prompt)
        return RagResult(answer=answer_text, text_sources=text_sources, image_sources=image_sources)
