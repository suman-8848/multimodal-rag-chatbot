"""Retrieval-augmented generation over the utility make-ready reference-doc ChromaDB index.

Retrieves relevant text chunks (MiniLM) and images (CLIP, queried via its text
encoder into the same embedding space as the indexed images) from the
collections built by ingest/build_index.py, then generates a grounded answer
using one of two interchangeable LLM backends (see RAG_LLM_BACKEND below):

- "ollama"  — a local Ollama server. Used for local development.
- "zerogpu" — a small open model (Qwen2.5-3B-Instruct) run in-process via
  llama.cpp (llama-cpp-python), on HF Spaces' free CPU tier (ZeroGPU's actual
  GPU dispatch doesn't work on this account/fleet — see below — so this always
  runs on CPU regardless of the "zerogpu" name, kept for its hardware-tier
  meaning on Spaces). A quantized GGUF build is used specifically *because*
  it's CPU-only: llama.cpp's CPU kernels are dramatically faster than eager
  PyTorch generation for token-by-token decoding, which matters a lot on
  Spaces' free 2 vCPU hardware. Chosen over Ollama-on-CPU (same speed problem,
  plus no Ollama server available on Spaces) and over HF's paid Inference
  Providers API (free tier is a thin $0.10/month credit) — this stays
  genuinely free. See README for details.
"""

# `spaces` must be imported before torch (directly or transitively via
# chromadb/sentence_transformers/transformers/ollama) — HF's ZeroGPU CUDA
# patching only takes effect if it initializes first. Importing it later
# produced "RuntimeError: No CUDA GPUs are available" deep inside spaces'
# own worker init on the deployed Space, even with correctly-decorated code.
try:
    import spaces

    _HAS_SPACES = True
except ImportError:
    _HAS_SPACES = False

import os
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import ollama
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

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

# Ungated (Apache-2.0), official Qwen-published GGUF quantization — no license
# click-through or HF_TOKEN needed. Q4_K_M is the standard "good quality, real
# speedup" quantization level (~2GB vs. ~6GB for the fp16 weights). Only used
# by the "zerogpu" backend.
ZEROGPU_GGUF_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
ZEROGPU_GGUF_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
ZEROGPU_CONTEXT_TOKENS = 4096

DEFAULT_LLM_BACKEND = os.environ.get("RAG_LLM_BACKEND", "ollama")

TOP_K_TEXT = 4
TOP_K_IMAGES = 2

# Image "descriptions" for PDF pages are the page's full extracted text (up to ~2,400
# chars for the NESC charts) — useful to store for embedding/context, but dumping all of
# it into the LLM prompt duplicates what text retrieval already surfaced and slows CPU
# generation on the deployed Space for no accuracy benefit. Capped here, at prompt-build
# time, so the full description is still stored/embeddable in Chroma.
IMAGE_DESCRIPTION_PROMPT_CHARS = 400

# Kept modest deliberately: generation runs on CPU on the deployed Space (see below), and
# grounded answers to these reference-doc questions don't need much room. Shorter cap
# also nudges the model toward direct answers instead of padding/rambling.
MAX_NEW_TOKENS = 300

SYSTEM_PROMPT = (
    "You are an assistant answering questions about utility make-ready reference "
    "documents, covering make-ready standard operating procedures, service request "
    "procedures, and NESC clearance requirements. Answer ONLY using the provided "
    "context. If the context doesn't contain the answer, say you don't know "
    "rather than guessing. Be concise and direct — a few sentences unless the "
    "question genuinely requires more."
)


@dataclass
class TextSource:
    text: str
    section: str
    subsection: str
    distance: float
    source_doc: str = ""


@dataclass
class ImageSource:
    image_path: str
    description: str
    section: str
    subsection: str
    distance: float
    source_doc: str = ""


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


# Generation runs on CPU deliberately, without real @spaces.GPU dispatch.
# ZeroGPU's actual GPU worker allocation proved unreliable when deployed
# (consistently "RuntimeError: No CUDA GPUs are available" deep inside
# spaces' own worker init, even after loading at true module scope with an
# explicit .to("cuda") call and fixing spaces/torch import order — all
# documented fixes for that error). Generation itself just runs on whatever
# CPU the container provides — small enough (3B) to be tolerable there.
#
# HF's platform separately *requires* at least one @spaces.GPU-decorated
# function to exist for ZeroGPU hardware to be accepted at all (Space
# startup otherwise fails outright with "No @spaces.GPU function detected
# during startup") — and ZeroGPU is what makes free personal-account Gradio
# hosting possible at all right now. _gpu_probe below exists solely to
# satisfy that platform check; it's never called on the real request path.
# See README known limitations.
if _HAS_SPACES:

    @spaces.GPU
    def _gpu_probe():
        return None


_ZEROGPU_LAZY = {}


def _get_zerogpu_llm():
    if not _ZEROGPU_LAZY:
        from llama_cpp import Llama

        _ZEROGPU_LAZY["llm"] = Llama.from_pretrained(
            repo_id=ZEROGPU_GGUF_REPO,
            filename=ZEROGPU_GGUF_FILENAME,
            n_ctx=ZEROGPU_CONTEXT_TOKENS,
            n_threads=os.cpu_count(),
            verbose=False,
        )
    return _ZEROGPU_LAZY["llm"]


def _zerogpu_generate(system_prompt: str, user_prompt: str) -> str:
    llm = _get_zerogpu_llm()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
    )
    return response["choices"][0]["message"]["content"].strip()


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
                source_doc=meta.get("source_doc", ""),
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
                source_doc=meta.get("source_doc", ""),
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
            section_label = " > ".join(b for b in (src.section, src.subsection) if b)
            header = " | ".join(b for b in (src.source_doc, section_label) if b)
            blocks.append(f"[Text {i} | {header}]\n{src.text}")
        for i, src in enumerate(image_sources, 1):
            section_label = " > ".join(b for b in (src.section, src.subsection) if b)
            header = " | ".join(b for b in (src.source_doc, section_label) if b)
            description = src.description
            if len(description) > IMAGE_DESCRIPTION_PROMPT_CHARS:
                description = description[:IMAGE_DESCRIPTION_PROMPT_CHARS].rstrip() + "..."
            blocks.append(f"[Image {i} | {header} | file: {src.image_path}]\n{description}")
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
