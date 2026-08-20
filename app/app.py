"""Gradio chat UI for the utility make-ready multimodal RAG chatbot.

Wraps rag.pipeline.RagPipeline in a gr.ChatInterface: each answer is grounded
in retrieved text chunks and images from the ChromaDB index, with the
source images and their document/section citations rendered inline below the answer.
"""

# Must be the first import in the process — see rag/pipeline.py for why.
try:
    import spaces  # noqa: F401
except ImportError:
    pass

import gradio as gr

from rag.pipeline import ROOT, RagPipeline

EXAMPLES = [
    "What should I do if a pole is failing sound and probe?",
    "How do we create a service request?",
    "What is the required vertical clearance over a swimming pool?",
    "What is the ground clearance for a 120/240V triplex service drop?",
]

# Hardcoded rather than left to RAG_LLM_BACKEND's default: this file is
# specifically the deployed-app entry point, and should always run the
# ZeroGPU backend regardless of Space env var configuration. Use rag/cli.py
# for local development against Ollama instead.
pipeline = RagPipeline(llm_backend="zerogpu")


def _section_label(source_doc: str, section: str, subsection: str) -> str:
    section_part = " > ".join(b for b in (section, subsection) if b)
    label = " | ".join(b for b in (source_doc, section_part) if b)
    return label or "Unknown section"


def respond(message: str, history: list) -> str:
    result = pipeline.answer(message)
    parts = [result.answer.strip()]

    if result.image_sources:
        parts.append("\n---\n**Source images:**")
        for src in result.image_sources:
            image_url = f"/gradio_api/file={(ROOT / src.image_path).as_posix()}"
            parts.append(f"\n*{_section_label(src.source_doc, src.section, src.subsection)}*\n\n![source]({image_url})")

    if result.text_sources:
        seen = []
        for src in result.text_sources:
            label = _section_label(src.source_doc, src.section, src.subsection)
            if label not in seen:
                seen.append(label)
        parts.append("\n\n**Referenced sections:** " + "; ".join(seen))

    return "\n".join(parts)


demo = gr.ChatInterface(
    fn=respond,
    title="Make-Ready Reference Q&A",
    description=(
        "Ask about make-ready procedures, service requests, or NESC clearance rules. Answers "
        "are grounded in the source documents, with relevant photos/charts and citations shown "
        "below each answer."
    ),
    examples=EXAMPLES,
)

demo.queue()

if __name__ == "__main__":
    demo.launch(allowed_paths=[str(ROOT / "data" / "images")])
