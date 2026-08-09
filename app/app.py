"""Gradio chat UI for the MR Tips & Tricks multimodal RAG chatbot.

Wraps rag.pipeline.RagPipeline in a gr.ChatInterface: each answer is grounded
in retrieved text chunks and images from the Phase 1 ChromaDB index, with the
source images and their section citations rendered inline below the answer.
"""

import gradio as gr

from rag.pipeline import ROOT, RagPipeline

EXAMPLES = [
    "What should I do if a pole is failing sound and probe?",
    "What is the rule for attaching comms on a clean pole with no existing attachers?",
    "What are the rules about drops attached without a cable?",
    "How should secondary risers be attached relative to the transformer?",
]

pipeline = RagPipeline()


def _section_label(section: str, subsection: str) -> str:
    return " > ".join(b for b in (section, subsection) if b) or "Unknown section"


def respond(message: str, history: list) -> str:
    result = pipeline.answer(message)
    parts = [result.answer.strip()]

    if result.image_sources:
        parts.append("\n---\n**Source images:**")
        for src in result.image_sources:
            image_url = f"/gradio_api/file={(ROOT / src.image_path).as_posix()}"
            parts.append(f"\n*{_section_label(src.section, src.subsection)}*\n\n![source]({image_url})")

    if result.text_sources:
        seen = []
        for src in result.text_sources:
            label = _section_label(src.section, src.subsection)
            if label not in seen:
                seen.append(label)
        parts.append("\n\n**Referenced sections:** " + "; ".join(seen))

    return "\n".join(parts)


demo = gr.ChatInterface(
    fn=respond,
    title="MR Tips & Tricks Q&A",
    description=(
        "Ask about utility pole make-ready rules. Answers are grounded in the source "
        "document, with relevant field photos and section citations shown below each answer."
    ),
    examples=EXAMPLES,
)

demo.queue()

if __name__ == "__main__":
    demo.launch(allowed_paths=[str(ROOT / "data" / "images")])
