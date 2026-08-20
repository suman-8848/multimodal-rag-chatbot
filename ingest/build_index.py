"""Ingestion pipeline for the multimodal RAG chatbot.

Walks every source file in `data/raw/` (currently `.docx` and `.pdf`) and produces two
aligned representations per file:
  - a linear text stream (every paragraph/table row for docx, every page for pdf), tagged
    with its section/subsection heading, chunked and embedded with MiniLM for general text
    retrieval
  - image + associated-description pairs, embedded with CLIP for image retrieval:
      * docx: each embedded image, described by the text accumulated since the last
        heading or the last image, whichever is closer (docx images have no formal
        captions in these source docs)
      * pdf: each *page*, rendered whole as an image and described by that page's
        extracted text. PDF pages here are dense reference charts assembled from dozens
        of small vector fragments (arrows, icons, table borders) rather than a few
        meaningful embedded photos, so extracting those fragments individually would
        produce junk; a full-page render is what's actually useful to retrieve and show.

Both are persisted to a local ChromaDB store as two collections, tagged with source_doc
so citations can distinguish which document an answer came from.
"""

import re

import fitz  # PyMuPDF
import torch
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

import chromadb

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
IMAGES_DIR = ROOT / "data" / "images"
CHROMA_DIR = ROOT / "data" / "chroma_db"

TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

TEXT_COLLECTION = "text_chunks"
IMAGE_COLLECTION = "image_chunks"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
IMAGE_BATCH_SIZE = 8
PDF_RENDER_DPI = 150

SUPPORTED_SUFFIXES = {".docx", ".pdf"}

# Heading styles that bound a section in a docx. A heading paragraph with no text (a
# formatting artifact where an image landed on a Heading-styled empty line) must NOT
# reset section tracking, so callers check `text` before using this.
SECTION_STYLES = {"Heading 2": "section", "Heading 3": "subsection"}


def _slug(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_")


def _resolve_image(document, drawing_element):
    """Resolve a w:drawing element's embedded image to (bytes, content_type)."""
    blip = drawing_element.find(".//" + qn("a:blip"))
    if blip is None:
        return None
    r_id = blip.get(qn("r:embed"))
    if r_id is None:
        return None
    part = document.part.related_parts[r_id]
    return part.blob, part.content_type


def extract_docx_records(docx_path: Path):
    """Walk a docx body in order, producing (full_text_stream, image_records) for it."""
    document = Document(str(docx_path))
    body = document.element.body
    source_doc = docx_path.name
    slug = _slug(docx_path)

    section = None
    subsection = None
    full_text_accum = []
    pending_image_block = []
    full_text_stream = []
    image_records = []
    order_index = 0
    image_counter = 0

    def flush_full_block():
        nonlocal full_text_accum, order_index
        if full_text_accum:
            full_text_stream.append({
                "text": "\n".join(full_text_accum),
                "section": section,
                "subsection": subsection,
                "order_index": order_index,
                "source_doc": source_doc,
            })
            order_index += 1
            full_text_accum = []

    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            para = Paragraph(child, document)
            text = para.text.strip()
            style = para.style.name if para.style else None
            drawings = child.findall(".//" + qn("w:drawing"))

            if style in SECTION_STYLES and text and not drawings:
                flush_full_block()
                pending_image_block = []
                if SECTION_STYLES[style] == "section":
                    section = text
                    subsection = None
                else:
                    subsection = text
                continue

            if drawings:
                for drawing in drawings:
                    resolved = _resolve_image(document, drawing)
                    if resolved is None:
                        continue
                    image_bytes, content_type = resolved
                    image_counter += 1
                    ext = ".png" if "png" in content_type else ".jpg"
                    image_path = IMAGES_DIR / f"{slug}_img_{image_counter:03d}{ext}"
                    image_path.write_bytes(image_bytes)

                    description_parts = pending_image_block + ([text] if text else [])
                    description = "\n".join(p for p in description_parts if p).strip()

                    image_records.append({
                        "id": f"{slug}_img_{image_counter:03d}",
                        "image_path": str(image_path.relative_to(ROOT)).replace("\\", "/"),
                        "description": description or "(no surrounding text found in source document)",
                        "section": section,
                        "subsection": subsection,
                        "order_index": order_index,
                        "source_doc": source_doc,
                    })
                pending_image_block = []
                if text:
                    full_text_accum.append(text)
                order_index += 1
                continue

            if text:
                full_text_accum.append(text)
                pending_image_block.append(text)

        elif child.tag == qn("w:tbl"):
            table = Table(child, document)
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                row_text = " | ".join(cells)
                if row_text:
                    full_text_accum.append(row_text)
                    pending_image_block.append(row_text)

    flush_full_block()
    return full_text_stream, image_records


def extract_pdf_records(pdf_path: Path):
    """Render each page of a pdf, producing (full_text_stream, image_records) for it."""
    source_doc = pdf_path.name
    slug = _slug(pdf_path)
    full_text_stream = []
    image_records = []

    with fitz.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            text = page.get_text().strip()
            subsection = f"Page {page_number}"

            if text:
                full_text_stream.append({
                    "text": text,
                    "section": None,
                    "subsection": subsection,
                    "order_index": page_index,
                    "source_doc": source_doc,
                })

            pix = page.get_pixmap(dpi=PDF_RENDER_DPI)
            image_path = IMAGES_DIR / f"{slug}_p{page_number:02d}.png"
            pix.save(str(image_path))

            image_records.append({
                "id": f"{slug}_p{page_number:02d}",
                "image_path": str(image_path.relative_to(ROOT)).replace("\\", "/"),
                "description": text or "(no extracted text on this page)",
                "section": None,
                "subsection": subsection,
                "order_index": page_index,
                "source_doc": source_doc,
            })

    return full_text_stream, image_records


def extract_all_records(raw_dir: Path):
    """Extract and concatenate records from every supported file in `raw_dir`."""
    full_text_stream = []
    image_records = []
    for path in sorted(raw_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        print(f"Parsing {path.name}...")
        if path.suffix.lower() == ".docx":
            text_stream, images = extract_docx_records(path)
        else:
            text_stream, images = extract_pdf_records(path)
        print(f"  -> {len(text_stream)} text blocks, {len(images)} images")
        full_text_stream.extend(text_stream)
        image_records.extend(images)
    return full_text_stream, image_records


def chunk_text_stream(full_text_stream):
    """Split the linear text stream into embedding-sized chunks, anchored to their heading."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for entry in full_text_stream:
        section_label = " > ".join(b for b in (entry["section"], entry["subsection"]) if b)
        header = " | ".join(b for b in (entry["source_doc"], section_label) if b)
        prefixed = f"{header}\n\n{entry['text']}" if header else entry["text"]
        for piece in splitter.split_text(prefixed):
            chunks.append({
                "id": f"txt_{len(chunks):04d}",
                "text": piece,
                "section": entry["section"],
                "subsection": entry["subsection"],
                "chunk_index": len(chunks),
                "order_index": entry["order_index"],
                "source_doc": entry["source_doc"],
            })
    return chunks


def embed_text_chunks(chunks, model_name=TEXT_MODEL_NAME):
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


def embed_images(image_records, model_name=CLIP_MODEL_NAME):
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()

    embeddings = []
    paths = [ROOT / rec["image_path"] for rec in image_records]
    for i in range(0, len(paths), IMAGE_BATCH_SIZE):
        batch_paths = paths[i:i + IMAGE_BATCH_SIZE]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            output = model.get_image_features(**inputs)
        # newer transformers versions return BaseModelOutputWithPooling (embedding in
        # .pooler_output) instead of a bare tensor
        features = output.pooler_output if hasattr(output, "pooler_output") else output
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        embeddings.extend(features.cpu().numpy().tolist())
    return embeddings


def build_chroma_index(text_chunks, text_embeddings, image_records, image_embeddings):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    for name in (TEXT_COLLECTION, IMAGE_COLLECTION):
        try:
            client.delete_collection(name)
        except Exception:
            pass

    text_coll = client.create_collection(TEXT_COLLECTION, metadata={"hnsw:space": "cosine"})
    text_coll.add(
        ids=[c["id"] for c in text_chunks],
        embeddings=text_embeddings,
        documents=[c["text"] for c in text_chunks],
        metadatas=[{
            "section": c["section"] or "",
            "subsection": c["subsection"] or "",
            "chunk_index": c["chunk_index"],
            "source_doc": c["source_doc"],
        } for c in text_chunks],
    )

    image_coll = client.create_collection(IMAGE_COLLECTION, metadata={"hnsw:space": "cosine"})
    image_coll.add(
        ids=[r["id"] for r in image_records],
        embeddings=image_embeddings,
        documents=[r["description"] for r in image_records],
        metadatas=[{
            "image_path": r["image_path"],
            "section": r["section"] or "",
            "subsection": r["subsection"] or "",
            "order_index": r["order_index"],
            "source_doc": r["source_doc"],
        } for r in image_records],
    )
    return text_coll, image_coll


def main():
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        raise FileNotFoundError(f"No source documents found in {RAW_DIR}")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    full_text_stream, image_records = extract_all_records(RAW_DIR)
    print(f"\nExtracted {len(full_text_stream)} text blocks and {len(image_records)} images total.")

    text_chunks = chunk_text_stream(full_text_stream)
    print(f"Split into {len(text_chunks)} text chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    print("Embedding text chunks with MiniLM...")
    text_embeddings = embed_text_chunks(text_chunks)

    print("Embedding images with CLIP...")
    image_embeddings = embed_images(image_records)

    print("Writing to ChromaDB...")
    text_coll, image_coll = build_chroma_index(text_chunks, text_embeddings, image_records, image_embeddings)

    print()
    print("=" * 70)
    print(f"text_chunks collection:  {text_coll.count()} records")
    print(f"image_chunks collection: {image_coll.count()} records")
    print(f"Text embedding dim: {len(text_embeddings[0])} | Image embedding dim: {len(image_embeddings[0])}")
    print("=" * 70)

    print("\nSample image records:\n")
    for rec in image_records[:5]:
        print(f"[{rec['id']}] {rec['source_doc']} | section: {rec['section']!r} > subsection: {rec['subsection']!r}")
        print(f"  image: {rec['image_path']}")
        desc = rec["description"][:300].replace("\n", " | ")
        print(f"  description: {desc}")
        print()


if __name__ == "__main__":
    main()
