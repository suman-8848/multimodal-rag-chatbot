---
title: Multimodal RAG Chatbot
emoji: 🔌
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.22.0
app_file: app/app.py
python_version: "3.12"
pinned: false
license: mit
short_description: Multimodal RAG chatbot with text+image retrieval and grounded, cited answers.
---

# Multimodal RAG Chatbot

A multimodal retrieval-augmented chatbot built over an internal utility-industry reference
document ("MR Tips & Tricks" — pole attachment make-ready rules), combining text and image
retrieval so questions can be answered using both written rules and the field-photo examples
embedded in the source doc.

No paid API keys are used anywhere — all embedding models are open-weight and run locally.

## Project structure

```
data/
  raw/          source .docx
  images/       images extracted from the source doc during ingestion
  chroma_db/    persistent Chroma vector store (gitignored, rebuilt by ingest/build_index.py)
ingest/         ingestion pipeline (parsing, chunking, embedding, indexing)
rag/            retrieval + generation pipeline (pipeline.py) and a CLI test harness (cli.py)
app/            Gradio chat UI (app.py)
eval/           retrieval evaluation set + harness (eval_set.json, evaluate.py)
```

## Phase roadmap

1. **Ingestion** (done) — parse the docx into text chunks and image+description pairs,
   embed with `sentence-transformers/all-MiniLM-L6-v2` (text) and
   `openai/clip-vit-base-patch32` (images), persist to ChromaDB.
2. **Retrieval** (done) — embed the query for each modality, pull top-k matches from both
   Chroma collections, and generate a grounded answer with a local Ollama model. See
   `rag/pipeline.py` (the `RagPipeline` class) and `rag/cli.py` for a terminal test harness.
3. **App** (done) — `app/app.py`, a Gradio `ChatInterface` over `RagPipeline` showing source
   images and section citations inline with each answer.
4. **Eval** (done) — `eval/evaluate.py` measures retrieval precision@k/recall@k against a
   15-question hand-labeled ground-truth set (`eval/eval_set.json`). See results below.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Running ingestion

```bash
python ingest/build_index.py
```

This parses `data/raw/MR Tips & Tricks.docx`, extracts images to `data/images/`, and builds
two collections in `data/chroma_db/`:

- `text_chunks` — chunked document text (MiniLM embeddings)
- `image_chunks` — extracted images paired with their surrounding description text (CLIP
  embeddings)

The script is idempotent — each run rebuilds both collections from scratch.

## Running retrieval + generation

Requires [Ollama](https://ollama.com) installed and running locally, with a model pulled
(e.g. `ollama pull llama3.2:3b` — see `rag/pipeline.py`'s `OLLAMA_MODEL` for the default
in use and notes on picking a model for your hardware).

```bash
python -m rag.cli                    # interactive
python -m rag.cli "your question"    # one-shot
```

This loads `RagPipeline` (`rag/pipeline.py`), which embeds your query with MiniLM (for
`text_chunks`) and CLIP's text encoder (for `image_chunks`, into the same embedding space
CLIP indexed the images in), retrieves the top-k matches from each collection, and asks the
configured Ollama model to answer using only that retrieved context. Prints the answer plus
the text/image sources used, so you can verify groundedness before wiring up a UI.

## Running the chat UI

```bash
python -m app.app
```

Launches a Gradio `ChatInterface` at `http://127.0.0.1:7860` with four example questions,
grounded answers, and the source images + section citations used for each answer rendered
inline below it.

## Evaluation

`eval/eval_set.json` is a 15-question, hand-labeled ground-truth set built directly from the
indexed document content (question → expected section[/subsection], plus expected image
IDs for the 8 questions that have a genuinely relevant source image). `eval/evaluate.py`
runs each question through `RagPipeline.retrieve()` (no LLM calls — retrieval only) and scores
it against ground truth:

- **Text relevance** = retrieved chunk's section (and subsection, where labeled) matches
  expected. Ground truth is labeled at section granularity, not exact chunk, so text
  recall@k is a hit-rate: 1 if *any* top-k chunk matches, else 0.
- **Image relevance** = retrieved image's ID is in the expected list. Labeled at the exact
  image level, so image recall@k is the standard fraction-of-relevant-items-found.

```bash
python -m eval.evaluate
```

### Results (n=15 questions, 8 with image ground truth)

| Metric | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| Text Precision@k | 0.73 | 0.50 | 0.38 | 0.33 |
| Text Recall@k (hit-rate) | 0.73 | 0.87 | 0.87 | 0.87 |

| Metric | k=1 | k=2 |
|---|---|---|
| Image Precision@k | 0.25 | 0.25 |
| Image Recall@k | 0.19 | 0.38 |

Text retrieval is solid — the correct section is found in the top-4 for 87% of questions.
Image retrieval is much weaker (recall@2 of 0.38) — see limitations below for why, verified
by inspecting the actual retrieved/missed images rather than guessing.

### Known limitations

- **Purely-visual sections are invisible to text search.** Both text misses (`q14`, `q15`)
  target sections (`COMM TAGS:`, and the `Comm Attachment based on line angle` subsection)
  that contain literally zero body text in the source doc — only images. No text chunk was
  ever indexed for them, so no amount of retrieval tuning fixes this; it's a structural gap
  between "what has a chunk" and "what a user might ask about." Fixing it would mean
  generating synthetic captions for these images (e.g. via a vision-language model) and
  indexing those as text too.
- **CLIP struggles with non-photographic images.** `img_001` (a UI screenshot with dense
  text annotations) and `img_003`/`img_004` (stock clip-art infographics — "Types of
  Insulators," "Anatomy of a Power Pole" — confirmed byte-identical to the source docx's
  embedded images, not an extraction bug) were never retrieved for their matching questions.
  CLIP's training distribution (natural photos with captions) doesn't generalize well to
  screenshots or cartoon-style diagrams, which make up a meaningful fraction of this
  document's images.
- **A couple of images act as generic "attractors."** `img_006` and `img_008` — both
  visually dense, annotated field photos with a worker in a high-vis vest and colored
  callout boxes — were retrieved in the top-2 for 5 of the 8 image questions, including
  ones they have nothing to do with (comm tags, triplex/transformer replacement). With only
  12 images total, a couple of visually "busy" photos can dominate CLIP's nearest-neighbor
  results regardless of true relevance. This would likely wash out with a larger, more
  diverse image index.
- **Small sample size.** 15 questions (8 image) is enough to surface real, reproducible
  patterns (confirmed above by inspecting the actual images), but not enough for the
  precision/recall numbers themselves to be statistically robust — read them directionally.
