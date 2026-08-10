# How the Multimodal RAG Chatbot Works

*A plain-language walkthrough — no engineering background required. A more visual,
designed version of this same explainer is linked from the main [README](../README.md).*

## The problem

Imagine a 200-page work manual — bullet-point rules mixed with real field photos. Someone
asks: *"What should I do if a pole is failing sound and probe?"* A person has to skim the
whole thing to find the one paragraph that answers it, and maybe miss the photo three pages
later that illustrates it.

This project builds a chatbot that does that skimming instantly — and can point to the
exact photo, not just the text, when a photo is what actually answers the question.

## The idea in one breath

Read the manual once, ahead of time, and organize it so any question can find the right
paragraph *and* the right photo in a fraction of a second. Then hand just those relevant
pieces to an AI and say: **"answer using only this — don't make anything up."**

```
1. Read the manual        Split it into sections and photos, once, ahead of time
        ↓
2. Make it searchable     Give every paragraph and photo a "meaning fingerprint"
        ↓
3. Ask a question         Find the paragraphs and photos whose fingerprints match closest
        ↓
4. Write the answer       An AI answers using only what was found — nothing else
        ↓
5. Show the proof         Answer, plus the actual paragraph and photo it came from
```

## Step by step

### 1. Reading the manual

The source is a real Word document. A script opens it and walks through it top to bottom,
keeping track of which section heading each paragraph falls under, and pulling out every
embedded photo along with the paragraphs written just before it — since the document never
labels photos with proper captions, "the text right before this photo" is the best
available description.

*Like a very patient intern who reads the whole binder and sticky-notes every section and
photo.*

### 2. Giving everything a "meaning fingerprint"

Computers can't search by *meaning* out of the box — only by matching exact words. So every
paragraph and every photo gets converted into a long list of numbers (called an
**embedding**) that captures what it's *about*, not just what words it uses. Two paragraphs
about the same topic end up with similar number-lists, even if they don't share a single
word.

Text and photos need two different translators for this — one trained on language, one
trained on images — so they end up in two separate "filing cabinets":

- **Cabinet 1 — Text passages.** Every chunk of manual text, fingerprinted by a small
  language model trained just for this kind of matching.
- **Cabinet 2 — Photos.** Every extracted photo, fingerprinted by a model trained on both
  images and language together — so a plain English question can find a matching photo
  directly.

*Like a librarian who can tell two books are related by their subject, not by scanning for
matching words on the spine.*

### 3. Finding the closest matches

When someone asks a question, it gets the same "meaning fingerprint" treatment, then both
filing cabinets are searched for the passages and photos whose fingerprints are numerically
closest to the question's. This takes a fraction of a second even though it's comparing
against everything in the manual.

### 4. Writing the answer — with a leash on

The best-matching passages and photo descriptions are handed to an AI language model with a
strict instruction: *answer using only this material, and say "I don't know" if it isn't in
here.* This is what keeps the chatbot from confidently making things up — it's never just
asked to "answer from memory."

*Like handing someone the three most relevant pages of the manual and saying "answer using
only these," instead of asking them to recite the whole manual from memory.*

### 5. Showing the receipts

The chat window shows the written answer *and* the actual source photos and the section
they came from — so anyone reading the answer can check it against the original material
instead of just trusting it.

## How well does it actually work?

This wasn't just built and assumed to work — it was tested against 15 questions with known
correct answers, written by hand from the real document.

| Verdict | What was tested | Result |
|---|---|---|
| **Strong** | Finding the right **text** passage | The correct section turned up in the top few results for 87% of test questions (13 of 15) |
| **Weaker** | Finding the right **photo** | Meaningfully less reliable — some photos are screenshots or clip-art diagrams rather than real field photos, and the "photo meaning" model struggles more with those than with straightforward text. Confirmed by opening the actual retrieved photos and checking them by eye, not just trusting the numbers |
| **Known gap** | Photo-only sections | A couple of sections in the manual have no written text at all under that heading. A question about one of those can never be answered from text search, because there's no text to find — a real, honest gap, not a bug to "fix" with more tuning |

## Where it lives

The chatbot runs on [Hugging Face Spaces](https://huggingface.co/spaces), a free hosting
service for this kind of project — at $0/month, with no paid API keys anywhere in the
system. The tradeoff for "free" is that it runs on shared, modest hardware, so it answers
in a handful of seconds rather than instantly.

**Why it's private right now:** the manual this chatbot was built on is a real internal
reference document — it names a real company and includes internal job numbers. The live
chatbot is access-restricted for that reason, even though the code, design, and this
explainer are shared openly.

## A few terms, plainly

- **Embedding** — the "meaning fingerprint": a long list of numbers a model produces for a
  piece of text or an image, positioned so similar meanings end up numerically close
  together.
- **RAG (Retrieval-Augmented Generation)** — the overall technique: look up relevant
  material first ("retrieval"), then have an AI write an answer using only that material
  ("generation") — instead of asking the AI to answer from memory alone.
- **Vector database** — the "filing cabinet" that stores every fingerprint and can
  instantly find the closest matches to a new one. This project uses one called ChromaDB,
  running locally — no separate server to manage.
- **LLM (Large Language Model)** — the AI that actually writes the final answer in natural
  sentences, once it's been handed the relevant material to work from.
- **Grounded answer** — an answer built strictly from retrieved material, with the model
  instructed to say "I don't know" rather than guess — the opposite of an AI making
  something up.
