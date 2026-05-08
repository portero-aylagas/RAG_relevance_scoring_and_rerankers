# Trustworthy AI RAG Pipeline

This project builds a retrieval-augmented generation pipeline for Trustworthy AI content. It combines a PDF document and a podcast transcript, indexes both sources in Pinecone, retrieves relevant chunks, scores them with an LLM, and generates grounded answers.

The goal is to show how a simple vector search workflow can be improved with metadata, relevance scoring, and source-aware retrieval.

## Project Highlights

- Converts a PDF into Markdown for section-aware chunking.
- Transcribes podcast audio with OpenAI Whisper.
- Stores generated Markdown and transcript artifacts in `data/processed/`.
- Chunks PDF and transcript content with source metadata.
- Creates the Pinecone index automatically if it does not exist.
- Runs baseline semantic retrieval.
- Applies LLM-based relevance scoring to explain and reorder retrieved chunks.
- Generates concise answers from the highest-ranked context.

## Files

- `relevance_scoring_rerankers.py`: main Python workflow.
- `requirements.txt`: required dependencies.
- `data/`: raw source files.
- `data/processed/`: generated Markdown and transcript files.
- `lab_summary.md`: short project summary.
- `lab_source/`: original project brief.

Key generated files:

- `data/processed/ai_hleg_ethics_guidelines_for_trustworthy_ai-en_87F84A41-A6E8-F38C-BFF661481B40077B_60419.md`: Markdown generated from the PDF.
- `data/processed/The_Blueprint_For_Trustworthy_AI_transcript.txt`: podcast transcript generated with Whisper.
- `data/processed/The_Blueprint_For_Trustworthy_AI_transcript_segments.txt`: timestamped transcript segments.

## Setup

This project needs Python packages, API keys, and `ffmpeg` for audio processing.

### 1. Install Python Dependencies

```bash
./.conda/bin/pip install -r requirements.txt
```

### 2. Install ffmpeg

`pydub` uses `ffmpeg` to read and convert audio files such as `.m4a`.

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install ffmpeg
```

On macOS with Homebrew:

```bash
brew install ffmpeg
```

On Windows with Chocolatey:

```powershell
choco install ffmpeg
```

Check that it is available:

```bash
ffmpeg -version
```

### 3. Add API Keys

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Then fill in your API keys:

```bash
OPENAI_API_KEY=...
PINECONE_API_KEY=...
```

The OpenAI key is used for embeddings, answer generation, relevance scoring, and audio transcription. The Pinecone key is used for vector storage and retrieval.

Optional Pinecone settings can also be added to `.env`:

```bash
PINECONE_INDEX_NAME=trustworthy-ai
PINECONE_NAMESPACE=trustworthy-ai
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

If these optional values are not set, the script uses the defaults shown above.

## Run

```bash
./.conda/bin/python relevance_scoring_rerankers.py
```

The script runs the full pipeline:

1. Converts the English Trustworthy AI PDF to Markdown in `data/processed/`.
2. Transcribes the Trustworthy AI podcast audio and stores transcript files in `data/processed/`.
3. Chunks the PDF and transcript with metadata.
4. Creates the Pinecone index automatically if needed.
5. Uploads embeddings to Pinecone.
6. Runs baseline vector retrieval.
7. Scores retrieved chunks with an LLM.
8. Shows metadata-filtered retrieval.
9. Generates a simple RAG answer from the top scored chunks.
10. Prints comparison notes for manual evaluation.

## Pipeline Output

When the script runs successfully, it prints each stage of the pipeline and shows example retrieval results. A recent run produced:

- PDF chunks created: `228`
- Podcast transcript chunks created: `19`
- Total chunks uploaded to Pinecone: `247`
- Vector index: `trustworthy-ai`
- Namespace: `trustworthy-ai`

The script then runs a few sample questions through baseline vector search, LLM relevance scoring, metadata filtering, and answer generation.

## Tech Stack

- Python
- LangChain
- OpenAI embeddings and chat models
- OpenAI Whisper for speech-to-text
- Pinecone for vector storage
- PyMuPDF4LLM for PDF-to-Markdown conversion
- pydub and ffmpeg for audio processing

## Example Questions

Question:

```text
What are the key requirements for trustworthy AI?
```

The top PDF chunks describe trustworthy AI as lawful, ethical, and robust. The relevance scoring step keeps those chunks at the top and adds a short reason for why they answer the question.

Answer summary:

```text
Trustworthy AI should be lawful, ethical, and robust. The guidelines also describe seven practical requirements: human agency and oversight, technical robustness and safety, privacy and data governance, transparency, diversity/non-discrimination/fairness, environmental and societal well-being, and accountability.
```

Question:

```text
How do the guidelines define human agency and oversight?
```

The retrieved chunks cover user autonomy and oversight mechanisms such as human-in-the-loop, human-on-the-loop, and human-in-command.

Answer summary:

```text
Human agency means people should be able to make informed and autonomous decisions when interacting with AI systems. Human oversight means governance mechanisms should help prevent AI systems from undermining human autonomy or causing harm, including approaches such as human-in-the-loop, human-on-the-loop, and human-in-command.
```

Question:

```text
What does the transparency requirement say about traceability, explainability, and communication?
```

The retrieved chunks include the `1.4 Transparency` section and related context on explicability.

Answer summary:

```text
Traceability means documenting the data sets, data gathering, data labelling, and algorithms that produce AI system decisions. Explainability means affected people should be able to understand decisions where possible. Communication means users should be told about the AI system's purpose, capabilities, limitations, and when they are interacting with a machine.
```

## Current Scope

This version processes:
- the English Trustworthy AI PDF
- the Trustworthy AI podcast audio transcript generated from `data/The_Blueprint_For_Trustworthy_AI.m4a`

## Notes

The project is intentionally kept as a single readable Python workflow instead of a larger application. The focus is on making the retrieval pipeline easy to inspect: source preparation, chunking, vector indexing, retrieval, relevance scoring, and answer generation all happen in one place.
