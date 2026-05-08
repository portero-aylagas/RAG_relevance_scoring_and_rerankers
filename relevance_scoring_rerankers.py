from __future__ import annotations

import os
import time
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trustworthy-ai")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "trustworthy-ai")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
CHAT_MODEL = "gpt-4o-mini"
WHISPER_MODEL = "whisper-1"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
AUDIO_CHUNK_SECONDS = 120

HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]

QUESTIONS = [
    "What are the key requirements for trustworthy AI?",
    "How do the guidelines define human agency and oversight?",
    "What does the transparency requirement say about traceability, explainability, and communication?",
]

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Score how useful the retrieved chunk is for answering the question. "
            "Use 5 if it directly answers, 4 if highly relevant, 3 if partly "
            "relevant, 2 if weakly related, 1 if barely related, and 0 if not relevant.",
        ),
        (
            "human",
            "Question:\n{question}\n\nChunk metadata:\n{metadata}\n\nChunk text:\n{chunk_text}",
        ),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using only the provided context. If the context is not enough, "
            "say that the retrieved chunks do not contain enough information. "
            "Cite chunks with [Chunk 1], [Chunk 2], etc.",
        ),
        ("human", "Question:\n{question}\n\nContext:\n{context}"),
    ]
)


class RelevanceScore(BaseModel):
    score: int = Field(ge=0, le=5)
    reason: str


def require_api_keys() -> None:
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")

    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def find_english_pdf() -> Path:
    for path in sorted(DATA_DIR.glob("*.pdf")):
        filename = path.name.lower()
        if "-en" in filename or "_en" in filename or "english" in filename:
            return path
    raise FileNotFoundError(f"No English PDF found in {DATA_DIR}")


def find_podcast_audio() -> Path:
    audio_extensions = {".m4a", ".mp3", ".wav"}
    for path in sorted(DATA_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in audio_extensions:
            return path
    raise FileNotFoundError(f"No podcast audio file found in {DATA_DIR}")


def markdown_path_for_pdf(pdf_path: Path) -> Path:
    return PROCESSED_DATA_DIR / f"{pdf_path.stem}.md"


def convert_pdf_to_markdown(pdf_path: Path) -> tuple[str, Path]:
    import pymupdf4llm

    markdown_text = pymupdf4llm.to_markdown(str(pdf_path))
    if not markdown_text.strip():
        raise ValueError(f"Markdown conversion produced no text for {pdf_path}")

    markdown_path = markdown_path_for_pdf(pdf_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_text, encoding="utf-8")
    return markdown_text, markdown_path


def transcript_text_path_for_audio(audio_path: Path) -> Path:
    return PROCESSED_DATA_DIR / f"{audio_path.stem}_transcript.txt"


def transcript_segments_path_for_audio(audio_path: Path) -> Path:
    return PROCESSED_DATA_DIR / f"{audio_path.stem}_transcript_segments.txt"


def transcribe_audio(audio_path: Path) -> tuple[str, Path, Path]:
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError(
            "pydub is required for audio transcription. Install it with: "
            "./.conda/bin/pip install pydub"
        ) from exc

    client = OpenAI(api_key=OPENAI_API_KEY)
    audio = AudioSegment.from_file(audio_path)
    chunk_length_ms = AUDIO_CHUNK_SECONDS * 1000
    chunks = [
        audio[i:i + chunk_length_ms]
        for i in range(0, len(audio), chunk_length_ms)
    ]

    all_segments = []
    for chunk_index, chunk in enumerate(chunks):
        chunk_offset_seconds = chunk_index * AUDIO_CHUNK_SECONDS
        buffer = BytesIO()
        chunk.export(buffer, format="mp3")
        buffer.seek(0)
        buffer.name = f"{audio_path.stem}_chunk_{chunk_index + 1}.mp3"

        transcript = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=buffer,
            language="en",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

        for segment in transcript.segments:
            all_segments.append(
                {
                    "start": segment.start + chunk_offset_seconds,
                    "end": segment.end + chunk_offset_seconds,
                    "text": segment.text.strip(),
                }
            )

    transcript_text = " ".join(segment["text"] for segment in all_segments).strip()
    if not transcript_text:
        raise ValueError(f"Audio transcription produced no text for {audio_path}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    transcript_text_path = transcript_text_path_for_audio(audio_path)
    transcript_segments_path = transcript_segments_path_for_audio(audio_path)

    transcript_text_path.write_text(transcript_text, encoding="utf-8")
    transcript_segments_path.write_text(
        "\n".join(
            f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}"
            for segment in all_segments
        ),
        encoding="utf-8",
    )

    return transcript_text, transcript_text_path, transcript_segments_path


def chunk_markdown(markdown_text: str, pdf_path: Path, markdown_path: Path) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    section_docs = header_splitter.split_text(markdown_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(section_docs)

    for index, chunk in enumerate(chunks):
        chunk.page_content = chunk.page_content.strip()
        chunk.metadata.update(
            {
                "source_file": pdf_path.name,
                "source_type": "pdf",
                "language": "en",
                "chunking_strategy": "markdown_headers_recursive",
                "markdown_file": str(markdown_path.relative_to(PROJECT_ROOT)),
                "chunk_index": index,
            }
        )

    return chunks


def chunk_transcript_text(
    transcript_text: str,
    audio_path: Path,
    transcript_path: Path,
) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.create_documents([transcript_text])

    for index, chunk in enumerate(chunks):
        chunk.page_content = chunk.page_content.strip()
        chunk.metadata.update(
            {
                "source_file": audio_path.name,
                "source_type": "podcast_transcript",
                "language": "en",
                "chunking_strategy": "recursive_transcript",
                "transcript_file": str(transcript_path.relative_to(PROJECT_ROOT)),
                "chunk_index": index,
            }
        )

    return chunks


def preview_chunks(chunks: list[Document], markdown_path: Path) -> None:
    print("\nStep 1 - Data preparation")
    print(f"Markdown saved to: {markdown_path.relative_to(PROJECT_ROOT)}")
    print(f"Chunks created: {len(chunks)}")
    print("\nFirst 3 chunks:")
    for chunk in chunks[:3]:
        print(f"\nChunk {chunk.metadata['chunk_index']}")
        print(f"Metadata: {chunk.metadata}")
        print(chunk.page_content[:350].replace("\n", " "))


def preview_transcript_chunks(chunks: list[Document], transcript_path: Path, segments_path: Path) -> None:
    print("\nStep 1 - Audio transcription")
    print(f"Transcript saved to: {transcript_path.relative_to(PROJECT_ROOT)}")
    print(f"Timestamped segments saved to: {segments_path.relative_to(PROJECT_ROOT)}")
    print(f"Transcript chunks created: {len(chunks)}")
    print("\nFirst 2 transcript chunks:")
    for chunk in chunks[:2]:
        print(f"\nChunk {chunk.metadata['chunk_index']}")
        print(f"Metadata: {chunk.metadata}")
        print(chunk.page_content[:350].replace("\n", " "))


def ensure_pinecone_index() -> None:
    from pinecone import Pinecone, ServerlessSpec

    pinecone = Pinecone(api_key=PINECONE_API_KEY)
    if pinecone.has_index(PINECONE_INDEX_NAME):
        return

    print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
    pinecone.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )

    for _ in range(60):
        description = pinecone.describe_index(PINECONE_INDEX_NAME)
        status = getattr(description, "status", {}) or {}
        is_ready = status.get("ready") if isinstance(status, dict) else getattr(status, "ready", False)
        if is_ready:
            return
        time.sleep(1)

    raise TimeoutError(f"Pinecone index {PINECONE_INDEX_NAME} was not ready in time.")


def build_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSION,
        api_key=OPENAI_API_KEY,
    )


def build_vectorstore() -> PineconeVectorStore:
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=build_embeddings(),
        pinecone_api_key=PINECONE_API_KEY,
        namespace=PINECONE_NAMESPACE,
    )


def ingest_chunks(chunks: list[Document]) -> None:
    print("\nStep 2 - Embeddings and Pinecone upload")
    ensure_pinecone_index()
    vectorstore = build_vectorstore()

    ids = [
        f"{chunk.metadata['source_file']}:{chunk.metadata['chunking_strategy']}:{chunk.metadata['chunk_index']}"
        for chunk in chunks
    ]
    vectorstore.add_documents(chunks, ids=ids)
    print(f"Uploaded {len(chunks)} chunks to Pinecone index '{PINECONE_INDEX_NAME}'.")


def retrieve(question: str, top_k: int = 5, metadata_filter: dict | None = None) -> list[tuple[Document, float]]:
    vectorstore = build_vectorstore()
    return vectorstore.similarity_search_with_score(
        question,
        k=top_k,
        filter=metadata_filter,
    )


def print_retrieval_results(question: str, results: list[tuple[Document, float]], title: str) -> None:
    print(f"\n{title}")
    print(f"Question: {question}")
    for rank, (doc, similarity_score) in enumerate(results, start=1):
        print(f"\nResult {rank} | similarity score: {similarity_score:.4f}")
        print(f"Metadata: {doc.metadata}")
        print(doc.page_content[:400].replace("\n", " "))


def score_relevance(question: str, results: list[tuple[Document, float]]) -> list[dict]:
    llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    scorer = RELEVANCE_PROMPT | llm.with_structured_output(RelevanceScore)

    scored_results = []
    for original_rank, (doc, similarity_score) in enumerate(results, start=1):
        relevance = scorer.invoke(
            {
                "question": question,
                "metadata": doc.metadata,
                "chunk_text": doc.page_content[:2500],
            }
        )
        combined_score = (0.7 * (relevance.score / 5)) + (0.3 * similarity_score)
        scored_results.append(
            {
                "document": doc,
                "original_rank": original_rank,
                "similarity_score": similarity_score,
                "relevance_score": relevance.score,
                "combined_score": combined_score,
                "reason": relevance.reason,
            }
        )

    return sorted(
        scored_results,
        key=lambda item: (
            -item["combined_score"],
            -item["relevance_score"],
            -item["similarity_score"],
        ),
    )


def print_scored_results(question: str, scored_results: list[dict], top_k: int = 3) -> None:
    print("\nStep 3 - LLM relevance scoring")
    print(f"Question: {question}")
    for rank, item in enumerate(scored_results[:top_k], start=1):
        doc = item["document"]
        print(
            f"\nScored result {rank} | relevance: {item['relevance_score']}/5 "
            f"| combined: {item['combined_score']:.4f} "
            f"| similarity: {item['similarity_score']:.4f} "
            f"| original rank: {item['original_rank']}"
        )
        print(f"Reason: {item['reason']}")
        print(f"Metadata: {doc.metadata}")
        print(doc.page_content[:400].replace("\n", " "))


def answer_with_context(question: str, scored_results: list[dict], top_k: int = 3) -> str:
    context_parts = []
    for rank, item in enumerate(scored_results[:top_k], start=1):
        doc = item["document"]
        context_parts.append(
            "\n".join(
                [
                    f"[Chunk {rank}]",
                    f"Source: {doc.metadata.get('source_file')}",
                    f"Section: {doc.metadata.get('Header 2', doc.metadata.get('Header 1', 'Unknown'))}",
                    doc.page_content,
                ]
            )
        )

    llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    response = (ANSWER_PROMPT | llm).invoke(
        {
            "question": question,
            "context": "\n\n---\n\n".join(context_parts),
        }
    )
    return str(response.content)


def run_lab() -> None:
    require_api_keys()

    pdf_path = find_english_pdf()
    markdown_text, markdown_path = convert_pdf_to_markdown(pdf_path)
    pdf_chunks = chunk_markdown(markdown_text, pdf_path, markdown_path)
    preview_chunks(pdf_chunks, markdown_path)

    audio_path = find_podcast_audio()
    transcript_text, transcript_path, segments_path = transcribe_audio(audio_path)
    transcript_chunks = chunk_transcript_text(transcript_text, audio_path, transcript_path)
    preview_transcript_chunks(transcript_chunks, transcript_path, segments_path)

    all_chunks = pdf_chunks + transcript_chunks

    ingest_chunks(all_chunks)

    for question in QUESTIONS:
        baseline_results = retrieve(question, top_k=5)
        print_retrieval_results(question, baseline_results[:3], "Step 2 - Baseline vector retrieval")

        scored_results = score_relevance(question, baseline_results)
        print_scored_results(question, scored_results, top_k=3)

        print("\nStep 5 - Metadata filtering example")
        filtered_results = retrieve(
            question,
            top_k=3,
            metadata_filter={"source_type": "podcast_transcript", "language": "en"},
        )
        print_retrieval_results(
            question,
            filtered_results,
            "Filtered retrieval: source_type=podcast_transcript, language=en",
        )

        print("\nStep 6 - Simple RAG answer using top scored chunks")
        print(answer_with_context(question, scored_results, top_k=3))

        print("\nStep 7 - Manual comparison note")
        print(
            "Compare the baseline result order above with the relevance-scored order. "
            "The relevance score adds a human-readable reason and a combined score "
            "for judging whether reranking helped."
        )


if __name__ == "__main__":
    run_lab()
