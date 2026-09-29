import math
import shutil
import tempfile
from pathlib import Path

from config import RAGConfig
from core.document import Document, Chunk
from core.loader import DocumentLoader
from core.ollama_client import OllamaClient
from core.splitter import RecursiveTextSplitter
from core.vector_store import VectorStore, pure_cosine_similarity
from core.rag_engine import RAGEngine


import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_tests():
    print("========================================")
    print("Running Custom RAG System Unit & E2E Tests")
    print("========================================\n")

    # 1. Test Cosine Similarity Math
    print("[1/6] Testing Cosine Similarity Math...")
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    v4 = [-1.0, 0.0, 0.0]

    assert math.isclose(pure_cosine_similarity(v1, v2), 1.0, abs_tol=1e-5), "Identical vectors must equal 1.0"
    assert math.isclose(pure_cosine_similarity(v1, v3), 0.0, abs_tol=1e-5), "Orthogonal vectors must equal 0.0"
    assert math.isclose(pure_cosine_similarity(v1, v4), -1.0, abs_tol=1e-5), "Opposite vectors must equal -1.0"
    print("  [OK] Pure cosine similarity math matches theoretical expectations.")

    # 2. Test Recursive Text Splitter
    print("[2/6] Testing Recursive Text Splitter...")
    splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)
    sample_text = (
        "Artificial intelligence is transforming industries. "
        "From healthcare diagnostics to natural language processing, "
        "machine learning models enable autonomous decision making. "
        "Vector search allows semantic information retrieval at scale."
    )
    chunks = splitter.split_text(sample_text)
    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
    for idx, c in enumerate(chunks):
        assert len(c) <= 120, f"Chunk {idx} exceeded max length: {len(c)}"
    print(f"  [OK] Split {len(sample_text)} chars into {len(chunks)} overlapping chunks.")

    # 3. Test Document Loader
    print("[3/6] Testing Document Loader...")
    sample_dir = Path("data/sample_docs")
    docs = DocumentLoader.load_directory(sample_dir)
    assert len(docs) >= 3, f"Expected at least 3 sample documents, got {len(docs)}"
    sources = [Path(d.source).name for d in docs]
    print(f"  [OK] Loaded {len(docs)} documents: {sources}")

    # 4. Test Ollama Client Connection & Embedding Generation
    print("[4/6] Testing Local Ollama API...")
    client = OllamaClient()
    connected = client.check_connection()
    assert connected, "Ollama server is not reachable on localhost:11434"
    models = client.list_models()
    print(f"  [OK] Ollama online. Found {len(models)} local models.")

    emb = client.get_embedding("Testing RAG vector embedding", model="embeddinggemma:latest")
    assert len(emb) == 768, f"Expected 768-dim vector, got {len(emb)}"
    print(f"  [OK] Successfully generated 768-dim embedding via embeddinggemma.")

    # 5. Test Vector Store Persistence & Retrieval
    print("[5/6] Testing Vector Store Persistence & Search...")
    temp_dir = Path(tempfile.mkdtemp())
    try:
        vs = VectorStore()
        chunk1 = Chunk(text="NASA Orion capsule", doc_id="d1", source="artemis.md", chunk_index=0, embedding=[1.0, 0.0])
        chunk2 = Chunk(text="Superconducting qubits", doc_id="d2", source="quantum.txt", chunk_index=0, embedding=[0.0, 1.0])
        vs.add_chunks([chunk1, chunk2])

        # Test search
        res = vs.search([0.9, 0.1], top_k=1)
        assert len(res) == 1
        assert res[0].chunk.doc_id == "d1", f"Expected d1, got {res[0].chunk.doc_id}"

        # Test save & load
        save_path = temp_dir / "test_index.json"
        vs.save(save_path)
        assert save_path.exists(), "Saved index file must exist"

        vs2 = VectorStore()
        vs2.load(save_path)
        assert len(vs2) == 2, f"Loaded vector store should have 2 chunks, got {len(vs2)}"
        res2 = vs2.search([0.1, 0.9], top_k=1)
        assert res2[0].chunk.doc_id == "d2"
        print("  [OK] Vector store save, reload, and cosine similarity search passed.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # 6. Test End-to-End RAG Ingestion & Query
    print("[6/6] Testing End-to-End RAG Engine...")
    test_storage = Path(tempfile.mkdtemp())
    try:
        cfg = RAGConfig(storage_dir=test_storage, chunk_size=300, chunk_overlap=50)
        engine = RAGEngine(config=cfg)

        # Ingest sample docs
        num_chunks = engine.ingest_path("data/sample_docs/artemis_mission.md")
        assert num_chunks > 0, "No chunks were indexed from artemis_mission.md"

        # Query
        test_question = "What rocket launches the Orion spacecraft?"
        answer, sources = engine.query(test_question, stream=False)
        print(f"  Question: '{test_question}'")
        print(f"  Retrieved {len(sources)} source chunk(s). Top score: {sources[0].score * 100:.1f}%")
        print(f"  Generated Answer snippet: {answer[:120]}...")
        assert len(sources) > 0, "Expected at least 1 retrieved source"
        assert len(answer.strip()) > 0, "Expected non-empty answer"
        print("  [OK] End-to-End RAG query pipeline passed.")
    finally:
        shutil.rmtree(test_storage, ignore_errors=True)

    print("\n========================================")
    print("ALL 6 TESTS PASSED SUCCESSFULLY! [OK]")
    print("========================================")


if __name__ == "__main__":
    run_tests()
