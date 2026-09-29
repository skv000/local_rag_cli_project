import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from config import RAGConfig, DEFAULT_CONFIG
from core.rag_engine import RAGEngine


# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# Terminal color styling
class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_banner():
    banner = f"""
{Colors.CYAN}{Colors.BOLD}===================================================================
      Custom RAG from Scratch (Local Python + Ollama llama3.2)
==================================================================={Colors.RESET}
{Colors.DIM}No LangChain | No External DBs | 100% Local Inference & Vectors{Colors.RESET}
"""
    print(banner)


def format_source_citation(result, index: int) -> str:
    path_name = Path(result.chunk.source).name
    score_pct = f"{result.score * 100:.1f}%"
    snippet = result.chunk.text.replace("\n", " ").strip()
    if len(snippet) > 140:
        snippet = snippet[:140] + "..."
    return (
        f"  {Colors.YELLOW}[{index}]{Colors.RESET} {Colors.BOLD}{path_name}{Colors.RESET} "
        f"({Colors.GREEN}{score_pct} match{Colors.RESET})\n"
        f"      {Colors.DIM}\"{snippet}\"{Colors.RESET}"
    )


def cmd_ingest(engine: RAGEngine, path: str):
    target = Path(path)
    if not target.exists():
        print(f"{Colors.RED}[Error] Path does not exist: {path}{Colors.RESET}")
        return

    print(f"\n{Colors.BOLD}Ingesting documents from:{Colors.RESET} {target.resolve()}")
    start_time = time.time()
    try:
        count = engine.ingest_path(target)
        elapsed = time.time() - start_time
        print(f"\n{Colors.GREEN}[OK] Done! Indexed {count} chunk(s) in {elapsed:.2f} seconds.{Colors.RESET}")
        print(f"Index persisted to: {Colors.DIM}{engine.index_path}{Colors.RESET}\n")
    except Exception as e:
        print(f"\n{Colors.RED}[Error] Ingestion failed: {e}{Colors.RESET}")


def cmd_query(engine: RAGEngine, question: str, stream: bool = True, top_k: Optional[int] = None):
    if len(engine.vector_store) == 0:
        print(f"{Colors.YELLOW}[Notice] Vector store is empty! Ingest some documents first:{Colors.RESET}")
        print("  python main.py ingest data/sample_docs\n")
        return

    print(f"\n{Colors.CYAN}{Colors.BOLD}Query:{Colors.RESET} {question}")
    print(f"{Colors.DIM}Searching knowledge base (top {top_k or engine.config.top_k} matches)...{Colors.RESET}")

    start_time = time.time()
    response, search_results = engine.query(question, stream=stream, top_k=top_k)

    print(f"\n{Colors.MAGENTA}{Colors.BOLD}--- Retrieved Sources ({len(search_results)}) ---{Colors.RESET}")
    if not search_results:
        print(f"  {Colors.DIM}(No documents met the similarity threshold){Colors.RESET}")
    else:
        for idx, res in enumerate(search_results, start=1):
            print(format_source_citation(res, idx))

    print(f"\n{Colors.GREEN}{Colors.BOLD}--- Response ({engine.config.llm_model}) ---{Colors.RESET}")
    if stream:
        for token in response:
            sys.stdout.write(token)
            sys.stdout.flush()
        print()
    else:
        print(response)

    elapsed = time.time() - start_time
    print(f"\n{Colors.DIM}[Completed in {elapsed:.2f}s]{Colors.RESET}\n")


def cmd_chat(engine: RAGEngine):
    if len(engine.vector_store) == 0:
        print(f"{Colors.YELLOW}[Notice] Vector store is currently empty.{Colors.RESET}")
        print("You can still ask questions, or ingest documents using:")
        print(f"  {Colors.BOLD}python main.py ingest data/sample_docs{Colors.RESET}\n")

    print(f"{Colors.CYAN}{Colors.BOLD}=== Interactive RAG Chat Session ==={Colors.RESET}")
    print(f"Model: {Colors.BOLD}{engine.config.llm_model}{Colors.RESET} | Embeddings: {Colors.BOLD}{engine.config.embedding_model}{Colors.RESET}")
    print(f"Indexed Chunks: {Colors.BOLD}{len(engine.vector_store)}{Colors.RESET}")
    print(f"{Colors.DIM}Type 'exit' or 'quit' to end. Type '!sources' after an answer to see chunk snippets.{Colors.RESET}\n")

    last_results = []

    while True:
        try:
            user_input = input(f"{Colors.BOLD}{Colors.BLUE}You > {Colors.RESET}").strip()
            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "q"}:
                print(f"{Colors.DIM}Goodbye!{Colors.RESET}")
                break

            if user_input.lower() == "!sources":
                if not last_results:
                    print(f"{Colors.DIM}No previous search results available.{Colors.RESET}\n")
                else:
                    print(f"\n{Colors.MAGENTA}{Colors.BOLD}--- Sources for previous question ---{Colors.RESET}")
                    for idx, res in enumerate(last_results, start=1):
                        print(format_source_citation(res, idx))
                    print()
                continue

            response_gen, results = engine.query(user_input, stream=True)
            last_results = results

            print(f"\n{Colors.GREEN}{Colors.BOLD}llama3.2 > {Colors.RESET}", end="", flush=True)
            for token in response_gen:
                sys.stdout.write(token)
                sys.stdout.flush()
            print("\n")

            if results:
                sources_str = ", ".join(f"{Path(r.chunk.source).name} ({r.score * 100:.0f}%)" for r in results)
                print(f"{Colors.DIM}  [Sources: {sources_str}] (type '!sources' for full context){Colors.RESET}\n")

        except KeyboardInterrupt:
            print(f"\n{Colors.DIM}Session ended.{Colors.RESET}")
            break
        except Exception as e:
            print(f"\n{Colors.RED}[Error] {e}{Colors.RESET}\n")


def cmd_stats(engine: RAGEngine):
    stats = engine.vector_store.get_stats()
    print(f"\n{Colors.BOLD}Vector Store Statistics:{Colors.RESET}")
    print(f"  - Indexed Chunks:      {stats['total_chunks']}")
    print(f"  - Unique Documents:    {stats['total_documents']}")
    print(f"  - Total Characters:    {stats['total_characters']}")
    print(f"  - NumPy Acceleration:  {'Enabled' if stats['has_numpy_acceleration'] else 'Pure Python'}")
    print(f"  - Storage Path:        {engine.index_path}")
    if stats["unique_sources"]:
        print(f"\n{Colors.BOLD}Ingested Sources:{Colors.RESET}")
        for src in stats["unique_sources"]:
            print(f"  - {Path(src).name} ({src})")
    print()


def cmd_clear(engine: RAGEngine):
    engine.clear_index()
    print(f"{Colors.YELLOW}Vector store index cleared.{Colors.RESET}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Custom RAG from Scratch using Python and local Ollama llama3.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    parser_ingest = subparsers.add_parser("ingest", help="Ingest a file or directory into vector store")
    parser_ingest.add_argument("path", help="Path to document file or directory")

    # Query command
    parser_query = subparsers.add_parser("query", help="Query the RAG pipeline with a question")
    parser_query.add_argument("question", help="The question to ask")
    parser_query.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser_query.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")

    # Chat command
    subparsers.add_parser("chat", help="Start an interactive multi-turn chat session")

    # Stats command
    subparsers.add_parser("stats", help="Show vector store status and indexed documents")

    # Clear command
    subparsers.add_parser("clear", help="Clear all indexed documents from vector store")

    args = parser.parse_args()

    print_banner()

    engine = RAGEngine(DEFAULT_CONFIG)

    # Health check
    ready, message = engine.check_health()
    if not ready:
        print(f"{Colors.RED}[Connection Warning] {message}{Colors.RESET}")
        print("Please ensure Ollama is running and models are installed:")
        print("  ollama run llama3.2:3b")
        print("  ollama pull embeddinggemma:latest\n")
        if args.command not in {"stats", "clear"}:
            sys.exit(1)

    if args.command == "ingest":
        cmd_ingest(engine, args.path)
    elif args.command == "query":
        cmd_query(engine, args.question, stream=not args.no_stream, top_k=args.top_k)
    elif args.command == "chat":
        cmd_chat(engine)
    elif args.command == "stats":
        cmd_stats(engine)
    elif args.command == "clear":
        cmd_clear(engine)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
